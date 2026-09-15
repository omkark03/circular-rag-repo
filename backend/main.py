"""FastAPI backend with RBAC.

Run:  uvicorn main:app --reload --port 8000
Default login: admin / admin123  — change it on the Admin page immediately.

Roles: admin (everything) · uploader (upload + chat) · viewer (chat only).
"""
import io
import json
import re
import threading
import traceback
import zipfile

from fastapi import Depends, FastAPI, Header, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

import auth
import config
import docx_export
import references as refx
import ingest
import rag
import scheduler
import translation
from datetime import date
import retrieval
import store

app = FastAPI(title="Circular Repository API")
store.init_db()
auth.ensure_default_admin()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)
app.mount("/files", StaticFiles(directory=config.FILES_DIR), name="files")


def _warmup():
    try:
        ingest.get_embedder().encode(["warmup"])
        retrieval._get_reranker()
        import requests as _rq
        _rq.post(config.OLLAMA_URL, json={
            "model": config.LLM_MODEL, "prompt": "Hi", "stream": False,
            "keep_alive": config.KEEP_ALIVE,
            "options": {"num_predict": 1}}, timeout=180)
    except Exception:
        pass  # warmup is best-effort


@app.on_event("startup")
def startup():
    if config.WARMUP:
        threading.Thread(target=_warmup, daemon=True).start()


# ---------------------------------------------------------------- auth deps

def current_user(authorization: str = Header(default="")) -> dict:
    if not authorization.startswith("Bearer "):
        raise HTTPException(401, "Not authenticated")
    payload = auth.decode_token(authorization[7:])
    if not payload:
        raise HTTPException(401, "Invalid or expired token")
    user = store.get_user(payload["u"])
    if not user:
        raise HTTPException(401, "User no longer exists")
    return {"id": user["id"], "username": user["username"], "role": user["role"]}


def require(*roles):
    def dep(user: dict = Depends(current_user)) -> dict:
        if user["role"] not in roles:
            raise HTTPException(403, f"Requires role: {' or '.join(roles)}")
        return user
    return dep


# ---------------------------------------------------------------- models

class LoginBody(BaseModel):
    username: str
    password: str


class UserBody(BaseModel):
    username: str
    password: str
    role: str


class PasswordBody(BaseModel):
    password: str


class AskBody(BaseModel):
    question: str


class RenameBody(BaseModel):
    title: str


class TestDetectBody(BaseModel):
    text: str


class TemplateGenBody(BaseModel):
    doc_type: str            # preset key ("panchnama", "fir", ...) or free text
    details: str = ""        # optional extra context, e.g. "warehouse inspection"


class ScheduleGenBody(BaseModel):
    stations: list[str]      # real station names, e.g. "Andheri PS [important]"
    details: str = ""        # optional extra context, e.g. "District: Mumbai"


class StationInput(BaseModel):
    name: str
    important: bool = False


class ExactScheduleGenBody(BaseModel):
    stations: list[StationInput]
    start_date: str                    # ISO format YYYY-MM-DD
    working_weekdays: list[int] | None = None  # 0=Mon..6=Sun; default Mon-Sat
    holidays: list[str] | None = None  # ISO date strings to skip


class ScheduleDatesGenBody(BaseModel):
    stations: list[str]                     # "Name" or "Name [important]"
    start_date: str                         # "YYYY-MM-DD"
    working_weekdays: list[int] = [0, 1, 2, 3, 4, 5]  # Mon=0 .. Sun=6; default Sun off
    holidays: list[str] = []                # ["YYYY-MM-DD", ...]
    window_days: int = 365
    gap_days: int = 0
    # Manual overrides: if BOTH given, skips LLM rule-extraction entirely.
    # If only one/neither given, missing values are extracted from the
    # repository — and if still not found, the request fails clearly
    # rather than guessing.
    regular_duration_days: int | None = None
    important_duration_days: int | None = None
    details: str = ""


class TemplateExportBody(BaseModel):
    title: str
    markdown: str


class CircularNoBody(BaseModel):
    circular_no: str  # empty string clears it


class BulkDeleteBody(BaseModel):
    ids: list[int]


class CrossRefToggleBody(BaseModel):
    enabled: bool


# ---------------------------------------------------------------- auth API

@app.post("/api/auth/login")
def login(body: LoginBody):
    user = auth.authenticate(body.username.strip(), body.password)
    if not user:
        raise HTTPException(401, "Invalid username or password")
    return {"token": auth.create_token(user["username"], user["role"]),
            "username": user["username"], "role": user["role"]}


@app.get("/api/auth/me")
def me(user: dict = Depends(current_user)):
    return user


@app.post("/api/auth/password")
def change_password(body: PasswordBody, user: dict = Depends(current_user)):
    if len(body.password) < 6:
        raise HTTPException(400, "Password must be at least 6 characters")
    store.update_user_password(user["username"], auth.hash_password(body.password))
    return {"ok": True}


# ---------------------------------------------------------------- admin API

@app.get("/api/admin/users")
def users(_: dict = Depends(require("admin"))):
    return store.list_users()


@app.post("/api/admin/users")
def add_user(body: UserBody, _: dict = Depends(require("admin"))):
    if body.role not in auth.ROLES:
        raise HTTPException(400, f"Role must be one of {auth.ROLES}")
    if store.get_user(body.username.strip()):
        raise HTTPException(400, "Username already exists")
    if len(body.password) < 6:
        raise HTTPException(400, "Password must be at least 6 characters")
    store.create_user(body.username.strip(), auth.hash_password(body.password), body.role)
    return {"ok": True}


@app.delete("/api/admin/users/{user_id}")
def remove_user(user_id: int, admin: dict = Depends(require("admin"))):
    if user_id == admin["id"]:
        raise HTTPException(400, "You cannot delete your own account")
    store.delete_user(user_id)
    return {"ok": True}


@app.post("/api/admin/reindex")
def reindex(_: dict = Depends(require("admin"))):
    """Re-embed the whole repository from stored files. Use after changing
    EMBED_MODEL, or after fixing OCR_LANG/installing a language pack — this
    fully re-runs extraction (including the OCR quality check), so documents
    quarantined as 'ocr_suspect' are retried and recover automatically if
    the underlying issue is fixed."""
    import chromadb
    client = chromadb.PersistentClient(path=config.CHROMA_DIR)
    try:
        client.delete_collection("circulars")
    except Exception:
        pass
    ingest._collection = None
    ingest._embedder = None

    done, suspect, failed = 0, 0, 0
    for d in store.all_documents():
        if d["status"] not in ("ok", "ocr_suspect") or not d["stored_name"]:
            continue
        path = config.FILES_DIR / d["stored_name"]
        outcome = ingest.finalize_document(d["id"], d["filename"], d["stored_name"], path)
        if outcome["status"] == "ok":
            done += 1
        elif outcome["status"] == "ocr_suspect":
            suspect += 1
        else:
            failed += 1
    store.resolve_pending_references()
    retrieval.invalidate()
    return {"reindexed": done, "still_suspect": suspect, "failed": failed,
            "embed_model": config.EMBED_MODEL}


@app.post("/api/admin/rescan-references")
def rescan_references(_: dict = Depends(require("admin"))):
    """Re-run text extraction, circular-number detection, and citation
    extraction on every stored document (including ones quarantined as
    'ocr_suspect') with the current patterns/config. No re-upload needed."""
    done, suspect, failed = 0, 0, 0
    for d in store.all_documents():
        if d["status"] not in ("ok", "ocr_suspect") or not d["stored_name"]:
            continue
        path = config.FILES_DIR / d["stored_name"]
        outcome = ingest.finalize_document(d["id"], d["filename"], d["stored_name"], path)
        if outcome["status"] == "ok":
            done += 1
        elif outcome["status"] == "ocr_suspect":
            suspect += 1
        else:
            failed += 1
    resolved = store.resolve_pending_references()
    return {"rescanned": done, "still_suspect": suspect, "failed": failed,
            "resolved": resolved}


@app.get("/api/admin/settings/cross-reference")
def get_cross_reference_setting(_: dict = Depends(require("admin"))):
    return {"enabled": store.cross_reference_check_enabled()}


@app.post("/api/admin/settings/cross-reference")
def set_cross_reference_setting(body: CrossRefToggleBody,
                                _: dict = Depends(require("admin"))):
    """Enable/disable cross-reference checking dashboard-wide. When enabled,
    the dashboard shows missing-reference flags and newly uploaded documents
    are scanned for citations. When disabled, the dashboard section is
    hidden and new uploads skip citation extraction — matching historical
    documents with inconsistent numbering doesn't scale, but is reliable
    for newly authored circulars that include an explicit Reference/संदर्भ
    line. Existing already-uploaded documents are unaffected until
    re-scanned or re-uploaded."""
    store.set_cross_reference_check_enabled(body.enabled)
    return {"enabled": body.enabled}


@app.post("/api/admin/translation/test")
def test_translation(_: dict = Depends(require("admin"))):
    """Actually attempts to load the translation model right now (retrying
    even a previously-cached failure) and reports the real error if it
    fails — same live-diagnostic pattern as the OCR configuration check,
    so you don't have to dig through server logs to see what's wrong."""
    translation.reset_load_state()
    sample = translation.translate_markdown_to_marathi(
        "The police station must be inspected once a year.")
    if sample is not None:
        return {"ok": True, "sample_translation": sample}
    return {"ok": False, "error": translation.load_error() or "unknown error"}


@app.get("/api/admin/backup")
def download_backup(_: dict = Depends(require("admin"))):
    """Download a full backup: the SQLite database (users, chats, documents,
    references, settings) plus every stored circular file — everything
    needed to restore the repository elsewhere. The vector index is
    intentionally excluded (Admin -> Rebuild search index recreates it
    quickly from the files + database, and it can be large/model-specific)."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        if config.DB_PATH.exists():
            zf.write(config.DB_PATH, arcname="repository.db")
        for f in config.FILES_DIR.iterdir():
            if f.is_file():
                zf.write(f, arcname=f"circulars/{f.name}")
    buf.seek(0)
    from datetime import datetime
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    return StreamingResponse(
        iter([buf.getvalue()]),
        media_type="application/zip",
        headers={"Content-Disposition":
                f'attachment; filename="circular-repository-backup-{stamp}.zip"'},
    )


@app.post("/api/admin/test-detection")
def test_detection(body: TestDetectBody, _: dict = Depends(require("admin", "uploader"))):
    """Paste any circular text and see exactly what the detector finds —
    use this to report undetected number formats."""
    text = body.text
    own = refx.detect_own_number(text)
    refs = [{"as_found": raw, "normalized": norm}
            for raw, norm in refx.split_body_references(text, own)]
    return {"own_number": own, "references": refs}


# ---------------------------------------------------------------- system

@app.get("/api/health")
def health():
    configured_langs = set(config.OCR_LANG.split("+"))
    available_langs = ingest.available_ocr_languages()
    missing = sorted(configured_langs - set(available_langs)) if available_langs else []
    return {"ok": True, "llm": config.LLM_MODEL,
            "embed_device": config.EMBED_DEVICE,
            "embed_model": config.EMBED_MODEL.split("/")[-1],
            "hybrid": config.HYBRID_ENABLED, "rerank": config.RERANK_ENABLED,
            "ocr_enabled": config.OCR_ENABLED, "ocr_lang": config.OCR_LANG,
            "ocr_available_langs": available_langs,
            "ocr_lang_missing": missing,
            "translation_enabled": config.TRANSLATION_ENABLED,
            "contradiction_detection_enabled": config.CONTRADICTION_DETECTION_ENABLED,
            "translation_venv_python": config.TRANSLATION_VENV_PYTHON,
            "mode": "GPU" if (config.OLLAMA_NUM_GPU != 0 or
                              config.EMBED_DEVICE != "cpu") else "CPU"}


# ---------------------------------------------------------------- uploads

@app.post("/api/upload")
async def upload(files: list[UploadFile],
                 _: dict = Depends(require("admin", "uploader"))):
    results = []
    for f in files:
        data = await f.read()
        results.append(ingest.ingest_upload(f.filename, data))
    store.resolve_pending_references()
    retrieval.invalidate()
    return {"results": results}


# ---------------------------------------------------------------- dashboard

@app.get("/api/dashboard")
def dashboard(_: dict = Depends(current_user)):
    xref_enabled = store.cross_reference_check_enabled()
    return {
        "stats": store.stats(),
        "failed": [dict(r) for r in store.failed_documents()],
        "missing_references": [dict(r) for r in store.unresolved_references()] if xref_enabled else [],
        "documents": [dict(r) for r in store.all_documents()],
        "cross_reference_check_enabled": xref_enabled,
    }


@app.delete("/api/documents/{doc_id}")
def delete_document(doc_id: int, _: dict = Depends(require("admin"))):
    row = ingest.delete_document(doc_id)
    if not row:
        raise HTTPException(404, "Document not found")
    store.resolve_pending_references()
    retrieval.invalidate()
    return {"ok": True, "deleted": row["filename"]}


@app.post("/api/documents/bulk-delete")
def bulk_delete_documents(body: BulkDeleteBody, _: dict = Depends(require("admin"))):
    """Delete multiple documents at once (repository multi-select). Runs
    reference resolution and cache invalidation once at the end rather than
    once per document, so deleting many at a time stays fast."""
    deleted, not_found = [], []
    for doc_id in body.ids:
        row = ingest.delete_document(doc_id)
        (deleted if row else not_found).append(doc_id)
    if deleted:
        store.resolve_pending_references()
        retrieval.invalidate()
    return {"deleted": deleted, "not_found": not_found}


@app.patch("/api/documents/{doc_id}/circular-no")
def set_circular_no(doc_id: int, body: CircularNoBody,
                    _: dict = Depends(require("admin"))):
    """Manually set/correct a document's own circular number. Use this when
    automatic detection misses or mis-reads it (common with degraded scans)
    — other circulars that cite this number will resolve immediately."""
    row = store.get_document(doc_id)
    if not row:
        raise HTTPException(404, "Document not found")
    value = body.circular_no.strip() or None
    norm = refx.normalize(value) if value else None
    store.update_document(doc_id, circular_no=norm)
    resolved = store.resolve_pending_references()
    return {"ok": True, "circular_no": norm, "resolved": resolved}


@app.post("/api/dashboard/recheck")
def recheck(_: dict = Depends(require("admin", "uploader"))):
    return {"resolved": store.resolve_pending_references()}


# ---------------------------------------------------------------- chats (per-user)

def _own_chat(chat_id: int, user: dict):
    if store.chat_owner(chat_id) != user["id"]:
        raise HTTPException(404, "Chat not found")


@app.get("/api/chats")
def chats(user: dict = Depends(current_user)):
    return store.list_chats(user["id"])


@app.post("/api/chats")
def new_chat(user: dict = Depends(current_user)):
    return store.create_chat(user["id"])


@app.get("/api/chats/{chat_id}")
def chat_detail(chat_id: int, user: dict = Depends(current_user)):
    _own_chat(chat_id, user)
    return {"messages": store.chat_messages(chat_id)}


@app.patch("/api/chats/{chat_id}")
def rename(chat_id: int, body: RenameBody, user: dict = Depends(current_user)):
    _own_chat(chat_id, user)
    store.rename_chat(chat_id, body.title)
    return {"ok": True}


@app.delete("/api/chats/{chat_id}")
def remove(chat_id: int, user: dict = Depends(current_user)):
    _own_chat(chat_id, user)
    store.delete_chat(chat_id)
    return {"ok": True}


def _format_schedule_chat_response(extracted: dict, schedule_result: dict | None) -> str:
    """Builds the chat-visible markdown for an inline schedule request.
    ALWAYS shows what was extracted from the user's own message first —
    station names, importance flags, start date — so a misparse from the
    (inherently less reliable) chat-text extraction step is immediately
    visible before any table, rather than silently baked into what looks
    like an authoritative schedule."""
    lines = []
    station_list = ", ".join(
        f"{s['name']}{' (important)' if s['important'] else ''}"
        for s in extracted["stations"]
    )
    lines.append(f"**Detected from your message:** {len(extracted['stations'])} "
                f"station(s) — {station_list}")
    if extracted.get("start_date"):
        lines.append(f"**Start date:** {extracted['start_date']}")
    if extracted.get("details"):
        lines.append(f"**Context:** {extracted['details']}")
    lines.append("")
    lines.append("_If anything above is wrong, retype your request more "
                "explicitly (list each station clearly, state the date "
                "plainly) rather than relying on this auto-detection for "
                "something operationally important — or use the Templates "
                "page's dedicated form, which never has this parsing step._")
    lines.append("")

    if schedule_result is None:
        lines.append("I found station names but no clear start date in your "
                     "message. Please specify one, e.g. \"starting 1 April 2027\".")
        return "\n".join(lines)

    if not schedule_result["ok"]:
        lines.append(f"⚠️ {schedule_result['error']}")
        if schedule_result.get("partial_schedule"):
            lines.append("\n| Station | Category | Start Date | End Date | Duration |")
            lines.append("|---|---|---|---|---|")
            for r in schedule_result["partial_schedule"]:
                lines.append(
                    f"| {r['station']} | {'Important' if r['important'] else 'Regular'} "
                    f"| {r['start_date']} | {r['end_date']} | {r['duration_days']} day(s) |")
            lines.append(f"\n**Stations that do NOT fit:** "
                        f"{', '.join(schedule_result['stations_that_dont_fit'])}")
        return "\n".join(lines)

    lines.append("| Station | Category | Start Date | End Date | Duration |")
    lines.append("|---|---|---|---|---|")
    for r in schedule_result["schedule"]:
        lines.append(
            f"| {r['station']} | {'Important' if r['important'] else 'Regular'} "
            f"| {r['start_date']} | {r['end_date']} | {r['duration_days']} day(s) |")
    return "\n".join(lines)


def _gen_normal_chat_answer(chat_id: int, q: str, history: list[dict]):
    """The standard RAG chat-answer generator: retrieval -> LLM streaming
    -> optional translation -> optional contradiction-check -> persist.
    Factored out as its own function so both the normal chat path AND the
    inline-schedule-detection fallback (when the schedule heuristic
    false-triggers, i.e. finds no real station names) can reuse the exact
    same logic via `yield from` rather than duplicate it.

    The whole body runs inside a try/except: by the time this generator
    starts yielding, the HTTP response has already committed its 200 OK
    headers, so an unhandled exception here doesn't turn into a clean
    error response -- it kills the connection mid-stream, which browsers
    surface as an opaque "Error in input stream" with no useful detail.
    Catching everything and yielding a normal error token instead keeps
    the stream well-formed no matter what fails."""
    try:
        yield from _gen_normal_chat_answer_inner(chat_id, q, history)
    except Exception as e:
        traceback.print_exc()
        yield json.dumps({
            "type": "token",
            "t": f"\n\n⚠️ Something went wrong while generating this answer: {e}",
        }) + "\n"
        yield json.dumps({"type": "done"}) + "\n"


def _gen_normal_chat_answer_inner(chat_id: int, q: str, history: list[dict]):
    prompt, refs, hits = rag.build_context(q, history=history)
    needs_translation = (config.TRANSLATION_ENABLED
                         and translation.is_devanagari_question(q))

    yield json.dumps({"type": "refs", "references": refs,
                      "hits": [{"filename": h["meta"]["filename"],
                                "score": h.get("score", 0),
                                "text": h["text"][:800]} for h in hits]}) + "\n"
    if prompt is None:
        store.add_message(chat_id, "assistant", rag.EMPTY_MSG, [])
        yield json.dumps({"type": "token", "t": rag.EMPTY_MSG}) + "\n"
        yield json.dumps({"type": "done"}) + "\n"
        return
    pieces = []
    for piece in rag.stream_phi3(prompt):
        pieces.append(piece)
        yield json.dumps({"type": "token", "t": piece}) + "\n"
    english = "".join(pieces)
    rag._check_for_leaked_markers(english)

    final_content, content_en = english, None
    if needs_translation:
        yield json.dumps({"type": "translating"}) + "\n"
        translated = translation.translate_markdown_to_marathi(english)
        if translated is not None:
            final_content, content_en = translated, english
            yield json.dumps({"type": "translation", "text": translated}) + "\n"
        else:
            reason = translation.load_error()
            yield json.dumps({
                "type": "translation_unavailable",
                "detail": ("Translation model unavailable — showing the "
                          "English answer. Actual error: "
                          + (reason or "unknown — check the backend console log")),
            }) + "\n"

    contradictions = []
    if config.CONTRADICTION_DETECTION_ENABLED and hits:
        yield json.dumps({"type": "checking_contradictions"}) + "\n"
        contradictions = rag.detect_contradictions(hits)
        if contradictions:
            yield json.dumps({"type": "contradictions",
                              "items": contradictions}) + "\n"

    store.add_message(chat_id, "assistant", final_content, refs, content_en,
                      contradictions)
    store.touch_chat(chat_id)
    yield json.dumps({"type": "done"}) + "\n"


@app.post("/api/chats/{chat_id}/ask")
def ask(chat_id: int, body: AskBody, user: dict = Depends(current_user)):
    _own_chat(chat_id, user)
    q = body.question.strip()
    if not q:
        raise HTTPException(400, "Empty question")

    history = store.chat_history_for_llm(chat_id)
    store.add_message(chat_id, "user", q)

    titles = {c["id"]: c["title"] for c in store.list_chats(user["id"])}
    if titles.get(chat_id) == "New chat":
        store.rename_chat(chat_id, q[:60])

    schedule_mode = config.INLINE_SCHEDULE_DETECTION_ENABLED and rag.looks_like_schedule_request(q)

    if schedule_mode:
        def gen_schedule():
            try:
                yield from _gen_schedule_inner(chat_id, q, history)
            except Exception as e:
                traceback.print_exc()
                yield json.dumps({
                    "type": "token",
                    "t": f"\n\n⚠️ Something went wrong while generating this schedule: {e}",
                }) + "\n"
                yield json.dumps({"type": "done"}) + "\n"

        return StreamingResponse(gen_schedule(), media_type="application/x-ndjson",
                                 headers={"X-Accel-Buffering": "no",
                                          "Cache-Control": "no-cache"})

    return StreamingResponse(_gen_normal_chat_answer(chat_id, q, history),
                             media_type="application/x-ndjson",
                             headers={"X-Accel-Buffering": "no",
                                      "Cache-Control": "no-cache"})


def _gen_schedule_inner(chat_id: int, q: str, history: list[dict]):
    # Streaming starts HERE, immediately -- before any LLM call.
    # The previous version called rag.extract_schedule_request(q)
    # BEFORE returning StreamingResponse at all, meaning the
    # entire HTTP response (not even headers) was blocked on that
    # LLM call completing, then a SECOND sequential LLM call
    # inside _run_scheduler ran with zero feedback either --
    # exactly the pattern that trips proxy/browser timeouts, with
    # nothing sent to the client the whole time.
    yield json.dumps({"type": "extracting_schedule"}) + "\n"
    schedule_extracted = rag.extract_schedule_request(q)
    if not schedule_extracted["stations"]:
        # heuristic false-triggered -- fall through to a normal
        # answer instead of forcing a broken schedule interaction
        yield from _gen_normal_chat_answer_inner(chat_id, q, history)
        return

    stations = schedule_extracted["stations"]
    schedule_result = None
    refs = []
    if schedule_extracted["start_date"]:
        yield json.dumps({"type": "checking_duration_rules"}) + "\n"
        start = date.fromisoformat(schedule_extracted["start_date"])
        schedule_result = _run_scheduler(
            stations, start, schedule_extracted.get("details", ""),
            original_message=q)
        refs = schedule_result.get("references", [])
    content = _format_schedule_chat_response(schedule_extracted, schedule_result)
    yield json.dumps({"type": "refs", "references": refs, "hits": []}) + "\n"
    yield json.dumps({"type": "token", "t": content}) + "\n"
    store.add_message(chat_id, "assistant", content, refs)
    store.touch_chat(chat_id)
    yield json.dumps({"type": "done"}) + "\n"


# ---------------------------------------------------------------- templates

@app.get("/api/templates/presets")
def template_presets(_: dict = Depends(current_user)):
    return [{"key": k, "label": v} for k, v in rag.TEMPLATE_PRESETS.items()]


@app.post("/api/templates/generate")
def generate_template(body: TemplateGenBody, _: dict = Depends(current_user)):
    """Streams a fillable document template (NDJSON: refs -> tokens -> done),
    grounded in whatever the repository's circulars actually prescribe."""
    doc_type = body.doc_type.strip()
    if not doc_type:
        raise HTTPException(400, "doc_type is required")
    label = rag.TEMPLATE_PRESETS.get(doc_type, doc_type)
    prompt, refs, hits = rag.build_template_context(label, body.details)

    def gen():
        yield json.dumps({"type": "refs", "references": refs,
                          "hits": [{"filename": h["meta"]["filename"],
                                    "score": h.get("score", 0),
                                    "text": h["text"][:800]} for h in hits]}) + "\n"
        for piece in rag.stream_phi3(prompt):
            yield json.dumps({"type": "token", "t": piece}) + "\n"
        yield json.dumps({"type": "done"}) + "\n"

    return StreamingResponse(gen(), media_type="application/x-ndjson",
                             headers={"X-Accel-Buffering": "no",
                                      "Cache-Control": "no-cache"})


@app.post("/api/templates/generate-schedule")
def generate_schedule(body: ScheduleGenBody, _: dict = Depends(current_user)):
    """Streams an actual inspection schedule (NDJSON: refs -> tokens -> done),
    grounded in circular-derived rules (frequency, duration, deadlines) and
    using ONLY the real station names provided — the roster of stations
    under a given jurisdiction isn't something any circular would contain,
    so it must come from the caller, never invented by the model."""
    stations = [s.strip() for s in body.stations if s.strip()]
    if not stations:
        raise HTTPException(400, "At least one station is required")
    prompt, refs, hits = rag.build_schedule_context(stations, body.details)

    def gen():
        yield json.dumps({"type": "refs", "references": refs,
                          "hits": [{"filename": h["meta"]["filename"],
                                    "score": h.get("score", 0),
                                    "text": h["text"][:800]} for h in hits]}) + "\n"
        for piece in rag.stream_phi3(prompt):
            yield json.dumps({"type": "token", "t": piece}) + "\n"
        yield json.dumps({"type": "done"}) + "\n"

    return StreamingResponse(gen(), media_type="application/x-ndjson",
                             headers={"X-Accel-Buffering": "no",
                                      "Cache-Control": "no-cache"})


_IMPORTANT_MARKER = re.compile(r"\s*\[\s*important\s*\]\s*$", re.IGNORECASE)


def _parse_station_list(raw_stations: list[str]) -> list[dict]:
    """'Andheri PS [important]' -> {'name': 'Andheri PS', 'important': True}."""
    parsed = []
    for s in raw_stations:
        s = s.strip()
        if not s:
            continue
        important = bool(_IMPORTANT_MARKER.search(s))
        name = _IMPORTANT_MARKER.sub("", s).strip()
        if name:
            parsed.append({"name": name, "important": important})
    return parsed


def _run_scheduler(stations: list[dict], start_date: date, details: str = "",
                   original_message: str = "",
                   regular_days: int | None = None, important_days: int | None = None,
                   working_weekdays: set[int] | None = None,
                   holidays: set[date] | None = None,
                   window_days: int = 365, gap_days: int = 0) -> dict:
    """Shared core: extract duration rules from the repository (unless
    manually overridden) and compute the actual dated schedule. Used by
    both the Templates page's dedicated endpoint and chat's inline
    schedule detection, so the two paths can never drift apart. Returns a
    dict matching the API response shape directly (dates as ISO strings).

    original_message (chat path only): the user's raw request text. A
    fixed generic query string alone was found to retrieve thematically-
    related-but-not-actually-relevant circulars for some repositories --
    the user's own phrasing carries far more retrieval signal than any
    fixed string can, so it's included whenever available."""
    refs, rules_note = [], ""
    if regular_days is None or important_days is None:
        query_parts = ["inspection schedule frequency duration days minimum"]
        if original_message:
            query_parts.append(original_message)
        if details:
            query_parts.append(details)
        query = " ".join(query_parts).strip()
        hits = rag.retrieve(query)
        rules = rag.extract_schedule_rules(hits)
        regular_days = regular_days or rules["regular_duration_days"]
        important_days = important_days or rules["important_duration_days"]
        rules_note = rules["source_note"]
        refs = [rag.build_apa_reference(store.get_document(h["meta"]["doc_id"]))
               for h in hits if store.get_document(h["meta"]["doc_id"])]

    has_regular = any(not s.get("important") for s in stations)
    has_important = any(s.get("important") for s in stations)

    missing = []
    if has_regular and regular_days is None:
        missing.append("regular_duration_days")
    if has_important and important_days is None:
        missing.append("important_duration_days")
    if missing:
        return {
            "ok": False,
            "error": ("Could not find a specific inspection duration in the "
                     "repository for: " + ", ".join(missing) + ". Either "
                     "upload/verify a circular that states this explicitly, "
                     "or provide " + " and ".join(missing) + " manually."),
            "rules_found": {"regular_duration_days": regular_days,
                           "important_duration_days": important_days,
                           "source_note": rules_note},
        }

    try:
        schedule = scheduler.compute_schedule(
            stations, start_date, regular_days, important_days,
            working_weekdays=working_weekdays or {0, 1, 2, 3, 4, 5},
            holidays=holidays or set(),
            window_days=window_days, gap_days=gap_days,
        )
        return {"ok": True, "schedule": [
                   {**row, "start_date": row["start_date"].isoformat(),
                    "end_date": row["end_date"].isoformat()}
                   for row in schedule],
               "regular_duration_days": regular_days,
               "important_duration_days": important_days,
               "rules_source": rules_note, "references": refs}
    except scheduler.ScheduleOverflowError as e:
        return {"ok": False, "error": str(e),
               "partial_schedule": [
                   {**row, "start_date": row["start_date"].isoformat(),
                    "end_date": row["end_date"].isoformat()}
                   for row in e.partial_schedule],
               "stations_that_dont_fit": e.remaining_stations,
               "regular_duration_days": regular_days,
               "important_duration_days": important_days}


@app.post("/api/templates/generate-schedule-dates")
def generate_schedule_dates(body: ScheduleDatesGenBody,
                            _: dict = Depends(current_user)):
    """Computes an actual dated inspection schedule. Duration rules come
    from the repository (via a narrow, structured LLM extraction — see
    rag.extract_schedule_rules) unless manually overridden; the DATES
    THEMSELVES are always computed deterministically in scheduler.py,
    never generated by the LLM. Returns JSON directly (not streamed) —
    this is fast, structured computation, not token-by-token generation."""
    stations = _parse_station_list(body.stations)
    if not stations:
        raise HTTPException(400, "At least one station is required")

    try:
        start_date = date.fromisoformat(body.start_date)
    except ValueError:
        raise HTTPException(400, f"Invalid start_date '{body.start_date}' — use YYYY-MM-DD")

    try:
        holidays = {date.fromisoformat(h) for h in body.holidays}
    except ValueError as e:
        raise HTTPException(400, f"Invalid holiday date — use YYYY-MM-DD: {e}")

    return _run_scheduler(
        stations, start_date, body.details,
        regular_days=body.regular_duration_days,
        important_days=body.important_duration_days,
        working_weekdays=set(body.working_weekdays), holidays=holidays,
        window_days=body.window_days, gap_days=body.gap_days,
    )


@app.post("/api/templates/export")
def export_template(body: TemplateExportBody, _: dict = Depends(current_user)):
    """Converts a generated (or edited) Markdown template into a downloadable
    .docx file."""
    if not body.markdown.strip():
        raise HTTPException(400, "Nothing to export")
    data = docx_export.markdown_to_docx(body.markdown, title=body.title or "Template")
    safe_name = re.sub(r"[^A-Za-z0-9 _-]", "", body.title or "template")[:60].strip() or "template"
    return StreamingResponse(
        iter([data]),
        media_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        headers={"Content-Disposition": f'attachment; filename="{safe_name}.docx"'},
    )
