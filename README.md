# 📜 Circular Repository — React + FastAPI + Qwen2.5 (CPU-first)

RAG question-answering over official circulars, now with a React frontend,
multiple chat sessions with a sidebar, and a one-line CPU→GPU switch.

## Architecture
```
frontend/   React 18 + Vite (Node.js tooling) — sidebar, multi-chat, dashboard
backend/    FastAPI (Python) — ingestion, ChromaDB, Phi-3 via Ollama
```
The React app talks to the backend at `/api/*`; APA reference links resolve
to `/files/*` (the stored circular PDFs). The answering pipeline is identical
to the Streamlit version, so responses are the same.

## Features
- **OCR language-mismatch detection** — if OCR runs without the right language pack installed (e.g. Marathi scans OCR'd as English-only), Tesseract doesn't error, it silently produces gibberish text. The app now detects this (near-zero Devanagari output when Devanagari was expected) and flags the document `ocr_suspect` on the dashboard instead of indexing it as if it were read correctly — with a direct explanation of the fix. Recovers automatically via Rebuild search index / Re-scan references once the language pack is installed.
- **Added explicit language-matching rule to the prompt** — the system prompt previously had zero instruction about which language to answer in, or how to handle a source excerpt whose script/language differs from the question. This can cause a model to silently translate the excerpt into English internally, reason there, then translate back into the answer language — each step compounding error. Now instructs the model to answer in the same language as the question and ground directly in the source's actual wording, with numbers/dates/circular numbers/proper names always copied unchanged regardless of answer language.
- **Fixed: model fabricating citations to unrelated real institutions** — the system prompt's example citation used a concrete, plausible-sounding real name ("Reserve Bank of India, 2024") to illustrate the APA format. Weaker/smaller models can and do literally parrot a memorized few-shot example back as a real answer when uncertain, producing confidently wrong citations attributing content to organizations that have nothing to do with the actual document (e.g. a Maharashtra Police inspection circular cited as "Reserve Bank of India"). Fixed by replacing every concrete example with an unmistakably abstract placeholder ("ISSUER-PLACEHOLDER, 19XX") and adding an explicit rule to copy the issuer/year verbatim from the real `[Source (Issuer, Year) — filename]` tag above each excerpt — never from memory, never from an example. Applied to both the chat and template-generation prompts.
- **Fixed: issuer detection picking up repeated page headers** — for multi-page documents with a running header/footer printed on every page (e.g. "DGP STANDING ORDERS/CIRCULARS/INVESTIGATION"), that header was being taken as the "issuer" since it's literally the first line of extracted text. Now any line repeating 3+ times across the document is recognized as page furniture and skipped in favor of the actual first content line.
- **Fixed: circular number left with a redundant "NO." prefix** — a document's own number like "Circular No. 42" was being stored as "NO.42" instead of a clean "42", because label-stripping had a length safety-net that (correctly) protected against over-aggressive stripping elsewhere, but (incorrectly) also blocked stripping short-but-genuine own numbers. Stripping is now unconditional, with the general minimum-length floor lowered from 4 to 2 characters so short real numbers survive — while the existing dedicated filter for overly generic *citations* (added for the "Circular No. 1" case) still independently keeps bare short numbers from being flagged as missing references.
- **Cross-reference checking can be enabled/disabled (Admin page)** — matching citation formats across decades of inconsistent historical documents doesn't scale the way it does for newly authored circulars that include an explicit "Reference"/"संदर्भ" line. When enabled (default), the dashboard shows missing-reference flags and new uploads are scanned for citations. When disabled, that dashboard section is hidden entirely and new uploads skip citation extraction — but a document's own circular number is always still detected regardless of this setting, since chat citations depend on it. Existing already-uploaded documents' stored references are left as-is until a re-scan; toggling only changes behavior going forward.
- **Fixed: overly generic English citations flagged as "missing references"** — English-style circular numbering ("Circular No. 1", "Circular No. 42") had two bugs: the label word "No." could leak into the captured identifier (same root cause as an earlier Marathi क्र fix), and a bare, unstructured number with no slash/year/department code (e.g. a citation to a decades-old superseded circular) was flagged as a missing reference even though it's too generic to be actionable. Citations without any distinguishing structure are now excluded — but a document's own number in the same bare form (e.g. "Circular No. 42" as this document's own identity) is untouched, since that filter only applies to citations of *other* documents.
- **Repository search + multi-select bulk delete** — a search box above the repository table filters by filename, circular number, issuer, title, or status; admins can now select multiple documents via checkboxes (with a select-all) and delete them in one action instead of one at a time. The bulk-delete request runs reference resolution and cache invalidation once at the end rather than per document, so deleting many stays fast.
- **Fixed: legitimate English attachments wrongly flagged as broken OCR** — a document consisting mostly of a Marathi cover circular plus a large genuinely-English attachment (e.g. a full court judgment enclosed with the circular) was being flagged `ocr_suspect`, because its overall Devanagari ratio is naturally low when English pages dominate by volume. The detector now distinguishes real English prose (checked via common function-word frequency — "the", "of", "court", "state", etc.) from OCR gibberish that merely looks Latin-shaped ("WH. WA/2HAT-BS-AATH"). Only the latter — actual language-pack failures — gets quarantined; a legitimately mixed-language document is now correctly indexed.
- **Fixed: irrelevant documents cited alongside a correct one** — retrieval previously always returned the "best available" chunks regardless of whether they were actually relevant, and every source among them got a reference card. Now each retrieved chunk must clear a relevance bar (an absolute floor + a floor relative to the best match for that specific question — see `RERANK_MIN_SCORE` / `RERANK_MAX_GAP` in config) before it can contribute to the answer or its references. If nothing clears the bar, the app honestly says nothing relevant was found rather than citing the closest-available unrelated document. Reference cards in chat now also show a **relevance score**, so you can visually verify each citation actually matched the question.
- **Fixed: multi-word Marathi phrases inside circular numbers** — real government file numbers sometimes embed a two-word descriptive phrase as one path segment (e.g. `पोमसं/23/54/साखळी चोरी/380/2016`, where "साखळी चोरी" = "chain theft" is two separate words). Detection previously split at the space, truncating the real number and turning the truncated remainder into a fake "reference". Fixed with a principled rule: multi-word joining only applies to non-numeric segments (descriptive phrases), never to purely numeric segments (serial numbers, years) — so trailing sentence words after a number (e.g. "...2016 अन्वये...") are never swallowed, verified against a 20+ case regression suite.
- **Fixed: scanned-PDF image corruption before OCR** — PDF pages were rasterized (PyMuPDF) and handed to Tesseract via a raw pixel-buffer reconstruction (`Image.frombytes`) that assumes an exact width×height×3 byte layout with zero row padding. That assumption isn't always safe, and when it breaks, the resulting image is silently scrambled — Tesseract still produces confident-looking output, just wrong, regardless of language pack. Fixed by encoding each page to PNG and letting PIL decode it (PyMuPDF's own recommended approach), which is immune to this. If your Marathi documents specifically fail as scanned PDFs but plain image uploads (PNG/JPG) work fine, this was almost certainly the cause.
- **Live OCR configuration check (Admin page)** — shows OCR_LANG as configured vs. the languages actually visible to the *running backend process* (queried live via pytesseract). Checking `tesseract --list-langs` in a terminal only proves the OS has the pack — it does not prove the backend process (which may run under a different user, container, or venv, or may have started before the pack was installed) can see it. This catches that mismatch directly instead of relying on manual shell checks.
- **OCR text-sample diagnostic** — every processed document (indexed or flagged) now stores a preview of its extracted text. Failed/suspect documents in the dashboard have a "view extracted text" toggle so you can see exactly what OCR produced — the fastest way to tell a genuinely missing language pack (gibberish) from a false-positive quality flag (real text, just numeric/English-heavy).
- **Manual circular-number correction (admin)** — click ✎ next to any document's circular number in the dashboard to set or fix it directly. Immediately re-resolves any citations pointing to it. Use this when OCR or detection can't reliably read a degraded scan.
- **Scrollable list windows** — upload/read failures, missing references, and the repository table are now fixed-height scrollable panels instead of growing the page indefinitely as your repository grows.
- **Document templates (Inspection Schedule, Panchnama, FIR, custom)** — generates a fillable Markdown template for a document type, grounded in whatever format your circulars actually prescribe (cited in APA, same as chat). If no format-prescribing circular exists in the repository, it says so explicitly and drafts a clearly labeled generic, non-mandated version instead of inventing an "official" one. Preview in-app, then export as a real .docx with headings, tables, and bold fields intact.
- **Tabular answers** — when a question calls for comparing rates, slabs, dates, or eligibility criteria across categories, the model formats that part as a Markdown table, rendered as a real HTML table in the chat (not raw `|pipes|`). Simple factual questions still get plain text — tables appear only when the content is genuinely tabular.
- **Login + RBAC** — roles: `admin` (users, reindex, everything), `uploader` (upload + chat), `viewer` (chat + dashboard). Chats are private per user. Default login `admin / admin123` — change it on the Admin page. Set a real `SECRET_KEY` in production: `SECRET_KEY=$(openssl rand -hex 32)`.
- **Header + auto-hide sidebar** — title bar with engine/mode chip and user badge; sidebar collapses with ☰ and auto-hides as an overlay on narrow windows.
- **Hybrid retrieval + reranking** — BM25 keyword search (exact circular numbers, clause terms) fused with vector search (RRF), then a cross-encoder reranks candidates before they reach Phi-3. Markedly better accuracy, still CPU-friendly. Toggle with `HYBRID_ENABLED=0` / `RERANK_ENABLED=0`; multilingual reranker for Marathi: `RERANK_MODEL=cross-encoder/mmarco-mMiniLMv2-L12-H384-v1`.
- **Rebuild search index (Admin page)** — re-embeds all stored documents with the current `EMBED_MODEL`, no re-upload needed. This is the correct way to “train” the system on your repository: knowledge lives in the index, the LLM stays frozen (fine-tuning an LLM on circulars invites hallucinated circular numbers and doesn't speed retrieval).
- **Multiple chats + sidebar** — create, switch, auto-titled from the first question, delete; full history persisted in SQLite.
- **CPU by default, GPU by env var** — see below.
- **APA clickable references** — rendered as catalog cards; click opens the source PDF.
- **OCR for scanned documents** — PDFs without a text layer are rasterized (PyMuPDF) and read with Tesseract; JPG/PNG/TIFF scans can be uploaded directly. Indexed items show "read via OCR".
- **Delete circulars (admin)** — removes the file, its search chunks and its cross-references in one click from the dashboard; circulars that cited the deleted one are automatically re-flagged as missing references.
- **Strict duplicate rejection** — a file identical (SHA-256) to one already in the repository is refused at upload with the name of the existing file; nothing is stored or recorded.
- **Upload/read failure detection** — corrupt, password-protected, unreadable-scan or duplicate files flagged per-file and on the dashboard with an actionable reason.
- **Cross-reference check** — circular numbers cited in bodies but missing from the repository are flagged; auto-resolve on later upload.

## Setup

### 1. LLM (Ollama)
```bash
ollama pull qwen2.5:7b     # default model (~4.7 GB download, ~6 GB RAM resident)
ollama serve               # usually auto-starts
```
Low-RAM machine? Use a lighter model instead: `ollama pull phi3` then start
the backend with `LLM_MODEL=phi3`. Any Ollama model works via `LLM_MODEL`.

### 2. Tesseract OCR (for scanned PDFs / images)
```bash
# Ubuntu/Debian
sudo apt install tesseract-ocr
# macOS
brew install tesseract
# Windows: https://github.com/UB-Mannheim/tesseract/wiki
```
Optional env vars: `OCR_LANG=eng+hin` (languages, install matching packs e.g.
`tesseract-ocr-hin`), `OCR_DPI=300` (accuracy vs speed), `OCR_MAX_PAGES=60`,
`OCR_ENABLED=0` to turn OCR off. If Tesseract is missing, scanned files are
flagged on the dashboard with install instructions instead of failing silently.

> ⚠️ **`OCR_LANG` defaults to `eng` only.** If your circulars are scanned
> Marathi documents (very common with "Print to PDF" exports that embed a
> font with no usable character map — the PDF *looks* like it has a text
> layer but extracts to nothing), you MUST start the backend with
> `OCR_LANG=mar+eng` and have `tesseract-ocr-mar` installed (see the
> [Marathi + English section](#marathi--english-multilingual-support)
> below). If you forget, OCR does **not** error — it silently forces
> Devanagari glyphs into English letter-shapes and produces gibberish like
> `WH. WA/2HAT-BS-AATH/200` instead of real text. The app detects this
> automatically (documents are flagged `ocr_suspect` on the dashboard
> instead of indexed as if correct), but the fix is still on you: install
> the language pack, set the env var, then use **Admin → Rebuild search
> index** or **Re-scan references** to recover already-uploaded documents
> without re-uploading them.

### 3. Backend (Python 3.10+)
```bash
cd backend
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
uvicorn main:app --reload --port 8000
```

### 4. Frontend (Node 18+)
```bash
cd frontend
npm install          # adds react-markdown + remark-gfm for table rendering
npm run dev         # open http://localhost:5173
```
Upgrading an existing install? Just re-run `npm install` in `frontend/` to
pick up the two new packages.

## CPU ↔ GPU switching
Everything runs **CPU-only by default**. To enable GPU, set env vars before
starting the backend — no code changes:

```bash
# Full GPU (NVIDIA CUDA):
EMBED_DEVICE=cuda OLLAMA_NUM_GPU=-1 uvicorn main:app --port 8000

# Apple Silicon (embeddings on Metal; Ollama uses Metal automatically):
EMBED_DEVICE=mps OLLAMA_NUM_GPU=-1 uvicorn main:app --port 8000

# Partial offload (e.g. 20 LLM layers on GPU, rest on CPU):
OLLAMA_NUM_GPU=20 uvicorn main:app --port 8000

# Pin CPU threads for faster CPU inference (e.g. 8 cores):
LLM_THREADS=8 uvicorn main:app --port 8000
```
- `EMBED_DEVICE` — device for the sentence-transformers embedder (`cpu` | `cuda` | `cuda:0` | `mps`). GPU embeddings need a CUDA build of PyTorch: `pip install torch --index-url https://download.pytorch.org/whl/cu121`
- `OLLAMA_NUM_GPU` — LLM layers offloaded to GPU: `0` = CPU-only, `-1` = all layers, `N` = partial.
- The sidebar shows the active mode (CPU/GPU) live from `/api/health`.

## Production build
```bash
cd frontend && npm run build     # outputs frontend/dist
```
Serve `dist/` with any static server (or nginx) and point `/api` + `/files`
at the backend.

## Avoiding timeouts (CPU performance)
Answers now **stream** token-by-token (NDJSON), so no request in the
browser→proxy→backend→Ollama chain ever idles long enough to time out, and
you see the answer forming immediately. Additional latency controls (env vars):

| Variable | Default | Effect |
|---|---|---|
| `KEEP_ALIVE` | `30m` | Keeps Phi-3 loaded in RAM between questions (a cold reload costs 30–90 s — the classic "first question times out") |
| `WARMUP` | `1` | Preloads embedder, reranker and LLM at backend startup |
| `NUM_PREDICT` | `1800` | Caps answer length (generation time is linear in tokens; lower it if answers feel slow) |
| `NUM_CTX` | `4096` | Context window; smaller = faster CPU prefill |
| `PROMPT_CHUNK_CHARS` | `1500` | Trims each retrieved excerpt in the prompt (raise further if answers still feel incomplete) |
| `LLM_THREADS` | auto | Set to your **physical** core count (not hyperthreads) |
| `LLM_TIMEOUT` | `600` | Backend read timeout, seconds |

If it's still too slow on your hardware: try `LLM_MODEL=phi3:mini` /
`gemma2:2b`, reduce `TOP_K` to 3–4, or enable GPU offload
(`OLLAMA_NUM_GPU=-1`). Also check RAM — if the model doesn't fit, the OS
swaps to disk and everything crawls.

## Why no LLM fine-tuning?
Fine-tuning Phi-3 on circulars teaches it *style*, not reliably retrievable
*facts* — and a fine-tuned model will confidently invent circular numbers it
half-remembers. Retrieval speed and accuracy come from the index instead:
embeddings are precomputed at upload, BM25 catches exact identifiers,
and the reranker (a small cross-encoder) picks the truly relevant chunks.
If you later want domain adaptation, the right target is the *embedder*
(e.g. LoRA on the sentence-transformer with query→passage pairs), not Phi-3.

## Marathi + English (multilingual) support
Three switches, no code changes:

```bash
# 1. Marathi OCR pack for scanned circulars
sudo apt install tesseract-ocr-mar          # macOS: brew tesseract includes all langs

# 2. Start the backend with multilingual settings
OCR_LANG=mar+eng uvicorn main:app --port 8000   # embedder/reranker are multilingual by default now
```
- **Retrieval**: the multilingual embedder handles Marathi and English questions
  against Marathi/English circulars. ⚠️ Re-upload documents after changing the
  embedding model — vectors from different models are not comparable (delete
  `backend/data/chroma` first).
- **Reference detection** understands `परिपत्रक क्र. १२/२०२४` and converts
  Devanagari digits, so १२/२०२४ matches a circular numbered 12/2024.
- **Answer language**: Phi-3 is English-first — it answers *about* Marathi
  circulars in English well, but for answers *in* Marathi switch the model:
  `LLM_MODEL=qwen2.5 ollama pull qwen2.5` (or `gemma2`), which handle Indic
  languages better. Test both on your own circulars.

## Customising reference detection
Circular-number formats vary by organisation — edit `REF_PATTERNS` in
`backend/references.py`.

## Fixed (properly, this time): placeholder text leaking into real citations
An ironic self-inflicted regression: an earlier fix replaced a risky
example citation ("Reserve Bank of India, 2024") with a deliberately
abstract placeholder ("ISSUER-PLACEHOLDER, 19XX") specifically so the
model couldn't parrot a real institution name — but the model parroted
the NEW placeholder text just as readily, producing literal
"(ISSUER-PLACEHOLDER, 19XX)"-style citations. Any literal example text in
the prompt can get echoed back, no matter how obviously fake it looks to
a human. Fixed properly this time: removed every literal example citation
string from both prompts entirely, replaced with a pure description of
the substitution mechanism ("read the real value after issuer=/year= and
write those two real values") plus an explicit ban on the exact strings
that leaked before. Also added a code-level safety net — `rag.py` now
scans every generated answer for known leak markers and logs a loud
warning if any appear, since prompt wording alone has proven unreliable
against this specific failure three times running. Check your backend
console for `[rag] WARNING: answer contains leaked prompt/placeholder
text` if you ever suspect this is happening again.

## Fixed: same question giving a different answer each time
The chat LLM's sampling `temperature` was set to 0.1 — low, but not zero,
meaning generation still involved some randomness in picking each next
token rather than always choosing the most likely one. For a tool
answering questions about official circulars, reproducibility matters:
the same question should give the same answer, both for user trust and
for auditability. Now defaults to `temperature=0` (deterministic/greedy
decoding) plus a fixed `seed=42` as a second layer of determinism.
Configurable via `LLM_TEMPERATURE` / `LLM_SEED` if you specifically want
varied phrasing across repeated asks. Note this only affects the LLM's
own generation — if you still see variation after this, the more likely
remaining cause is retrieval returning different excerpts across asks;
check the relevance scores/raw-excerpts viewer to confirm the same
sources are being used each time.

## Fixed: Marathi conversation history biasing later English answers
Once a turn in a chat got translated to Marathi, the STORED "content" for
that message was the Marathi text shown to the user. Every LATER question
in that same chat included that Marathi text as conversation history in
the prompt — meaning the chat LLM was reasoning with Marathi-heavy context
regardless of what language the NEW question was in. This explained two
distinct symptoms at once: the model drifting into broken, repetitive
Marathi even when explicitly told to answer in English, and a plain
English question producing a Marathi answer. Fixed: conversation history
sent to the chat LLM now always uses the English original (`content_en`)
for translated turns, never the displayed Marathi — the chat LLM should
always reason in English internally; translation is a separate, downstream
step. Verified directly: a chat with a translated Marathi turn followed by
a new question now sends pure English history to the model.

## Fixed: closing-summary sentences inventing placeholder citations
A separate bug, same root symptom class: despite an existing "no closing
summary" rule, the model sometimes still wrote one — and when it did, it
invented meaningless placeholder citations like `(X, 1966; Y, 1981)`
instead of real issuer names, apparently because it couldn't recall which
specific source supported a compressed summary statement. Strengthened the
rule to explicitly prohibit placeholder-letter citations and to frame not
writing a summary as the direct way to avoid the problem.

## Fixed: Marathi answers much shorter than English
IndicTrans2 is a *sentence-level* translation model, but the app was
treating each entire Markdown paragraph — which the chat LLM writes as
multiple sentences — as ONE translation unit, capped at 256 tokens for
both input and output. Long paragraphs were silently truncated before
translation even happened. Fixed: paragraphs (and bullets, and table
cells) are now split into individual sentences before translation and
rejoined afterward, and the length cap was also raised to 384 as a second
safety net. **This requires redeploying `translation.py` and
`translation_worker.py`** — no changes to the isolated venv itself, no
re-run of `setup_translation_venv.sh` needed.

## Translation runs in an isolated environment
IndicTrans2's custom model code needs an OLDER `transformers` version than
the embedder/reranker do (discovered through real deployment debugging:
a removed `transformers.onnx` module, a changed internal generation-cache
format, and more). Rather than risk that pin destabilizing the reranker/
embedder, translation runs as a **separate subprocess in its own isolated
venv** (`translation_venv/`), talking to the main app over stdin/stdout.
Nothing in the main app's environment changes.

**One-time setup:**
```bash
cd backend
bash setup_translation_venv.sh
```
This creates `translation_venv/`, installs pinned dependencies (starting
point: `transformers==4.41.2`, may still need adjustment — see below), and
patches a known `IndicTransToolkit` 1.1.1 bug automatically.

**If the model still fails to load after setup**, the pinned version in
`setup_translation_venv.sh` may need adjusting — do this entirely inside
the isolated venv, with zero risk to the rest of the app:
```bash
translation_venv/bin/pip install "transformers==<some other version>"
python3 translation_worker.py ai4bharat/indictrans2-en-indic-dist-200M cpu
# watch for {"ready": true} on its own line
```

## Diagnosing translation problems
If chat shows "Translation model unavailable," don't guess — go to
**Admin → 🌐 Translation model (live check) → Test translation model**. This
actually attempts to load the model right now and shows the REAL underlying
error (missing dependency, version mismatch, network failure downloading
the model, etc.) instead of a generic message. The same real error is also
printed to the backend console log and included in the chat-side warning.
`IndicTransToolkit` is a young library — its own README warns "you may
encounter broken stuff and import issues once in a while" — so seeing the
exact error matters more than usual here.

## Marathi translation setup (optional)
Disabled by default — enable only if you've confirmed (via the raw-excerpts
viewer in chat) that your Marathi answer quality problem is generation-side,
not an OCR/source-document problem.

```bash
pip install IndicTransToolkit   # transformers/torch already present via sentence-transformers
TRANSLATION_ENABLED=1 uvicorn main:app --port 8000
```
- `TRANSLATION_ENABLED` — off by default; turns the feature on.
- `TRANSLATION_MODEL` — default `ai4bharat/indictrans2-en-indic-dist-200M` (CPU-practical; the 1B variants exist but are much heavier).
- `TRANSLATION_DEVICE` — `cpu` by default; set to `cuda` if you have a GPU.
- `TRANSLATION_TRIGGER_RATIO` — fraction of Devanagari letters in a question needed to trigger translation mode (default 0.3).

First request after enabling downloads the model (~500MB) — expect a delay
the first time. If the dependency isn't installed or the model can't load,
the app automatically falls back to showing the English answer with a note,
rather than failing.
