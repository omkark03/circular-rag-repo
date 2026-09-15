"""English -> Marathi translation via IndicTrans2 (AI4Bharat), used to avoid
asking the general-purpose chat LLM to generate Marathi directly — testing
showed that produces Hindi/Marathi word-mixing (सुझवले instead of सुचवले),
repetition, and broken grammar regardless of which chat model is used.
Translation is a narrower task than open-ended generation, and a dedicated
NMT model is far more reliable at it for a low-resource language pair.

Model: ai4bharat/indictrans2-en-indic-dist-200M — the distilled 200M variant,
chosen specifically because it's practical on CPU (the 1B variants exist but
are much heavier); this app is CPU-first throughout.

ISOLATION: IndicTrans2's custom model code needs an older transformers
version than the rest of this app (the reranker/embedder need a modern one),
and getting that pin exactly right took several rounds of real deployment
debugging. Rather than risk that pin destabilizing the working reranker/
embedder, translation runs in a SEPARATE, isolated Python venv
(translation_venv/, set up once via setup_translation_venv.sh) as a
persistent subprocess, communicated with over stdin/stdout using
newline-delimited JSON. See translation_worker.py for the subprocess side.

Everything here is lazy-started and OPT-IN (config.TRANSLATION_ENABLED). If
the isolated venv isn't set up, the worker fails to start, or it crashes,
translation is skipped and the English answer is shown as-is with a note —
same graceful-degradation pattern as OCR/reranking elsewhere in this app.
"""
import json
import queue
import re
import subprocess
import threading
from pathlib import Path

import config

_proc = None
_reader_thread = None
_response_queue = None
_proc_lock = threading.Lock()
_load_failed = False
_load_error = None  # human-readable reason, set when loading fails

SRC_LANG = "eng_Latn"
TGT_LANG = "mar_Deva"

# Overridable for testing the subprocess IPC mechanism itself with a fake
# worker (no torch/transformers needed) — production code never changes this.
_WORKER_SCRIPT = str(Path(__file__).parent / "translation_worker.py")

# Generous timeouts: first start includes downloading the ~1.2GB model.
STARTUP_TIMEOUT_SECONDS = 600
REQUEST_TIMEOUT_SECONDS = 120


def load_error() -> str | None:
    """The actual reason the translation worker failed to start/respond, if
    it did. None means either it's working or hasn't been tried yet."""
    return _load_error


def reset_load_state():
    """Clear a cached failure so the next call re-attempts starting the
    worker — used by the Admin 'test translation' action after fixing an
    issue, without needing a full backend restart."""
    global _load_failed, _load_error, _proc
    _load_failed = False
    _load_error = None
    if _proc is not None:
        try:
            _proc.kill()
        except Exception:
            pass
        _proc = None


def _devanagari_ratio(text: str) -> float:
    letters = [c for c in text if c.isalpha()]
    if not letters:
        return 0.0
    dev = sum(1 for c in letters if "\u0900" <= c <= "\u097F")
    return dev / len(letters)


def is_devanagari_question(text: str) -> bool:
    """True when a question is written in Devanagari script (Marathi/Hindi
    etc.) rather than English — the trigger for translating the answer."""
    if len(text.strip()) < 2:
        return False
    return _devanagari_ratio(text) >= config.TRANSLATION_TRIGGER_RATIO


def _reader_loop(proc, q):
    """Runs in a background thread: continuously reads lines from the
    worker's stdout and pushes them onto a queue, so the caller can do a
    timed wait instead of a blocking readline() that could hang forever."""
    for line in iter(proc.stdout.readline, ""):
        q.put(line)
    q.put(None)  # signals stdout closed (worker exited)


def _ensure_worker():
    """Start the isolated translation worker subprocess if not already
    running. Returns True if it's ready to receive requests, False if
    unavailable (venv missing, worker crashed, etc.) — callers degrade
    gracefully rather than crashing."""
    global _proc, _reader_thread, _response_queue, _load_failed, _load_error
    if _load_failed:
        return False
    if _proc is not None and _proc.poll() is None:
        return True  # already running

    python = config.TRANSLATION_VENV_PYTHON
    worker = _WORKER_SCRIPT

    if not Path(python).exists():
        _load_error = (
            f"Isolated translation venv not found at {python}. Run "
            f"setup_translation_venv.sh first (see README)."
        )
        _load_failed = True
        print(f"[translation] {_load_error}", flush=True)
        return False

    try:
        proc = subprocess.Popen(
            [python, worker, config.TRANSLATION_MODEL, config.TRANSLATION_DEVICE],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, bufsize=1,
        )
    except Exception as e:
        _load_error = f"Failed to start translation worker process: {e}"
        _load_failed = True
        print(f"[translation] {_load_error}", flush=True)
        return False

    q = queue.Queue()
    t = threading.Thread(target=_reader_loop, args=(proc, q), daemon=True)
    t.start()

    try:
        line = q.get(timeout=STARTUP_TIMEOUT_SECONDS)
    except queue.Empty:
        proc.kill()
        _load_error = (
            f"Translation worker did not become ready within "
            f"{STARTUP_TIMEOUT_SECONDS}s (model download/load taking too long, "
            f"or hung). Check network access to huggingface.co on this server."
        )
        _load_failed = True
        print(f"[translation] {_load_error}", flush=True)
        return False

    if line is None or line == "":
        # stdout closed / process exited before printing anything -> read
        # stderr for the real reason
        err = proc.stderr.read()
        _load_error = f"Worker exited before becoming ready: {err.strip()[-3000:]}"
        _load_failed = True
        print(f"[translation] {_load_error}", flush=True)
        return False

    try:
        msg = json.loads(line)
    except Exception:
        _load_error = f"Worker sent unexpected startup output: {line.strip()[:500]}"
        _load_failed = True
        print(f"[translation] {_load_error}", flush=True)
        return False

    if not msg.get("ready"):
        _load_error = msg.get("error", "unknown worker startup error")[:3000]
        _load_failed = True
        print(f"[translation] {_load_error}", flush=True)
        return False

    _proc, _reader_thread, _response_queue = proc, t, q
    return True


def _translate_sentences(sentences: list[str]) -> list[str]:
    """Batch-translate a list of English sentences to Marathi via the
    isolated worker subprocess. Returns the input unchanged if the worker
    isn't available."""
    global _load_failed, _load_error, _proc
    if not sentences:
        return sentences
    if not _ensure_worker():
        return sentences

    non_empty = [(i, s) for i, s in enumerate(sentences) if s.strip()]
    if not non_empty:
        return sentences
    idxs, texts = zip(*non_empty)

    with _proc_lock:
        try:
            _proc.stdin.write(json.dumps({"sentences": list(texts)}) + "\n")
            _proc.stdin.flush()
        except Exception as e:
            _load_error = f"Failed to send request to translation worker: {e}"
            _load_failed = True
            return sentences

        try:
            line = _response_queue.get(timeout=REQUEST_TIMEOUT_SECONDS)
        except queue.Empty:
            _load_error = (
                f"Translation worker did not respond within "
                f"{REQUEST_TIMEOUT_SECONDS}s — treating it as hung."
            )
            _load_failed = True
            try:
                _proc.kill()
            except Exception:
                pass
            return sentences

    if line is None or line == "":
        err = _proc.stderr.read() if _proc.stderr else ""
        _load_error = f"Translation worker exited unexpectedly: {err.strip()[-2000:]}"
        _load_failed = True
        return sentences

    try:
        resp = json.loads(line)
    except Exception:
        _load_error = f"Worker sent unparseable response: {line[:500]}"
        _load_failed = True
        return sentences

    if not resp.get("ok"):
        _load_error = resp.get("error", "unknown translation error")
        _load_failed = True
        return sentences

    translated = resp["translations"]
    out = list(sentences)
    for i, t in zip(idxs, translated):
        out[i] = t
    return out


# ---------------------------------------------------------------- markdown-aware translation

_TABLE_SEP_ROW = re.compile(r"^\s*\|?[\s:\-|]+\|?\s*$")
_HEADING = re.compile(r"^(\#{1,6}\s+)(.*)$")
_BULLET = re.compile(r"^(\s*(?:[-*]|\d+[.)])\s+)(.*)$")
# Splits on sentence-ending punctuation followed by whitespace and a new
# capital letter/digit/quote/Devanagari letter -- conservative, avoids
# splitting on abbreviation-style periods mid-sentence in most cases.
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'\u0900-\u097F])")


def _split_sentences(text: str) -> list[str]:
    """Break a paragraph into individual sentences. IndicTrans2 is a
    sentence-level NMT model — feeding it a whole multi-sentence paragraph
    as one unit risks silent truncation at the token-length cap, which is
    exactly what caused Marathi answers to come out much shorter than the
    English original: long paragraphs were being cut off before translation
    even happened, not just translated tersely."""
    text = text.strip()
    if not text:
        return [text]
    parts = [p.strip() for p in _SENTENCE_SPLIT.split(text) if p.strip()]
    return parts or [text]


def _register(text: str, to_translate: list[str]) -> list[int]:
    """Split text into sentences, append each to to_translate, and return
    the list of indices so the caller can rejoin them after translation."""
    idxs = []
    for sentence in _split_sentences(text):
        to_translate.append(sentence)
        idxs.append(len(to_translate) - 1)
    return idxs


def _split_translatable_lines(markdown_text: str):
    """Break Markdown into (kind, payload) units so structure survives
    translation: headings keep their '#'s, list items keep their bullet
    markers, table cells are translated individually rather than as one
    pipe-delimited blob, and pure separator rows / blank lines pass through
    untouched. Each unit's payload holds a LIST of to_translate indices (one
    per sentence in that unit) rather than a single index, so a multi-
    sentence paragraph is translated sentence-by-sentence rather than as one
    oversized chunk. Returns (units, to_translate)."""
    units = []
    to_translate = []

    for line in markdown_text.split("\n"):
        if not line.strip():
            units.append(("blank", None))
            continue
        if _TABLE_SEP_ROW.match(line) and "|" in line:
            units.append(("raw", line))
            continue
        if "|" in line.strip().strip("|"):
            cells = line.split("|")
            cell_slots = []
            for cell in cells:
                stripped = cell.strip()
                if stripped:
                    idxs = _register(stripped, to_translate)
                    cell_slots.append(("t", idxs,
                                       cell[:len(cell) - len(cell.lstrip())],
                                       cell[len(cell.rstrip()):]))
                else:
                    cell_slots.append(("raw", cell))
            units.append(("table_row", cell_slots))
            continue
        m = _HEADING.match(line)
        if m:
            idxs = _register(m.group(2), to_translate)
            units.append(("heading", (m.group(1), idxs)))
            continue
        m = _BULLET.match(line)
        if m:
            idxs = _register(m.group(2), to_translate)
            units.append(("bullet", (m.group(1), idxs)))
            continue
        idxs = _register(line, to_translate)
        units.append(("line", idxs))

    return units, to_translate


def _join(idxs: list[int], translated: list[str]) -> str:
    return " ".join(translated[i] for i in idxs)


def _reassemble(units, translated: list[str]) -> str:
    out_lines = []
    for kind, data in units:
        if kind == "blank":
            out_lines.append("")
        elif kind == "raw":
            out_lines.append(data)
        elif kind == "line":
            out_lines.append(_join(data, translated))
        elif kind == "heading":
            prefix, idxs = data
            out_lines.append(prefix + _join(idxs, translated))
        elif kind == "bullet":
            prefix, idxs = data
            out_lines.append(prefix + _join(idxs, translated))
        elif kind == "table_row":
            cells = []
            for slot in data:
                if slot[0] == "raw":
                    cells.append(slot[1])
                else:
                    _, idxs, lead, trail = slot
                    cells.append(lead + _join(idxs, translated) + trail)
            out_lines.append("|".join(cells))
    return "\n".join(out_lines)


def translate_markdown_to_marathi(markdown_text: str) -> str | None:
    """Translate an English Markdown answer to Marathi, preserving
    headings/lists/table structure. Returns None if translation is
    unavailable (caller should fall back to showing the English text)."""
    if not _ensure_worker():
        return None
    units, to_translate = _split_translatable_lines(markdown_text)
    translated = _translate_sentences(to_translate)
    if _load_failed:
        # The worker was ready at the start of this call but failed DURING
        # it (crashed, hung, sent garbage) -- _translate_sentences degrades
        # by returning the original text unchanged, which must NOT be
        # mistaken by the caller for a successful translation.
        return None
    return _reassemble(units, translated)
