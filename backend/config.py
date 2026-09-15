"""Central configuration.

CPU / GPU switching
-------------------
The app runs CPU-only by default. To use a GPU, set environment variables
before starting the backend (or edit the defaults below):

  EMBED_DEVICE=cuda        # embeddings on GPU ("cpu" | "cuda" | "cuda:0" | "mps")
  OLLAMA_NUM_GPU=-1        # LLM layers on GPU: 0 = CPU only, -1 = offload all,
                           # or a number of layers for partial offload

Example:  EMBED_DEVICE=cuda OLLAMA_NUM_GPU=-1 uvicorn main:app
"""
import os
from pathlib import Path

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
FILES_DIR = BASE_DIR / "static" / "circulars"

DB_PATH = DATA_DIR / "repository.db"
CHROMA_DIR = str(DATA_DIR / "chroma")

# ---- Compute device -------------------------------------------------
EMBED_DEVICE = os.getenv("EMBED_DEVICE", "cpu")
OLLAMA_NUM_GPU = int(os.getenv("OLLAMA_NUM_GPU", "0"))   # 0 = CPU-only
LLM_THREADS = int(os.getenv("LLM_THREADS", "0"))          # 0 = let Ollama decide

# ---- Models ----------------------------------------------------------
# Default embedder is English-only. For Marathi/Hindi + English circulars use:
#   EMBED_MODEL=sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
# (covers 50+ languages incl. Marathi; re-upload documents after switching,
# because embeddings from different models are not comparable)
# Multilingual by default (Marathi + English). English-only alternative
# (slightly better for pure-English repos, smaller):
#   EMBED_MODEL=sentence-transformers/all-MiniLM-L6-v2
EMBED_MODEL = os.getenv(
    "EMBED_MODEL",
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434/api/generate")
# qwen2.5:7b: better accuracy + Marathi than phi3, ~6 GB RAM resident.
# Lighter fallback for low-RAM machines: LLM_MODEL=phi3
LLM_MODEL = os.getenv("LLM_MODEL", "qwen2.5:14b")
LLM_TIMEOUT = int(os.getenv("LLM_TIMEOUT", "600"))  # read timeout (s)

# ---- Latency controls (timeout avoidance) -----------------------------
KEEP_ALIVE = os.getenv("KEEP_ALIVE", "30m")   # keep model in RAM between asks
NUM_PREDICT = int(os.getenv("NUM_PREDICT", "1800"))  # max answer tokens
# Deterministic by default: the same question should give the same answer
# every time, both for user trust and for auditability of an official
# circular-reference tool. temperature=0 makes generation greedy (always
# picks the most likely next token) instead of sampling with randomness;
# a fixed seed is a second layer of determinism for any residual
# nondeterminism in the runtime. Raise LLM_TEMPERATURE above 0 only if you
# specifically want varied phrasing across repeated asks.
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0"))
LLM_SEED = int(os.getenv("LLM_SEED", "42"))
NUM_CTX = int(os.getenv("NUM_CTX", "8192"))          # context window (Devanagari needs headroom)
PROMPT_CHUNK_CHARS = int(os.getenv("PROMPT_CHUNK_CHARS", "1500"))  # per-excerpt cap
WARMUP = os.getenv("WARMUP", "1") != "0"      # preload models at startup

# ---- Auth ------------------------------------------------------------
# Token signing key. Set a strong value in production:
#   SECRET_KEY=$(openssl rand -hex 32)
SECRET_KEY = os.getenv("SECRET_KEY", "dev-secret-change-me")

# ---- Retrieval quality -----------------------------------------------
# Hybrid search: BM25 keyword scores fused with vector similarity (RRF),
# then a cross-encoder reranks the candidates for the final TOP_K.
HYBRID_ENABLED = os.getenv("HYBRID_ENABLED", "1") != "0"
RERANK_ENABLED = os.getenv("RERANK_ENABLED", "1") != "0"
# English default; multilingual (Marathi) alternative:
#   RERANK_MODEL=cross-encoder/mmarco-mMiniLMv2-L12-H384-v1
# Multilingual reranker by default; English-only alternative:
#   RERANK_MODEL=cross-encoder/ms-marco-MiniLM-L-6-v2
RERANK_MODEL = os.getenv("RERANK_MODEL",
                         "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1")
TOP_K_CANDIDATES = int(os.getenv("TOP_K_CANDIDATES", "20"))  # before rerank

# Relevance floor: without this, retrieval always returns the "best
# available" chunks even when NONE are actually relevant, and every source
# among them gets cited — producing answers that reference documents
# unrelated to the question. Cross-encoder scores are unbounded logits
# (roughly -11 to +11 for MS-MARCO-style models, not a 0-1 scale), so two
# checks are used together:
#   RERANK_MIN_SCORE — an absolute floor; candidates below this are dropped
#   RERANK_MAX_GAP   — a RELATIVE floor: candidates scoring much worse than
#                      the best match for THIS question are dropped too,
#                      which adapts per-query and is less sensitive to your
#                      corpus not matching the reranker's training domain
#                      than an absolute cutoff alone.
# If your repository is a very different domain (e.g. dense legal Marathi
# text) and relevant answers start getting rejected, raise RERANK_MAX_GAP
# or lower RERANK_MIN_SCORE. If irrelevant citations still slip through,
# tighten them.
RERANK_MIN_SCORE = float(os.getenv("RERANK_MIN_SCORE", "-4.0"))
RERANK_MAX_GAP = float(os.getenv("RERANK_MAX_GAP", "5.0"))
# Fallback floor when reranking is disabled/unavailable: vector cosine
# similarity (0-1 scale) below this is treated as "not actually relevant".
VECTOR_MIN_SCORE = float(os.getenv("VECTOR_MIN_SCORE", "0.2"))

# ---- RAG -------------------------------------------------------------
CHUNK_SIZE = 900
CHUNK_OVERLAP = 150
TOP_K = int(os.getenv("TOP_K", "8"))
HISTORY_TURNS = 6           # prior messages included for follow-up questions

# ---- Contradiction detection -------------------------------------------
# Runs a SEPARATE, focused LLM pass over the same retrieved excerpts,
# specifically asking it to identify where different circulars state
# conflicting requirements on the same matter (e.g. one saying "minimum 2
# days" and another saying "minimum 4 days" for what appears to be the same
# inspection requirement — a real example found in this repository's own
# documents). A dedicated pass with a narrow JSON-only task is more reliable
# than asking the main chat model to also notice and mention contradictions
# in free-form prose, given demonstrated unreliability with compound
# instructions. Adds one extra LLM call per question -- real latency cost
# on CPU -- so it's toggleable.
CONTRADICTION_DETECTION_ENABLED = os.getenv("CONTRADICTION_DETECTION_ENABLED", "1") != "0"

# ---- Inline schedule detection in chat ---------------------------------
# When enabled, a chat message that looks like a schedule request (matched
# by a cheap keyword heuristic first, to avoid slowing down ordinary chat)
# triggers a narrow LLM extraction of station names/dates from the
# message itself, then the SAME deterministic scheduler used by the
# Templates page. This is less reliable at the input-parsing step than
# the Templates page's structured form (an LLM is reading free text
# instead of a clean list), which is why the response always shows
# exactly what was extracted before any schedule table — so a misparse
# is immediately visible rather than silently baked into an operational
# document. Toggle off to require the dedicated form instead.
INLINE_SCHEDULE_DETECTION_ENABLED = os.getenv("INLINE_SCHEDULE_DETECTION_ENABLED", "1") != "0"

# ---- Translation (opt-in) --------------------------------------------
# When enabled, a Devanagari-script question makes the chat LLM answer in
# English (where it's demonstrably more reliable) and translates that
# answer to Marathi with a dedicated NMT model (IndicTrans2) instead of
# asking the chat LLM to generate Marathi directly. Requires:
#   pip install IndicTransToolkit
# (transformers/torch are already present as sentence-transformers deps.)
TRANSLATION_ENABLED = os.getenv("TRANSLATION_ENABLED", "0") != "0"
TRANSLATION_MODEL = os.getenv("TRANSLATION_MODEL",
                              "ai4bharat/indictrans2-en-indic-dist-200M")
TRANSLATION_DEVICE = os.getenv("TRANSLATION_DEVICE", "cpu")


def _default_translation_venv_python() -> str:
    """Python venvs use a different internal layout per OS: Windows creates
    Scripts\\python.exe, Linux/macOS create bin/python3. Hardcoding one
    convention breaks the other silently (confirmed: a Windows setup
    reported "venv not found" at a bin/python3 path that never existed on
    that OS — the venv itself had been created fine, just at Scripts/
    instead). Check both actual candidate paths and use whichever exists;
    fall back to the OS-appropriate guess if the venv hasn't been set up
    yet at all."""
    venv_dir = BASE_DIR / "translation_venv"
    posix_candidate = venv_dir / "bin" / "python3"
    windows_candidate = venv_dir / "Scripts" / "python.exe"
    if os.name == "nt":
        candidates = [windows_candidate, posix_candidate]
    else:
        candidates = [posix_candidate, windows_candidate]
    for c in candidates:
        if c.exists():
            return str(c)
    return str(candidates[0])  # venv not set up yet -- best guess for this OS


# Path to the ISOLATED translation venv's Python interpreter (set up once
# via setup_translation_venv.sh). This is deliberately a separate venv from
# the main app: IndicTrans2's custom model code needs an older transformers
# version than the embedder/reranker do, and pinning that in the shared
# environment risks breaking them.
TRANSLATION_VENV_PYTHON = os.getenv(
    "TRANSLATION_VENV_PYTHON", _default_translation_venv_python())
# Question is treated as Devanagari-script if at least this fraction of its
# letters are in the Devanagari range.
TRANSLATION_TRIGGER_RATIO = float(os.getenv("TRANSLATION_TRIGGER_RATIO", "0.3"))

MIN_TEXT_CHARS = 120
ALLOWED_EXTENSIONS = {".pdf", ".docx", ".txt",
                      ".png", ".jpg", ".jpeg", ".tiff", ".tif"}

# ---- OCR (for scanned PDFs and image files) --------------------------
# Requires the Tesseract binary on the system PATH:
#   Ubuntu/Debian:  sudo apt install tesseract-ocr
#   macOS:          brew install tesseract
#   Windows:        https://github.com/UB-Mannheim/tesseract/wiki
OCR_ENABLED = os.getenv("OCR_ENABLED", "1") != "0"
OCR_LANG = os.getenv("OCR_LANG", "eng")     # Marathi+English: "mar+eng"
                                            # (install: tesseract-ocr-mar)
OCR_DPI = int(os.getenv("OCR_DPI", "220"))  # rasterization resolution
OCR_DENOISE_H = int(os.getenv("OCR_DENOISE_H", "5"))
OCR_DENOISE_ENABLED = os.getenv("OCR_DENOISE_ENABLED", "1") != "0"
OCR_CLAHE_CLIP = float(os.getenv("OCR_CLAHE_CLIP", "1.0"))
OCR_MAX_PAGES = int(os.getenv("OCR_MAX_PAGES", "60"))
# If OCR_LANG includes 'mar'/'hin' but the resulting text has fewer than
# this fraction of Devanagari letters, OCR almost certainly ran without the
# language pack actually installed (silent fallback to English-shaped
# glyphs) — the document is flagged 'ocr_suspect' instead of indexed as if
# it were read correctly.
OCR_MIN_SCRIPT_RATIO = float(os.getenv("OCR_MIN_SCRIPT_RATIO", "0.15"))
# Page Segmentation Mode: 3 = fully automatic (good default for most
# circulars); 6 = assume one uniform block of text (try if pages have a
# very consistent single-column layout); 11 = sparse text, any order
# (try for complex/non-standard layouts, e.g. tables mixed with prose).
OCR_PSM = int(os.getenv("OCR_PSM", "3"))
# OCR Engine Mode: 1 = LSTM neural net only (best accuracy on modern
# language packs, recommended); 3 = default/legacy fallback.
OCR_OEM = int(os.getenv("OCR_OEM", "1"))
# Deskew + adaptive contrast enhancement (CLAHE) before OCR — measurably
# improves accuracy on real-world scans: photocopies at a slight angle,
# faded ink, uneven lighting near a binding. Requires opencv-python-headless
# (pip install opencv-python-headless); gracefully skipped if not installed.
OCR_PREPROCESS = os.getenv("OCR_PREPROCESS", "1") != "0"

# Base URL the frontend uses for clickable file links
PUBLIC_FILES_URL = os.getenv("PUBLIC_FILES_URL", "/files")

for d in (DATA_DIR, FILES_DIR):
    d.mkdir(parents=True, exist_ok=True)
