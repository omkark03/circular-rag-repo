#!/usr/bin/env bash
# Sets up an ISOLATED Python venv for translation, with its own pinned
# dependency versions, fully decoupled from the main app's environment.
#
# WHY a separate venv: IndicTrans2's custom model code needs an older
# transformers version than the embedder/reranker do. Getting that pin
# exactly right took several rounds of real debugging (tokenizer import
# paths, a removed transformers.onnx module, a Pillow.Resampling
# incompatibility, and a past_key_values Cache-object format change).
# Isolating it means none of that can ever destabilize the reranker/
# embedder, which already work correctly on your main environment's
# transformers version.
#
# Usage:  bash setup_translation_venv.sh
# Run from the backend/ directory (same place you run uvicorn from).

set -e

VENV_DIR="translation_venv"

echo "Creating isolated venv at ./${VENV_DIR} ..."
python3 -m venv "${VENV_DIR}"

# Windows venvs use Scripts\python.exe, Linux/macOS use bin/python3 -- this
# script may run under Git Bash on Windows, where the venv module still
# creates the native Windows layout. Detect which one actually exists
# rather than assuming, so this works either way.
if [ -f "${VENV_DIR}/Scripts/python.exe" ]; then
    VENV_PY="${VENV_DIR}/Scripts/python.exe"
    VENV_PIP="${VENV_DIR}/Scripts/pip.exe"
else
    VENV_PY="${VENV_DIR}/bin/python3"
    VENV_PIP="${VENV_DIR}/bin/pip"
fi

echo "Installing pinned dependencies (this venv is independent of the main app's packages) ..."
"${VENV_PIP}" install --upgrade pip

# NOTE: this starting pin is based on the errors seen so far (needs
# transformers.onnx, which transformers removed in 5.1.0, and an
# encoder-decoder past_key_values format that later versions changed).
# If the model still fails to load with this version, that's the next
# thing to adjust -- ONLY in this isolated venv, with zero risk to the
# main app's environment. Check by running:
#   <VENV_PY> translation_worker.py ai4bharat/indictrans2-en-indic-dist-200M cpu
# and watching for the {"ready": true} line.
"${VENV_PIP}" install \
    "transformers==4.41.2" \
    "torch>=2.0,<2.5" \
    "Pillow>=10.0" \
    "IndicTransToolkit" \
    "sentencepiece"

echo ""
echo "Patching a known IndicTransToolkit 1.1.1 bug (wrong import path for"
echo "PreTrainedTokenizerBase) -- safe to re-run if this venv is rebuilt:"
COLLATOR=$("${VENV_PY}" -c "import IndicTransToolkit, os; print(os.path.join(os.path.dirname(IndicTransToolkit.__file__), 'collator.py'))" 2>/dev/null || true)
if [ -n "$COLLATOR" ] && [ -f "$COLLATOR" ]; then
    sed -i 's/from transformers.tokenization_utils import PreTrainedTokenizerBase/from transformers.tokenization_utils_base import PreTrainedTokenizerBase/' "$COLLATOR"
    echo "  patched: $COLLATOR"
else
    echo "  could not locate collator.py automatically -- if you see an"
    echo "  ImportError mentioning PreTrainedTokenizerBase later, patch it"
    echo "  manually (see README's translation troubleshooting section)."
fi

echo ""
echo "Done. This venv's Python interpreter is at:"
echo "  ${VENV_PY}"
echo "The app auto-detects this on startup, so TRANSLATION_VENV_PYTHON"
echo "usually doesn't need to be set manually -- but if you ever need to:"
echo "  export TRANSLATION_VENV_PYTHON=\$(pwd)/${VENV_PY#./}"
echo "  export TRANSLATION_ENABLED=1"
echo ""
echo "Test it directly before starting the full app:"
echo "  ${VENV_PY} translation_worker.py ai4bharat/indictrans2-en-indic-dist-200M cpu"
echo "  (watch for a line reading exactly: {\"ready\": true})"
