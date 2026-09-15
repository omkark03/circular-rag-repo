"""Regression tests for translation.py — the isolated-venv subprocess
architecture (real IPC tested with a fake worker, no torch/transformers
needed) and Markdown-aware English->Marathi translation structure
preservation.

Run with:  pytest tests/test_translation.py -v
Or standalone:  python tests/test_translation.py
"""
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import config
import translation as t

FAKE_WORKER = str(Path(__file__).parent / "fixtures" / "fake_worker.py")


def _reset(mode="normal", startup_timeout=10, request_timeout=10):
    """Point translation.py at the fake worker (real subprocess, no torch
    needed) instead of the real translation_worker.py, and clear cached
    state so each test starts fresh."""
    os.environ["FAKE_WORKER_MODE"] = mode
    t._WORKER_SCRIPT = FAKE_WORKER
    config.TRANSLATION_VENV_PYTHON = sys.executable
    t.STARTUP_TIMEOUT_SECONDS = startup_timeout
    t.REQUEST_TIMEOUT_SECONDS = request_timeout
    t.reset_load_state()


# ---------------------------------------------------------------- language detection

def test_devanagari_question_detected():
    assert t.is_devanagari_question("पोलीस स्टेशन निरीक्षण कसे होते?") is True
    assert t.is_devanagari_question("How is inspection done?") is False
    assert t.is_devanagari_question("") is False
    assert t.is_devanagari_question("१२३") is False


def test_mixed_script_majority_devanagari_triggers():
    mixed = "S.P. यांनी पोलीस स्टेशन चे निरीक्षण कसे करावे?"
    assert t.is_devanagari_question(mixed) is True


# ---------------------------------------------------------------- real subprocess IPC

def test_successful_round_trip_via_real_subprocess():
    """A real subprocess is spawned, the ready handshake completes, a
    request is sent and a response received — nothing here is mocked."""
    _reset("normal")
    result = t.translate_markdown_to_marathi("The police station must be inspected.")
    assert result == "[MR:The police station must be inspected.]"


def test_worker_crash_on_startup_detected():
    _reset("crash_on_startup")
    result = t.translate_markdown_to_marathi("Test")
    assert result is None
    assert "simulated startup crash" in t.load_error()


def test_worker_hang_on_startup_times_out():
    """Must not hang forever — a stuck worker has to time out and report
    failure rather than block the whole request indefinitely."""
    _reset("hang_on_startup", startup_timeout=3)
    start = time.time()
    result = t.translate_markdown_to_marathi("Test")
    elapsed = time.time() - start
    assert result is None
    assert elapsed < 8, f"should time out around 3s, took {elapsed}s"


def test_worker_crash_mid_request_returns_none_not_fake_success():
    """Regression: a worker that crashes DURING a request must not have its
    fallback (unchanged original text) mistaken by the caller for a
    successful translation."""
    _reset("crash_on_request")
    result = t.translate_markdown_to_marathi("Test sentence.")
    assert result is None, f"must be None, not a disguised English passthrough: {result!r}"


def test_worker_malformed_response_returns_none_not_fake_success():
    """Same regression class as above, different trigger: garbage on
    stdout instead of a crash."""
    _reset("bad_json")
    result = t.translate_markdown_to_marathi("Test sentence.")
    assert result is None, f"must be None, not a disguised English passthrough: {result!r}"


def test_missing_venv_reports_clear_error():
    _reset("normal")
    config.TRANSLATION_VENV_PYTHON = "/nonexistent/path/to/python3"
    t.reset_load_state()
    result = t.translate_markdown_to_marathi("Test")
    assert result is None
    assert "not found" in t.load_error()


# ---------------------------------------------------------------- markdown structure preservation

def test_markdown_structure_preserved_through_translation():
    """Headings, bullets, and table cells must survive translation intact —
    only the natural-language text inside them should change."""
    _reset("normal")

    md = """# Inspection Procedure

The Superintendent of Police must inspect each station annually.

- Minimum two days per visit
- More time for important stations

| Category | Requirement |
|---|---|
| Regular station | 2 days |
| Important station | Additional days |

Officers should file a report within 15 days."""

    result = t.translate_markdown_to_marathi(md)
    lines = result.split("\n")

    assert lines[0] == "# [MR:Inspection Procedure]"
    assert lines[2] == "[MR:The Superintendent of Police must inspect each station annually.]"
    assert lines[4].startswith("- [MR:Minimum two days per visit]")
    assert lines[5].startswith("- [MR:More time for important stations]")
    assert "|---|---|" in result  # separator row untouched
    assert "[MR:Category]" in result and "[MR:Requirement]" in result
    assert "[MR:Regular station]" in result and "[MR:2 days]" in result


def test_multisentence_paragraph_split_before_translation():
    """Regression: a long multi-sentence paragraph (exactly what the chat
    LLM produces under the "don't artificially shorten" rule) must be split
    into individual sentences before translation, not sent as one oversized
    unit that risks silent truncation at the NMT model's token cap. This
    was the actual cause of "Marathi answer much shorter than English"."""
    call_log = []

    def fake_translate(sentences):
        call_log.append(list(sentences))
        return [f"[MR:{s}]" for s in sentences]

    t._translate_sentences = fake_translate
    t._ensure_worker = lambda: True

    paragraph = (
        "The Superintendent of Police must inspect each police station at "
        "least once a year. For unimportant stations, a minimum of two days "
        "should be spent on inspection. For important stations, additional "
        "time should be allocated based on the volume of work."
    )
    result = t.translate_markdown_to_marathi(paragraph)

    sent_to_translator = call_log[0]
    assert len(sent_to_translator) == 3, \
        f"expected 3 separate sentences, got {len(sent_to_translator)}: {sent_to_translator}"
    assert result.count("[MR:") == 3
    # sentences must be rejoined into one coherent paragraph, not left
    # as disconnected fragments
    assert result.startswith("[MR:The Superintendent")
    assert "[MR:For important stations" in result


def test_multisentence_bullet_and_table_cell_also_split():
    """The same sentence-splitting must apply inside bullets and table
    cells, not just plain paragraphs -- any of them could contain more
    than one sentence."""
    call_log = []

    def fake_translate(sentences):
        call_log.append(list(sentences))
        return [f"[MR:{s}]" for s in sentences]

    t._translate_sentences = fake_translate
    t._ensure_worker = lambda: True

    md = ("- Two days minimum for regular stations. Additional time may be "
         "granted on request.\n\n"
         "| Category | Note |\n|---|---|\n"
         "| Regular | Applies to all districts. Exceptions must be approved. |")
    result = t.translate_markdown_to_marathi(md)

    all_sent = call_log[0]
    assert len(all_sent) == 7, f"expected 7 sentences total, got {len(all_sent)}: {all_sent}"
    assert result.count("[MR:") == 7


if __name__ == "__main__":
    tests = [(n, f) for n, f in list(globals().items())
             if n.startswith("test_") and callable(f)]
    failed = 0
    for name, fn in tests:
        try:
            fn()
            print(f"PASS  {name}")
        except AssertionError as e:
            failed += 1
            print(f"FAIL  {name}: {e}")
        except Exception as e:
            failed += 1
            print(f"ERROR {name}: {type(e).__name__}: {e}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    sys.exit(1 if failed else 0)
