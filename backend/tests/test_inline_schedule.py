"""Regression tests for inline schedule detection/extraction in chat
(rag.looks_like_schedule_request, rag.extract_schedule_request, and the
JSON parsing behind it). The deterministic date math itself is already
covered by test_scheduler.py -- these tests are about the (inherently
less reliable) step of pulling structured stations/date/details out of
a free-text chat message.

Run with:  pytest tests/test_inline_schedule.py -v
Or standalone:  python tests/test_inline_schedule.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import rag


# ---------------------------------------------------------------- detection heuristic

def test_heuristic_triggers_on_real_schedule_requests():
    positive = [
        "Create an inspection schedule for Andheri Police Station and Bandra Police Station",
        "I need a schedule for inspecting these stations",
        "Prepare an inspection programme for the following police stations",
    ]
    for m in positive:
        assert rag.looks_like_schedule_request(m), m


def test_heuristic_does_not_trigger_on_ordinary_questions():
    negative = [
        "What is the KYC limit for savings accounts?",
        "How should the Superintendent of Police inspect a station?",
        "What is the inspection process at the station?",
        "Tell me about circular G/3239",
    ]
    for m in negative:
        assert not rag.looks_like_schedule_request(m), m


# ---------------------------------------------------------------- JSON parsing

def test_clean_extraction_json_parsed_correctly():
    clean = ('{"stations": [{"name": "Andheri Police Station", "important": true}, '
            '{"name": "Bandra Police Station", "important": false}], '
            '"start_date": "2027-04-01", "details": "District: Mumbai"}')
    result = rag._parse_schedule_request_json(clean)
    assert result["stations"] == [
        {"name": "Andheri Police Station", "important": True},
        {"name": "Bandra Police Station", "important": False},
    ]
    assert result["start_date"] == "2027-04-01"
    assert result["details"] == "District: Mumbai"


def test_markdown_fenced_json_correctly_stripped():
    clean = '{"stations": [{"name": "X", "important": false}], "start_date": null, "details": ""}'
    fenced = "```json\n" + clean + "\n```"
    assert rag._parse_schedule_request_json(fenced) == rag._parse_schedule_request_json(clean)


def test_json_with_surrounding_prose_extracted():
    clean = '{"stations": [{"name": "X", "important": false}], "start_date": null, "details": ""}'
    messy = "Sure, here's what I found:\n\n" + clean + "\n\nLet me know if you need changes!"
    assert rag._parse_schedule_request_json(messy) == rag._parse_schedule_request_json(clean)


def test_malformed_output_degrades_to_empty_not_crash():
    garbage = "I don't understand the request."
    result = rag._parse_schedule_request_json(garbage)
    assert result == {"stations": [], "start_date": None, "details": ""}


def test_non_iso_date_rejected_not_trusted():
    """Regression: a relative/ambiguous date string like 'next month' must
    never be passed through as if it were a real date -- only a real
    YYYY-MM-DD is trusted, everything else becomes None so the caller
    asks the user to clarify instead of guessing."""
    bad = '{"stations": [{"name": "X", "important": false}], "start_date": "next month", "details": ""}'
    result = rag._parse_schedule_request_json(bad)
    assert result["start_date"] is None


def test_empty_stations_list_represented_correctly():
    none_case = '{"stations": [], "start_date": null, "details": ""}'
    assert rag._parse_schedule_request_json(none_case)["stations"] == []


# ---------------------------------------------------------------- end-to-end extraction

def test_extraction_end_to_end_with_mocked_llm():
    rag.ask_phi3 = lambda p: (
        '{"stations": [{"name": "Andheri Police Station", "important": true}, '
        '{"name": "Bandra Police Station", "important": false}], '
        '"start_date": "2027-04-01", "details": "District: Mumbai"}'
    )
    result = rag.extract_schedule_request(
        "Create an inspection schedule for Andheri Police Station (important) "
        "and Bandra Police Station starting April 1, 2027")
    assert len(result["stations"]) == 2
    assert result["start_date"] == "2027-04-01"


def test_extraction_degrades_gracefully_on_llm_failure():
    rag.ask_phi3 = lambda p: (_ for _ in ()).throw(Exception("model unreachable"))
    result = rag.extract_schedule_request("Create a schedule for X police station")
    assert result == {"stations": [], "start_date": None, "details": ""}


def test_false_trigger_yields_empty_stations_for_fallthrough():
    """Heuristic matches (mentions 'schedule' + 'station') but no specific
    station is actually named -- extraction must correctly find nothing,
    so the caller falls through to normal chat instead of forcing a
    broken schedule interaction."""
    rag.ask_phi3 = lambda p: '{"stations": [], "start_date": null, "details": ""}'
    result = rag.extract_schedule_request(
        "Can you explain the schedule policy for police stations in general?")
    assert result["stations"] == []


def test_extract_schedule_rules_is_a_real_independent_function():
    """Regression: extract_schedule_rules and extract_schedule_request are
    DIFFERENT functions (one parses circular excerpts for duration rules,
    the other parses the user's own chat message for stations/dates). An
    editing mistake once left extract_schedule_rules's body without its
    own 'def' line, silently swallowing it as dead code at the tail of
    extract_schedule_request -- py_compile didn't catch it since the
    result was still syntactically valid Python, just structurally wrong.
    This confirms both exist as independent, correctly-signatured
    module-level callables, not just that the module imports cleanly."""
    import inspect
    assert hasattr(rag, "extract_schedule_rules")
    assert hasattr(rag, "extract_schedule_request")
    assert rag.extract_schedule_rules is not rag.extract_schedule_request
    assert list(inspect.signature(rag.extract_schedule_rules).parameters) == ["hits"]
    assert list(inspect.signature(rag.extract_schedule_request).parameters) == ["message"]


def test_extract_schedule_rules_end_to_end_with_mocked_llm():
    rag.ask_phi3 = lambda p: (
        '{"regular_duration_days": 2, "important_duration_days": 4, '
        '"times_per_year": 1, "source_note": "Per Standing Order 145"}'
    )
    hits = [{"text": "Minimum 2 days for regular, 4 for important stations.",
            "meta": {"doc_id": 1, "filename": "x.pdf",
                    "issuer": "Maharashtra Police", "year": "1966"}}]
    result = rag.extract_schedule_rules(hits)
    assert result["regular_duration_days"] == 2
    assert result["important_duration_days"] == 4


def test_extract_schedule_rules_empty_hits():
    result = rag.extract_schedule_rules([])
    assert result["regular_duration_days"] is None


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
