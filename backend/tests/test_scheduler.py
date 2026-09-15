"""Regression tests for scheduler.py — deterministic inspection-date
computation. Pure stdlib date arithmetic, no LLM or DB dependency.

Run with:  pytest tests/test_scheduler.py -v
Or standalone:  python tests/test_scheduler.py
"""
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import scheduler as s


def test_correct_per_category_durations_no_overlap():
    stations = [
        {"name": "Andheri PS", "important": True},
        {"name": "Bandra PS", "important": False},
        {"name": "Colaba PS", "important": False},
    ]
    result = s.compute_schedule(stations, date(2027, 1, 4),
                                regular_duration_days=2, important_duration_days=4)
    assert result[0]["duration_days"] == 4
    assert result[1]["duration_days"] == 2
    assert result[2]["duration_days"] == 2
    for i in range(1, len(result)):
        assert result[i]["start_date"] > result[i - 1]["end_date"]


def test_inspection_never_starts_on_non_working_day():
    stations = [{"name": "Test PS", "important": False}]
    result = s.compute_schedule(stations, date(2027, 1, 3),  # a Sunday
                                regular_duration_days=2, important_duration_days=4)
    assert result[0]["start_date"] == date(2027, 1, 4)
    assert result[0]["start_date"].weekday() != 6  # not Sunday


def test_holidays_are_skipped_for_start_date():
    stations = [{"name": "Test PS", "important": False}]
    holiday = date(2027, 1, 4)
    result = s.compute_schedule(stations, date(2027, 1, 3),
                                regular_duration_days=2, important_duration_days=4,
                                holidays={holiday})
    assert result[0]["start_date"] != holiday
    assert result[0]["start_date"] == date(2027, 1, 5)


def test_gap_days_adds_travel_buffer():
    stations = [{"name": "A", "important": False}, {"name": "B", "important": False}]
    result = s.compute_schedule(stations, date(2027, 1, 4),
                                regular_duration_days=2, important_duration_days=4,
                                gap_days=3)
    gap = (result[1]["start_date"] - result[0]["end_date"]).days
    assert gap >= 4  # 3-day gap plus the mandatory 1-day advance


def test_empty_station_list_returns_empty_schedule():
    result = s.compute_schedule([], date(2027, 1, 4),
                                regular_duration_days=2, important_duration_days=4)
    assert result == []


def test_invalid_duration_rejected():
    stations = [{"name": "Test PS", "important": False}]
    try:
        s.compute_schedule(stations, date(2027, 1, 4),
                           regular_duration_days=0, important_duration_days=4)
        assert False, "should have raised ValueError"
    except ValueError:
        pass


def test_overflow_reports_every_station_never_silently_drops():
    """Critical safety property for an operational document: when stations
    don't fit in the window, every one must be accounted for -- either
    successfully scheduled, or explicitly listed as not fitting. Never
    silently dropped, never silently overrun the window."""
    many_stations = [{"name": f"Station {i}", "important": True} for i in range(100)]
    try:
        s.compute_schedule(many_stations, date(2027, 1, 4),
                           regular_duration_days=2, important_duration_days=4,
                           window_days=365)
        assert False, "should have raised ScheduleOverflowError"
    except s.ScheduleOverflowError as e:
        total = len(e.partial_schedule) + len(e.remaining_stations)
        assert total == 100, "every station must be accounted for"
        assert len(e.remaining_stations) > 0


def test_realistic_case_fits_cleanly():
    stations = [{"name": f"Station {i}", "important": (i % 5 == 0)} for i in range(20)]
    result = s.compute_schedule(stations, date(2027, 4, 1),
                                regular_duration_days=2, important_duration_days=4,
                                window_days=365)
    assert len(result) == 20


def test_unused_important_duration_none_does_not_crash():
    """Regression: a real production bug. When a request contains ONLY
    regular stations, important_duration_days is correctly None (never
    found/needed) -- this must NOT crash the validation, since the value
    is never actually looked up for any station in this request. Found
    via a real user report: a 4-station schedule with zero important
    stations was being blocked entirely by a missing important-station
    duration value that didn't even apply to the request."""
    stations = [{"name": "satara", "important": False},
               {"name": "koregaon", "important": False},
               {"name": "pusegaon", "important": False},
               {"name": "dahiwadi", "important": False}]
    result = s.compute_schedule(stations, date(2026, 9, 15), 2, None)
    assert len(result) == 4
    assert all(r["duration_days"] == 2 for r in result)


def test_unused_regular_duration_none_does_not_crash():
    """Mirror case: all-important stations with regular_duration_days=None."""
    stations = [{"name": "X", "important": True}, {"name": "Y", "important": True}]
    result = s.compute_schedule(stations, date(2026, 9, 15), None, 4)
    assert len(result) == 2
    assert all(r["duration_days"] == 4 for r in result)


def test_mixed_categories_both_values_needed_and_provided():
    stations = [{"name": "X", "important": False}, {"name": "Y", "important": True}]
    result = s.compute_schedule(stations, date(2026, 9, 15), 2, 4)
    assert result[0]["duration_days"] == 2
    assert result[1]["duration_days"] == 4


def test_mixed_categories_missing_needed_value_still_raises():
    """Critical: the fix must not become too permissive -- if a category
    that IS actually present in the request has no duration value, that
    must still raise, not silently proceed with a wrong/missing duration."""
    stations = [{"name": "X", "important": False}, {"name": "Y", "important": True}]
    try:
        s.compute_schedule(stations, date(2026, 9, 15), 2, None)
        assert False, "should have raised -- important_duration_days IS needed here"
    except ValueError:
        pass


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
