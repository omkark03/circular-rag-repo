"""Deterministic inspection-schedule computation.

Deliberately NOT LLM-generated: computing exact calendar dates reliably
(skipping non-working days and holidays, not overrunning the available
year, never double-booking) is exactly the kind of task this project has
repeatedly found this class of model unreliable at. The LLM's role is
limited to extracting scheduling RULES from circulars as structured data
(frequency, duration per station category) — see rag.py's
extract_schedule_rules(). Actually turning those rules into real dates
happens here, in plain, fully-testable Python arithmetic.
"""
from datetime import date, timedelta


class ScheduleOverflowError(Exception):
    """Raised when the requested stations don't fit within the allowed
    scheduling window — this must be surfaced clearly to the user, never
    silently truncated or overrun."""
    def __init__(self, message, partial_schedule, remaining_stations):
        super().__init__(message)
        self.partial_schedule = partial_schedule
        self.remaining_stations = remaining_stations


def _is_working_day(d: date, working_weekdays: set[int], holidays: set[date]) -> bool:
    return d.weekday() in working_weekdays and d not in holidays


def _next_working_day(d: date, working_weekdays: set[int], holidays: set[date]) -> date:
    while not _is_working_day(d, working_weekdays, holidays):
        d += timedelta(days=1)
    return d


def compute_schedule(
    stations: list[dict],
    start_date: date,
    regular_duration_days: int | None,
    important_duration_days: int | None,
    *,
    working_weekdays: set[int] = frozenset({0, 1, 2, 3, 4, 5}),  # Mon-Sat, Sun off
    holidays: set[date] = frozenset(),
    window_days: int = 365,
    gap_days: int = 0,
) -> list[dict]:
    """Sequentially assigns each station a real, contiguous calendar-date
    range for its inspection, starting from the first working day on or
    after start_date.

    Each inspection is CONSECUTIVE CALENDAR days once started (matching
    how a multi-day inspection actually works in practice — you don't
    pause an inspection mid-way for a weekend), but a station's inspection
    never STARTS on a non-working day or a holiday.

    stations: [{"name": str, "important": bool}, ...] — order given is
    the order scheduled; caller controls priority by list order.
    regular_duration_days / important_duration_days: from
    rag.extract_schedule_rules() (or user override).
    window_days: the scheduling period (default 365 = one year) measured
    from start_date. If the stations don't fit, raises
    ScheduleOverflowError with the partial schedule and the stations that
    didn't fit, rather than silently overrunning the window or dropping
    stations without saying so.

    Returns a list of dicts: [{"station": str, "important": bool,
    "start_date": date, "end_date": date, "duration_days": int}, ...]
    """
    has_regular = any(not s.get("important", False) for s in stations)
    has_important = any(s.get("important", False) for s in stations)
    if has_regular and (regular_duration_days is None or regular_duration_days < 1):
        raise ValueError("regular_duration_days must be at least 1 for this request")
    if has_important and (important_duration_days is None or important_duration_days < 1):
        raise ValueError("important_duration_days must be at least 1 for this request")

    deadline = start_date + timedelta(days=window_days)
    schedule = []
    cursor = start_date

    for i, station in enumerate(stations):
        name = station["name"]
        important = bool(station.get("important", False))
        duration = important_duration_days if important else regular_duration_days

        inspection_start = _next_working_day(cursor, working_weekdays, holidays)
        inspection_end = inspection_start + timedelta(days=duration - 1)

        if inspection_end >= deadline:
            raise ScheduleOverflowError(
                f"{len(stations) - i} station(s) do not fit within the "
                f"{window_days}-day scheduling window starting {start_date}. "
                f"Extend the window, add more working days, or reduce "
                f"per-station duration.",
                partial_schedule=schedule,
                remaining_stations=[s["name"] for s in stations[i:]],
            )

        schedule.append({
            "station": name,
            "important": important,
            "start_date": inspection_start,
            "end_date": inspection_end,
            "duration_days": duration,
        })
        cursor = inspection_end + timedelta(days=1 + gap_days)

    return schedule
