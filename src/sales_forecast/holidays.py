"""German public holiday calendar used to extend StateHoliday into the forecast window.

Rossmann operates in Germany; StateHoliday takes values a (public holiday),
b (Easter related) and c (Christmas). The calendar below reproduces those flags
for dates beyond the training horizon so that holiday features exist for
future forecasts.
"""

from __future__ import annotations

import datetime as dt

FIXED_HOLIDAYS = {
    (1, 1): ("a", "New Year"),
    (5, 1): ("a", "Labour Day"),
    (10, 3): ("a", "German Unity Day"),
    (12, 25): ("c", "Christmas Day"),
    (12, 26): ("c", "Second Christmas Day"),
}

FULL_YEARS = range(2012, 2018)


def easter(year: int) -> dt.date:
    """Gauss/Tantzen algorithm for Easter Sunday."""
    a = year % 19
    b = year // 100
    c = year % 100
    d = (19 * a + b - b // 4 - (b - (b + 8) // 25 + 1) // 3 + 15) % 30
    e = (32 + 2 * (b % 4) + 2 * (c // 4) - d - c % 4) % 7
    f = d + e - 7 * ((a + 11 * d + 22 * e) // 451) + 114
    return dt.date(year, f // 31, f % 31 + 1)


def build_holiday_map() -> dict[dt.date, tuple[str, str]]:
    """Return {date: (state_holiday_flag, name)} for supported years."""
    out: dict[dt.date, tuple[str, str]] = {}
    for year in FULL_YEARS:
        e = easter(year)
        movable = {
            e - dt.timedelta(days=2): ("b", "Good Friday"),
            e + dt.timedelta(days=1): ("b", "Easter Monday"),
            e + dt.timedelta(days=39): ("b", "Ascension Day"),
            e + dt.timedelta(days=50): ("b", "Whit Monday"),
        }
        out.update(movable)
        for (m, d), flag_name in FIXED_HOLIDAYS.items():
            out[dt.date(year, m, d)] = flag_name
    return out


HOLIDAY_MAP = build_holiday_map()


def state_holiday_for(date: dt.date) -> str:
    """State holiday flag (0/a/b/c) for a given date."""
    return HOLIDAY_MAP.get(date, ("0", "None"))[0]
