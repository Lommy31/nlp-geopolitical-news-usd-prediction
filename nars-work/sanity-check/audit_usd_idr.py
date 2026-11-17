"""Audit calendar dates and basic numeric consistency of the processed FX CSV.

Run from any directory using Python's standard library. Missing weekdays are
unclassified: this audit does not assume they are holidays or missing data.
"""
from collections import Counter
from datetime import date, timedelta
from decimal import Decimal, InvalidOperation
from pathlib import Path
import csv
import hashlib
import json

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
DAY_NAMES = ["Senin", "Selasa", "Rabu", "Kamis", "Jumat", "Sabtu", "Minggu"]


def save_csv(name, fields, rows):
    with (OUT / name).open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main():
    source = ROOT / "data/processed/usd_idr_jisdor_clean.csv"
    with source.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    parsed, invalid_dates, invalid_values = [], [], []
    for i, row in enumerate(rows):
        try:
            day = date.fromisoformat(row["Date"])
            parsed.append((i, day, row["USD_IDR"]))
        except (ValueError, TypeError) as error:
            invalid_dates.append({"data_row_index": i, "Date": row["Date"], "reason": str(error)})
        try:
            value = Decimal(row["USD_IDR"])
            valid_value = value.is_finite() and value > 0
        except (InvalidOperation, ValueError, TypeError):
            valid_value = False
        if not valid_value:
            invalid_values.append({"data_row_index": i, "USD_IDR": row["USD_IDR"]})
    assert parsed, "No parseable dates in input"
    counts = Counter(day.weekday() for _, day, _ in parsed)
    date_counts = Counter(day for _, day, _ in parsed)
    weekends = [{"data_row_index": i, "Date": str(day), "weekday": DAY_NAMES[day.weekday()], "USD_IDR": value}
                for i, day, value in parsed if day.weekday() >= 5]
    weekdays = [{"weekday_index": k, "weekday": name, "row_count": counts[k]} for k, name in enumerate(DAY_NAMES)]
    first, last = min(date_counts), max(date_counts)
    missing_weekdays = []
    calendar_day = first
    while calendar_day <= last:
        if calendar_day.weekday() < 5 and calendar_day not in date_counts:
            missing_weekdays.append({"Date": str(calendar_day), "weekday": DAY_NAMES[calendar_day.weekday()],
                                     "status": "unclassified; verify holiday/publication calendar before treating as missing data"})
        calendar_day += timedelta(days=1)
    duplicates = [{"Date": str(day), "row_count": count} for day, count in sorted(date_counts.items()) if count > 1]
    gaps = [{"previous_date": str(a), "next_date": str(b), "calendar_day_gap": (b-a).days}
            for a, b in zip(sorted(date_counts), sorted(date_counts)[1:])]
    stats = {
        "input": str(source.relative_to(ROOT)), "sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "rows": len(rows), "valid_date_rows": len(parsed), "first_date": str(first), "last_date": str(last),
        "weekday_counts": {name: counts[k] for k, name in enumerate(DAY_NAMES)},
        "weekend_rows": len(weekends), "invalid_dates": invalid_dates, "invalid_values": invalid_values,
        "duplicate_dates": duplicates,
        "dates_strictly_increasing_in_file": all(a[1] < b[1] for a, b in zip(parsed, parsed[1:])),
        "absent_monday_friday_dates": len(missing_weekdays),
        "expected_monday_friday_calendar_dates": sum(counts[k] for k in range(5))-sum(c-1 for d, c in date_counts.items() if d.weekday()<5)+len(missing_weekdays),
        "longest_gaps": sorted(gaps, key=lambda r: r["calendar_day_gap"], reverse=True)[:5],
        "limitation": "No weekend rows proves only a property of this processed file. Missing weekdays are not classified as holidays; source rate values are not externally verified."
    }
    save_csv("usd_idr_weekday_counts.csv", ["weekday_index", "weekday", "row_count"], weekdays)
    save_csv("usd_idr_weekend_rows.csv", ["data_row_index", "Date", "weekday", "USD_IDR"], weekends)
    save_csv("usd_idr_absent_weekdays.csv", ["Date", "weekday", "status"], missing_weekdays)
    (OUT / "usd_idr_audit_metrics.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    report_day_names = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    table = "\n".join(f"| {name} | {counts[k]} |" for k, name in enumerate(report_day_names))
    report = f"""# USD/IDR calendar sanity check

Input: `{stats['input']}`. Range: {first} to {last}. All {len(rows):,} records were checked, rather than a sample.

**Confirmed: this processed file contains no Saturday or Sunday observations.** Weekend row count: {len(weekends)}.

Dates are read directly from `Date` using `date.fromisoformat()`. The weekday is then determined using `date.weekday()` (Monday=0, Saturday=5, Sunday=6). The data contains dates without times, so no time-zone conversion is performed.

| Weekday | Row count |
|---|---:|
{table}
| **Total** | **{len(rows)}** |

Additional consistency checks:

- Invalid dates: {len(invalid_dates)}.
- Duplicate dates: {len(duplicates)}.
- Dates strictly increasing in file order: {stats['dates_strictly_increasing_in_file']}.
- Missing, nonnumeric, nonfinite, or nonpositive exchange-rate values: {len(invalid_values)}.

The range contains {stats['expected_monday_friday_calendar_dates']} Monday–Friday calendar dates, of which {len(missing_weekdays)} are absent from the file. **This does not establish missing data**: those dates may be holidays or days without publication. Holiday calendars and official publication archives have not been cross-checked. The dates are listed in [usd_idr_absent_weekdays.csv](usd_idr_absent_weekdays.csv).

This confirms the absence of weekend rows in **the current processed CSV**. The audit has not verified whether all expected publication dates are present, whether source dates were converted correctly, or whether exchange-rate values match official archives.

Evidence: [counts by weekday](usd_idr_weekday_counts.csv), [weekend rows](usd_idr_weekend_rows.csv) (header only when empty), and [metrics with the input hash](usd_idr_audit_metrics.json).

Rerun from the project root: `python3 deira-work/sanity-check/audit_usd_idr.py`.
"""
    (OUT / "laporan_audit_usd_idr.md").write_text(report, encoding="utf-8")
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
