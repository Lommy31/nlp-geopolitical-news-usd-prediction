# USD/IDR calendar sanity check

Input: `data/processed/usd_idr_jisdor_clean.csv`. Range: 2021-09-01 to 2026-09-01. All 1,202 records were checked, rather than a sample.

**Confirmed: this processed file contains no Saturday or Sunday observations.** Weekend row count: 0.

Dates are read directly from `Date` using `date.fromisoformat()`. The weekday is then determined using `date.weekday()` (Monday=0, Saturday=5, Sunday=6). The data contains dates without times, so no time-zone conversion is performed.

| Weekday | Row count |
|---|---:|
| Monday | 238 |
| Tuesday | 247 |
| Wednesday | 243 |
| Thursday | 238 |
| Friday | 236 |
| Saturday | 0 |
| Sunday | 0 |
| **Total** | **1202** |

Additional consistency checks:

- Invalid dates: 0.
- Duplicate dates: 0.
- Dates strictly increasing in file order: True.
- Missing, nonnumeric, nonfinite, or nonpositive exchange-rate values: 0.

The range contains 1305 Monday–Friday calendar dates, of which 103 are absent from the file. **This does not establish missing data**: those dates may be holidays or days without publication. Holiday calendars and official publication archives have not been cross-checked. The dates are listed in [usd_idr_absent_weekdays.csv](usd_idr_absent_weekdays.csv).

This confirms the absence of weekend rows in **the current processed CSV**. The audit has not verified whether all expected publication dates are present, whether source dates were converted correctly, or whether exchange-rate values match official archives.

Evidence: [counts by weekday](usd_idr_weekday_counts.csv), [weekend rows](usd_idr_weekend_rows.csv) (header only when empty), and [metrics with the input hash](usd_idr_audit_metrics.json).

Rerun from the project root: `python3 deira-work/sanity-check/audit_usd_idr.py`.
