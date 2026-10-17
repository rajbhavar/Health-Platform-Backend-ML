# FAKE sample data dictionary

No real patient information. All identifiers are fictional.

| CSV | Fields |
|---|---|
| emergencies.csv | emergency_id (fictional), reported_at (local YYYY-MM-DD HH:MM:SS), emergency_type (sample category), response_minutes (numeric or blank) |
| ambulances.csv | ambulance_id (fictional), status (ACTIVE, AVAILABLE, BUSY, MAINTENANCE, OFFLINE) |
| hospitals.csv | hospital_id (fictional), status (OPEN, FULL, CLOSED), icu_beds_available (numeric) |

Provisional rules, pending agreement with Shravan and Raman:
- Today is the selected reference date; week is the inclusive seven-day period ending on it.
- Active ambulance means normalized status ACTIVE only.
- Available hospital means normalized status OPEN and ICU beds > 0.
- Average response is today's valid durations from 0 through 240 minutes, rounded to two decimals; 0 with an informational no-data alert if none.
- Unknown status values are warned about and excluded. Bad timestamps/numbers are excluded; missing columns cause an error.
- Priority warnings: OPEN hospital with 0–2 ICU beds; today's mean response > 20 minutes.
- Hourly observed counts contain 24 entries; estimate output is a separate 3-hour trailing moving average, labeled `isEstimate: true`, not a validated future prediction.
- Required six top-level summary fields are fixed by Raj's task plan; nested chart structure and thresholds must be agreed before integration.
