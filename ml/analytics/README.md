# Raj — AI/ML Home Page Phase 1 (AI-07, AI-08, AI-09)

**FAKE SAMPLE DATA ONLY.** No real patient information and no Node/React integration.

## PowerShell commands from project root
```powershell
cd ml\\analytics
python -m venv .venv
.\\.venv\\Scripts\\python.exe -m pip install -r requirements.txt
.\\.venv\\Scripts\\python.exe generate_sample_data.py --date 2026-10-13
.\\.venv\\Scripts\\python.exe analytics.py --date 2026-10-13
.\\.venv\\Scripts\\python.exe -m unittest discover -s tests -v
```
Omit `--date` in both scripts for current local date. `analytics.py` reads CSV inputs and regenerates `dashboard_summary.json` plus `analytics_details.json` without notebook edits. The generator overwrites the fake CSVs, not the analytics script.

## Files and handover
`data/*.csv`: fictional inputs. `generate_sample_data.py`: repeatable generator. `analytics.py`: KPI, alerts, observed hourly counts and estimated moving-average trend. `dashboard_summary.json`: exact six required top-level keys for Shravan's future GET /api/dashboard/summary. `analytics_details.json`: separate weekly KPI and labeled estimates, not part of the required API contract. `tests/`: empty and invalid input tests. `data_dictionary.md`: provisional rules.

Share sample JSON with Shravan and Raman. Confirm status meanings, KPI rules, alert levels and hourly item schema before integration; never rename the six required fields independently. Use a feature branch and PR, not main/dev. Run tests, self-review, document blockers, and demonstrate generator → analytics → JSON → tests.

Limitations: sample-only thresholds, local naive timestamps, no trained forecasting model, no real-time data, no API integration. Zero average on no data is accompanied by an info alert.
