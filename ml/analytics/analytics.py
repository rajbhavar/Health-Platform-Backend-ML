import argparse
import json
import sys

from datetime import date, timedelta
from pathlib import Path

import pandas as pd


BASE = Path(__file__).resolve().parent

COLUMNS = {
    "emergencies.csv": [
        "emergency_id",
        "reported_at",
        "emergency_type",
        "response_minutes",
    ],
    "ambulances.csv": [
        "ambulance_id",
        "status",
    ],
    "hospitals.csv": [
        "hospital_id",
        "status",
        "icu_beds_available",
    ],
}

VALID_AMBULANCE_STATUSES = {
    "ACTIVE",
    "AVAILABLE",
    "BUSY",
    "MAINTENANCE",
    "OFFLINE",
}

VALID_HOSPITAL_STATUSES = {
    "OPEN",
    "FULL",
    "CLOSED",
}


def read_csv(folder, filename):
    path = Path(folder) / filename

    if not path.is_file():
        raise FileNotFoundError(
            f"Missing input file: {path}. "
            "Run generate_sample_data.py first."
        )

    dataframe = pd.read_csv(
        path,
        dtype=str,
        keep_default_na=False,
    )

    missing = sorted(
        set(COLUMNS[filename]) - set(dataframe.columns)
    )

    if missing:
        raise ValueError(
            f"{filename}: missing required columns {missing}"
        )

    return dataframe


def warn(message):
    # Send diagnostic messages to stderr so that stdout
    # can remain suitable for normal command-line output.
    print(
        f"DATA QUALITY WARNING: {message}",
        file=sys.stderr,
    )


def calculate(folder=BASE / "data", day=None):
    """
    Return:
        summary: dashboard metrics
        details: supplementary analytics and estimates
    """

    if isinstance(day, str):
        selected_day = date.fromisoformat(day)
    elif day is None:
        selected_day = date.today()
    elif isinstance(day, date):
        selected_day = day
    else:
        raise ValueError(
            "day must be an ISO date string or datetime.date"
        )

    emergencies = read_csv(folder, "emergencies.csv")
    ambulances = read_csv(folder, "ambulances.csv")
    hospitals = read_csv(folder, "hospitals.csv")

    # -------------------------------
    # Clean emergency records
    # -------------------------------

    emergencies["when"] = pd.to_datetime(
        emergencies["reported_at"],
        errors="coerce",
        utc=True,
    )

    emergencies["minutes"] = pd.to_numeric(
        emergencies["response_minutes"],
        errors="coerce",
    )

    # Prototype data-quality rule:
    # response time must be between 0 and 240 minutes.
    emergencies.loc[
        (emergencies["minutes"] < 0)
        | (emergencies["minutes"] > 240),
        "minutes",
    ] = float("nan")

    invalid_timestamps = int(
        emergencies["when"].isna().sum()
    )

    invalid_response_times = int(
        emergencies["minutes"].isna().sum()
    )

    if invalid_timestamps:
        warn(
            f"invalid timestamps: {invalid_timestamps}"
        )

    if invalid_response_times:
        warn(
            "invalid/missing response minutes: "
            f"{invalid_response_times}"
        )

    # -------------------------------
    # Clean ambulance records
    # -------------------------------

    ambulances["clean"] = (
        ambulances["status"]
        .str.strip()
        .str.upper()
    )

    unknown_ambulance_statuses = sorted(
        set(ambulances["clean"])
        - VALID_AMBULANCE_STATUSES
    )

    if unknown_ambulance_statuses:
        warn(
            "ambulance unknown statuses: "
            f"{unknown_ambulance_statuses}"
        )

    # -------------------------------
    # Clean hospital records
    # -------------------------------

    hospitals["clean"] = (
        hospitals["status"]
        .str.strip()
        .str.upper()
    )

    hospitals["beds"] = pd.to_numeric(
        hospitals["icu_beds_available"],
        errors="coerce",
    )

    # ICU beds must be a non-negative whole number.
    invalid_beds = (
        hospitals["beds"].isna()
        | (hospitals["beds"] < 0)
        | (hospitals["beds"] % 1 != 0)
    )

    hospitals.loc[
        invalid_beds,
        "beds",
    ] = float("nan")

    invalid_bed_count = int(
        invalid_beds.sum()
    )

    if invalid_bed_count:
        warn(
            "invalid/missing ICU beds: "
            f"{invalid_bed_count}"
        )

    unknown_hospital_statuses = sorted(
        set(hospitals["clean"])
        - VALID_HOSPITAL_STATUSES
    )

    if unknown_hospital_statuses:
        warn(
            "hospital unknown statuses: "
            f"{unknown_hospital_statuses}"
        )

    # -------------------------------
    # Select daily and weekly records
    # -------------------------------

    emergency_dates = emergencies["when"].dt.date

    current = emergencies[
        emergency_dates == selected_day
    ]

    week = emergencies[
        (emergency_dates >= selected_day - timedelta(days=6))
        & (emergency_dates <= selected_day)
    ]

    valid_response_times = (
        current["minutes"].dropna()
    )

    if len(valid_response_times):
        average_response = round(
            float(valid_response_times.mean()),
            2,
        )
    else:
        average_response = 0.0

    # -------------------------------
    # Hourly observed demand
    # -------------------------------

    hourly_counts = (
        current.groupby(
            current["when"].dt.hour
        )
        .size()
        .to_dict()
    )

    observed = [
        {
            "hour": f"{hour:02d}:00",
            "count": int(
                hourly_counts.get(hour, 0)
            ),
        }
        for hour in range(24)
    ]

    # -------------------------------
    # Priority alerts
    # -------------------------------

    alerts = []

    low_icu_capacity = (
        (hospitals["clean"] == "OPEN")
        & (hospitals["beds"] >= 0)
        & (hospitals["beds"] <= 2)
    )

    if low_icu_capacity.any():
        alerts.append({
            "level": "warning",
            "text": (
                "Sample alert: an open hospital has "
                "2 or fewer ICU beds."
            ),
        })

    if (
        len(valid_response_times)
        and average_response > 20
    ):
        alerts.append({
            "level": "warning",
            "text": (
                "Sample alert: average response "
                "exceeds 20 minutes."
            ),
        })

    if not len(valid_response_times):
        alerts.append({
            "level": "info",
            "text": (
                "No valid response times "
                "for the selected date."
            ),
        })

    if current.empty:
        alerts.append({
            "level": "info",
            "text": (
                "No emergencies for the selected date."
            ),
        })

    # -------------------------------
    # Dashboard summary
    # -------------------------------

    summary = {
        "emergencyCountToday": int(len(current)),
        "activeAmbulances": int(
            (ambulances["clean"] == "ACTIVE").sum()
        ),
        "hospitalsAvailable": int(
            (
                (hospitals["clean"] == "OPEN")
                & (hospitals["beds"] > 0)
            ).sum()
        ),
        "avgResponseMinutes": average_response,
        "priorityAlerts": alerts,
        "emergenciesByHour": observed,
    }

    # -------------------------------
    # Simple descriptive trend estimate
    # -------------------------------

    counts = [
        item["count"]
        for item in observed
    ]

    estimate = []

    for hour in range(24):
        recent_counts = counts[
            max(0, hour - 2):hour + 1
        ]

        estimated_count = round(
            sum(recent_counts) / len(recent_counts),
            2,
        )

        estimate.append({
            "hour": f"{hour:02d}:00",
            "estimatedCount": estimated_count,
            "isEstimate": True,
        })

    details = {
        "referenceDate": selected_day.isoformat(),
        "weeklyEmergencyCount": int(len(week)),
        "hourlyDemandTrendEstimate": estimate,
        "note": (
            "FAKE DATA. Hourly values are simple "
            "descriptive moving-average estimates, "
            "not trained ML predictions or observed counts."
        ),
    }

    return summary, details


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Generate emergency healthcare "
            "dashboard analytics from CSV data."
        )
    )

    parser.add_argument(
        "--date",
        default=None,
        help="Reference date in YYYY-MM-DD format",
    )

    parser.add_argument(
        "--data-dir",
        type=Path,
        default=BASE / "data",
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=BASE / "dashboard_summary.json",
    )

    args = parser.parse_args()

    summary, details = calculate(
        folder=args.data_dir,
        day=args.date,
    )

    args.output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    details_path = (
        args.output.parent / "analytics_details.json"
    )

    args.output.write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )

    details_path.write_text(
        json.dumps(details, indent=2) + "\n",
        encoding="utf-8",
    )

    print(f"Wrote {args.output}")
    print(f"Wrote {details_path}")


if __name__ == "__main__":
    main()
