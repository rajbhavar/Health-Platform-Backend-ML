import argparse
import csv
import math
import random

from datetime import date, datetime, timedelta
from pathlib import Path


BASE = Path(__file__).resolve().parent
DATA_DIR = BASE / "data"

RANDOM_SEED = 7
HISTORY_DAYS = 120


def save_csv(filename, columns, rows):
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    path = DATA_DIR / filename

    with path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.writer(file)
        writer.writerow(columns)
        writer.writerows(rows)

    print(
        f"[GENERATED] {path.name}: {len(rows)} fictional records"
    )


def generate_dashboard_data(day, rng):
    """
    Preserve the existing dashboard CSV schema.
    """

    emergencies = []

    for offset in range(7):
        current_day = day - timedelta(days=offset)

        for index in range(8 + offset % 4):
            reported_hour = index * 3 % 24

            response_minutes = (
                ""
                if offset == 3 and index == 0
                else rng.randint(5, 32)
            )

            emergencies.append([
                f"FAKE-E-{offset}-{index}",
                (
                    f"{current_day.isoformat()}T"
                    f"{reported_hour:02d}:00:00Z"
                ),
                ["Medical", "Accident", "Fire"][index % 3],
                response_minutes,
            ])

    save_csv(
        "emergencies.csv",
        [
            "emergency_id",
            "reported_at",
            "emergency_type",
            "response_minutes",
        ],
        emergencies,
    )

    ambulance_statuses = [
        "ACTIVE",
        "ACTIVE",
        "ACTIVE",
        "AVAILABLE",
        "BUSY",
        "MAINTENANCE",
        "OFFLINE",
    ]

    save_csv(
        "ambulances.csv",
        [
            "ambulance_id",
            "status",
        ],
        [
            [f"FAKE-A-{index}", status]
            for index, status in enumerate(
                ambulance_statuses,
                start=1,
            )
        ],
    )

    hospital_records = [
        ("OPEN", 5),
        ("OPEN", 1),
        ("OPEN", 0),
        ("FULL", 0),
        ("CLOSED", 3),
    ]

    save_csv(
        "hospitals.csv",
        [
            "hospital_id",
            "status",
            "icu_beds_available",
        ],
        [
            [f"FAKE-H-{index}", status, beds]
            for index, (status, beds) in enumerate(
                hospital_records,
                start=1,
            )
        ],
    )


def generate_demand_history(day, rng):
    """
    Hourly emergency counts over the historical period.

    Prediction target:
        emergency_count

    Available explanatory features:
        hour
        day_of_week
        is_weekend

    These are simulated patterns, not observed real demand.
    """

    rows = []

    start_day = day - timedelta(days=HISTORY_DAYS - 1)

    for day_offset in range(HISTORY_DAYS):
        current_day = start_day + timedelta(days=day_offset)

        day_of_week = current_day.weekday()
        is_weekend = int(day_of_week >= 5)

        for hour in range(24):
            morning_peak = (
                3.0
                if 7 <= hour <= 10
                else 0.0
            )

            evening_peak = (
                4.0
                if 17 <= hour <= 21
                else 0.0
            )

            weekend_effect = (
                1.5
                if is_weekend
                else 0.0
            )

            seasonal_effect = (
                0.8 * math.sin(day_offset / 9)
            )

            noise = rng.uniform(-1.5, 1.5)

            count = max(
                0,
                round(
                    2.0
                    + morning_peak
                    + evening_peak
                    + weekend_effect
                    + seasonal_effect
                    + noise
                ),
            )

            rows.append([
                (
                    f"{current_day.isoformat()}T"
                    f"{hour:02d}:00:00Z"
                ),
                hour,
                day_of_week,
                is_weekend,
                count,
            ])

    save_csv(
        "demand_history.csv",
        [
            "timestamp",
            "hour",
            "day_of_week",
            "is_weekend",
            "emergency_count",
        ],
        rows,
    )


def generate_response_history(day, rng):
    """
    Fictional ambulance dispatch examples.

    Prediction target:
        response_minutes

    Features are assumed to be available when an
    ambulance is assigned. No future trip outcome
    is used as an input.
    """

    rows = []

    start_day = day - timedelta(days=HISTORY_DAYS - 1)

    record_number = 0

    for day_offset in range(HISTORY_DAYS):
        current_day = start_day + timedelta(days=day_offset)

        for _ in range(12):
            record_number += 1

            hour = rng.randint(0, 23)
            distance_km = round(
                rng.uniform(0.5, 18.0),
                2,
            )

            traffic_level = rng.randint(0, 2)
            weather_level = rng.randint(0, 2)
            priority_level = rng.randint(1, 3)

            peak_hour = int(
                7 <= hour <= 10
                or 17 <= hour <= 21
            )

            # Simulated response-time relationship.
            response_minutes = (
                3.5
                + distance_km * 1.65
                + traffic_level * 3.8
                + weather_level * 1.7
                + peak_hour * 2.0
                - (priority_level - 1) * 0.8
                + rng.uniform(-2.5, 2.5)
            )

            response_minutes = round(
                max(1.0, response_minutes),
                2,
            )

            rows.append([
                f"FAKE-R-{record_number}",
                (
                    f"{current_day.isoformat()}T"
                    f"{hour:02d}:00:00Z"
                ),
                hour,
                distance_km,
                traffic_level,
                weather_level,
                priority_level,
                peak_hour,
                response_minutes,
            ])

    save_csv(
        "response_history.csv",
        [
            "dispatch_id",
            "reported_at",
            "hour",
            "distance_km",
            "traffic_level",
            "weather_level",
            "priority_level",
            "peak_hour",
            "response_minutes",
        ],
        rows,
    )


def generate_capacity_history(day, rng):
    """
    Daily fictional ICU-capacity observations.

    Prediction target:
        next_day_available_beds

    Features represent information available at
    the end of the current observation day.
    """

    rows = []

    hospital_ids = [
        "FAKE-H-1",
        "FAKE-H-2",
        "FAKE-H-3",
    ]

    start_day = day - timedelta(days=HISTORY_DAYS - 1)

    for hospital_id in hospital_ids:
        total_beds = rng.randint(12, 30)

        available_beds = rng.randint(
            3,
            total_beds - 2,
        )

        for day_offset in range(HISTORY_DAYS):
            current_day = (
                start_day + timedelta(days=day_offset)
            )

            admissions = rng.randint(0, 5)
            discharges = rng.randint(0, 5)

            # Admissions cannot exceed available beds.
            actual_admissions = min(
                admissions,
                available_beds,
            )

            occupied_beds = (
                total_beds - available_beds
            )

            actual_discharges = min(
                discharges,
                occupied_beds,
            )

            next_available = (
                available_beds
                - actual_admissions
                + actual_discharges
            )

            next_available = max(
                0,
                min(total_beds, next_available),
            )

            rows.append([
                hospital_id,
                current_day.isoformat(),
                total_beds,
                available_beds,
                actual_admissions,
                actual_discharges,
                next_available,
            ])

            available_beds = next_available

    save_csv(
        "capacity_history.csv",
        [
            "hospital_id",
            "date",
            "total_icu_beds",
            "available_beds",
            "admissions_today",
            "discharges_today",
            "next_day_available_beds",
        ],
        rows,
    )


def generate(day):
    """
    Generate all dashboard and experimental ML data.
    """

    if not isinstance(day, date):
        raise TypeError(
            "day must be a datetime.date instance"
        )

    # Independent seeds keep each dataset reproducible.
    generate_dashboard_data(
        day,
        random.Random(RANDOM_SEED),
    )

    generate_demand_history(
        day,
        random.Random(RANDOM_SEED + 1),
    )

    generate_response_history(
        day,
        random.Random(RANDOM_SEED + 2),
    )

    generate_capacity_history(
        day,
        random.Random(RANDOM_SEED + 3),
    )

    print(
        "\n[PASS] All fictional dashboard "
        "and ML training datasets generated."
    )


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Generate fictional healthcare "
            "dashboard and ML training datasets."
        )
    )

    parser.add_argument(
        "--date",
        default=date.today().isoformat(),
        help="Last historical observation date: YYYY-MM-DD",
    )

    args = parser.parse_args()

    generate(
        date.fromisoformat(args.date)
    )


if __name__ == "__main__":
    main()
