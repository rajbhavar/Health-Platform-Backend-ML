import argparse
import json
import math
import os
import tempfile

from datetime import timedelta
from pathlib import Path

import joblib
import pandas as pd


BASE = Path(__file__).resolve().parent
DATA = BASE / "data"

MODEL_NAMES = (
    "demand",
    "response_time",
    "hospital_capacity",
)


def load_artifact(name):
    path = DATA / f"{name}_model.joblib"

    if not path.is_file():
        raise FileNotFoundError(
            f"Missing {path.name}. Run train_models.py --model all first."
        )

    # joblib uses pickle internally. Never load an untrusted file.
    artifact = joblib.load(path)

    required = {
        "model",
        "modelName",
        "modelVersion",
        "features",
        "simulated",
    }

    if not isinstance(artifact, dict) or not required.issubset(artifact):
        raise ValueError(f"Invalid model artifact: {path.name}")

    if artifact["modelName"] != name or artifact["simulated"] is not True:
        raise ValueError(f"Unexpected model metadata: {path.name}")

    return artifact


def load_evaluation():
    path = DATA / "model_evaluation.json"

    if not path.is_file():
        raise FileNotFoundError(
            "Missing model_evaluation.json. Train all models first."
        )

    report = json.loads(path.read_text(encoding="utf-8"))

    if report.get("simulated") is not True:
        raise ValueError("Expected fictional-data evaluation metadata")

    models = report.get("models")

    if not isinstance(models, dict):
        raise ValueError("Invalid model evaluation report")

    for name in MODEL_NAMES:
        if name not in models:
            raise ValueError(
                f"Missing evaluation for {name}. "
                "Run train_models.py --model all."
            )

    return report


def predict_value(artifact, features, maximum=None):
    expected = artifact["features"]

    if set(features) != set(expected):
        raise ValueError(
            f"{artifact['modelName']}: prediction feature mismatch"
        )

    frame = pd.DataFrame(
        [{name: features[name] for name in expected}],
        columns=expected,
    )

    for column in expected:
        frame[column] = pd.to_numeric(
            frame[column],
            errors="raise",
        )

    if not all(
        math.isfinite(float(frame.iloc[0][column]))
        for column in expected
    ):
        raise ValueError("Prediction features must be finite")

    value = float(artifact["model"].predict(frame)[0])

    if not math.isfinite(value):
        raise ValueError("Model returned a non-finite prediction")

    value = max(0.0, value)

    if maximum is not None:
        value = min(value, float(maximum))

    return round(value, 2)


def predict_demand(artifact):
    path = DATA / "demand_history.csv"

    frame = pd.read_csv(path)

    frame["timestamp"] = pd.to_datetime(
        frame["timestamp"],
        utc=True,
        errors="coerce",
    )

    frame = frame.dropna(subset=["timestamp"])

    if frame.empty:
        raise ValueError("Demand history contains no valid timestamps")

    last_hour = frame["timestamp"].max()

    predictions = []

    for offset in range(1, 25):
        target = last_hour + timedelta(hours=offset)

        features = {
            "hour": target.hour,
            "day_of_week": target.weekday(),
            "is_weekend": int(target.weekday() >= 5),
        }

        predicted = predict_value(
            artifact,
            features,
        )

        predictions.append({
            "timestamp": target.isoformat().replace("+00:00", "Z"),
            "estimatedEmergencyCount": predicted,
        })

    return {
        "horizon": "next_24_hours_after_training_history",
        "unit": "emergencies_per_hour",
        "predictions": predictions,
    }


def predict_response_time(artifact, scenario):
    if not 0 <= scenario["hour"] <= 23:
        raise ValueError("hour must be between 0 and 23")

    if not 0 <= scenario["distance_km"] <= 100:
        raise ValueError("distance_km must be between 0 and 100")

    for key in ("traffic_level", "weather_level"):
        if scenario[key] not in (0, 1, 2):
            raise ValueError(f"{key} must be 0, 1 or 2")

    if scenario["priority_level"] not in (1, 2, 3):
        raise ValueError("priority_level must be 1, 2 or 3")

    hour = scenario["hour"]

    features = {
        **scenario,
        "peak_hour": int(
            7 <= hour <= 10 or 17 <= hour <= 21
        ),
    }

    predicted = predict_value(
        artifact,
        features,
    )

    return {
        "scenario": features,
        "estimatedResponseMinutes": predicted,
        "unit": "minutes",
        "note": (
            "Illustrative input scenario only. "
            "Not a live ambulance ETA."
        ),
    }


def predict_capacity(artifact):
    path = DATA / "capacity_history.csv"

    frame = pd.read_csv(path)

    required = [
        "hospital_id",
        "date",
        "total_icu_beds",
        "next_day_available_beds",
        "admissions_today",
        "discharges_today",
    ]

    missing = set(required) - set(frame.columns)

    if missing:
        raise ValueError(
            f"Capacity history missing columns: {sorted(missing)}"
        )

    frame["date"] = pd.to_datetime(
        frame["date"],
        errors="coerce",
    )

    frame = frame.dropna(subset=["date"])

    if frame.empty:
        raise ValueError("Capacity history contains no valid dates")

    # The training script defines current_end_available
    # as next_day_available_beds in this fictional dataset.
    # Take the latest end-of-day observation per hospital.
    latest = (
        frame.sort_values("date")
        .groupby("hospital_id", sort=True)
        .tail(1)
        .sort_values("hospital_id")
    )

    predictions = []

    for _, row in latest.iterrows():
        total = int(row["total_icu_beds"])
        current_available = int(row["next_day_available_beds"])

        if total <= 0 or not 0 <= current_available <= total:
            raise ValueError("Invalid hospital bed counts")

        observation_date = row["date"].date()

        features = {
            "total_icu_beds": total,
            "current_end_available": current_available,
            "admissions_today": int(row["admissions_today"]),
            "discharges_today": int(row["discharges_today"]),
            "day_of_week": observation_date.weekday(),
        }

        predicted = predict_value(
            artifact,
            features,
            maximum=total,
        )

        predictions.append({
            "hospitalId": str(row["hospital_id"]),
            "observationDate": observation_date.isoformat(),
            "targetDate": (
                observation_date + timedelta(days=1)
            ).isoformat(),
            "totalIcuBeds": total,
            "currentAvailableBeds": current_available,
            "estimatedAvailableBeds": predicted,
        })

    return {
        "horizon": "next_day_end_of_day",
        "unit": "available_icu_beds",
        "predictions": predictions,
    }


def write_json_atomically(path, payload):
    """
    Replace the output only after a complete JSON document
    has been written successfully.
    """
    path.parent.mkdir(parents=True, exist_ok=True)

    temporary_path = None

    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            suffix=".tmp",
            prefix="predictions-",
            dir=path.parent,
            delete=False,
        ) as file:
            temporary_path = Path(file.name)
            json.dump(payload, file, indent=2, allow_nan=False)
            file.write("\n")

        os.replace(temporary_path, path)

    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(
        description="Generate fictional experimental ML predictions"
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=BASE / "predictions.json",
    )

    args = parser.parse_args()

    evaluation = load_evaluation()

    artifacts = {
        name: load_artifact(name)
        for name in MODEL_NAMES
    }

    versions = {
        artifact["modelVersion"]
        for artifact in artifacts.values()
    }

    if len(versions) != 1 or evaluation.get("modelVersion") not in versions:
        raise ValueError(
            "Model versions do not match. Retrain all models together."
        )

    # Demonstration-only dispatch scenario.
    # No real ambulance, route or patient is involved.
    response_scenario = {
        "hour": 18,
        "distance_km": 6.5,
        "traffic_level": 1,
        "weather_level": 0,
        "priority_level": 2,
    }

    payload = {
        "success": True,
        "simulated": True,
        "source": "EXPERIMENTAL_PYTHON_ML",
        "modelVersion": evaluation["modelVersion"],
        "trainingData": "FICTIONAL",
        "models": {
            name: {
                "modelMAE": evaluation["models"][name]["modelMAE"],
                "baselineMAE": evaluation["models"][name]["baselineMAE"],
                "beatsBaseline": evaluation["models"][name]["beatsBaseline"],
            }
            for name in MODEL_NAMES
        },
        "predictions": {
            "demand": predict_demand(artifacts["demand"]),
            "responseTime": predict_response_time(
                artifacts["response_time"],
                response_scenario,
            ),
            "hospitalCapacity": predict_capacity(
                artifacts["hospital_capacity"]
            ),
        },
        "warning": (
            "Fictional-data experimental predictions only. "
            "Not validated for clinical, dispatch or "
            "hospital operational decisions."
        ),
    }

    write_json_atomically(args.output, payload)

    print(f"[PASS] Wrote {args.output}")
    print(
        "[PASS] Demand predictions:",
        len(payload["predictions"]["demand"]["predictions"]),
    )
    print(
        "[PASS] Hospital predictions:",
        len(
            payload["predictions"]["hospitalCapacity"]["predictions"]
        ),
    )
    print("[PASS] Response-time demonstration prediction generated")


if __name__ == "__main__":
    main()
