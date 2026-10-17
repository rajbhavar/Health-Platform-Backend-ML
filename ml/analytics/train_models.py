import argparse
import json

from datetime import date
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error


BASE = Path(__file__).resolve().parent
DATA = BASE / "data"

MODEL_VERSION = "experimental-1"
RANDOM_STATE = 42
TEST_FRACTION = 0.20


def load_dataset(filename, required_columns):
    path = DATA / filename

    if not path.is_file():
        raise FileNotFoundError(
            f"Missing {path}. Run generate_sample_data.py first."
        )

    frame = pd.read_csv(path)

    missing = sorted(
        set(required_columns) - set(frame.columns)
    )

    if missing:
        raise ValueError(
            f"{filename}: missing columns {missing}"
        )

    if frame.empty:
        raise ValueError(
            f"{filename}: dataset is empty"
        )

    return frame


def clean_numeric(frame, columns):
    frame = frame.copy()

    for column in columns:
        frame[column] = pd.to_numeric(
            frame[column],
            errors="coerce",
        )

    frame = frame.replace(
        [np.inf, -np.inf],
        np.nan,
    )

    frame = frame.dropna(
        subset=columns,
    )

    if frame.empty:
        raise ValueError(
            "No valid training rows remain after cleaning"
        )

    return frame


def make_model():
    return HistGradientBoostingRegressor(
        max_iter=120,
        learning_rate=0.06,
        max_leaf_nodes=15,
        min_samples_leaf=10,
        l2_regularization=0.1,
        random_state=RANDOM_STATE,
    )


def chronological_split(frame, time_column):
    """
    Split on unique timestamps/dates so records at the same
    time cannot appear in both training and testing.
    """

    frame = frame.sort_values(
        time_column,
    ).reset_index(drop=True)

    unique_times = (
        frame[time_column]
        .drop_duplicates()
        .sort_values()
        .reset_index(drop=True)
    )

    if len(unique_times) < 10:
        raise ValueError(
            "At least 10 distinct time points are required"
        )

    split_index = int(
        len(unique_times) * (1 - TEST_FRACTION)
    )

    split_index = max(
        1,
        min(split_index, len(unique_times) - 1),
    )

    cutoff = unique_times.iloc[split_index]

    train = frame[
        frame[time_column] < cutoff
    ].copy()

    test = frame[
        frame[time_column] >= cutoff
    ].copy()

    if train.empty or test.empty:
        raise ValueError(
            "Chronological split produced an empty partition"
        )

    return train, test


def train_and_evaluate(
    *,
    name,
    frame,
    features,
    target,
    time_column,
    baseline_function,
    prediction_unit,
    prediction_horizon,
):
    train, test = chronological_split(
        frame,
        time_column,
    )

    model = make_model()

    model.fit(
        train[features],
        train[target],
    )

    predictions = model.predict(
        test[features]
    )

    # Counts, durations and available beds cannot be negative.
    predictions = np.maximum(
        predictions,
        0,
    )

    baseline_predictions = np.asarray(
        baseline_function(train, test),
        dtype=float,
    )

    if len(baseline_predictions) != len(test):
        raise ValueError(
            f"{name}: baseline length mismatch"
        )

    actual = test[target].to_numpy(
        dtype=float,
    )

    model_mae = float(
        mean_absolute_error(
            actual,
            predictions,
        )
    )

    baseline_mae = float(
        mean_absolute_error(
            actual,
            baseline_predictions,
        )
    )

    # Fit the deployable experimental model on all available
    # history AFTER obtaining the held-out evaluation metrics.
    final_model = make_model()

    final_model.fit(
        frame[features],
        frame[target],
    )

    artifact = {
        "model": final_model,
        "modelName": name,
        "modelVersion": MODEL_VERSION,
        "features": features,
        "target": target,
        "predictionUnit": prediction_unit,
        "predictionHorizon": prediction_horizon,
        "simulated": True,
        "trainingRows": int(len(frame)),
        "lastTrainingDate": str(
            frame[time_column].max()
        ),
    }

    model_path = DATA / f"{name}_model.joblib"

    joblib.dump(
        artifact,
        model_path,
    )

    report = {
        "modelName": name,
        "modelVersion": MODEL_VERSION,
        "simulated": True,
        "trainingRows": int(len(train)),
        "testingRows": int(len(test)),
        "trainingEnd": str(
            train[time_column].max()
        ),
        "testingStart": str(
            test[time_column].min()
        ),
        "modelMAE": round(model_mae, 4),
        "baselineMAE": round(baseline_mae, 4),
        "beatsBaseline": bool(
            model_mae < baseline_mae
        ),
        "predictionUnit": prediction_unit,
        "predictionHorizon": prediction_horizon,
        "artifact": model_path.name,
    }

    print(
        f"\n===== {name.upper()} ====="
    )

    print(
        f"Training rows: {len(train)}"
    )

    print(
        f"Testing rows:  {len(test)}"
    )

    print(
        f"Model MAE:    {model_mae:.4f}"
    )

    print(
        f"Baseline MAE: {baseline_mae:.4f}"
    )

    print(
        f"Beats baseline: {report['beatsBaseline']}"
    )

    print(
        f"Saved: {model_path.name}"
    )

    return report


def train_demand():
    frame = load_dataset(
        "demand_history.csv",
        [
            "timestamp",
            "hour",
            "day_of_week",
            "is_weekend",
            "emergency_count",
        ],
    )

    frame["timestamp"] = pd.to_datetime(
        frame["timestamp"],
        utc=True,
        errors="coerce",
    )

    frame = frame.dropna(
        subset=["timestamp"]
    )

    features = [
        "hour",
        "day_of_week",
        "is_weekend",
    ]

    frame = clean_numeric(
        frame,
        features + ["emergency_count"],
    )

    if (frame["emergency_count"] < 0).any():
        raise ValueError(
            "Emergency counts cannot be negative"
        )

    def baseline(train, test):
        # Historical average for each hour.
        hourly_mean = train.groupby(
            "hour"
        )["emergency_count"].mean()

        overall_mean = float(
            train["emergency_count"].mean()
        )

        return (
            test["hour"]
            .map(hourly_mean)
            .fillna(overall_mean)
            .to_numpy()
        )

    return train_and_evaluate(
        name="demand",
        frame=frame,
        features=features,
        target="emergency_count",
        time_column="timestamp",
        baseline_function=baseline,
        prediction_unit="emergencies_per_hour",
        prediction_horizon="specified_future_hour",
    )


def train_response_time():
    frame = load_dataset(
        "response_history.csv",
        [
            "reported_at",
            "hour",
            "distance_km",
            "traffic_level",
            "weather_level",
            "priority_level",
            "peak_hour",
            "response_minutes",
        ],
    )

    frame["reported_at"] = pd.to_datetime(
        frame["reported_at"],
        utc=True,
        errors="coerce",
    )

    frame = frame.dropna(
        subset=["reported_at"]
    )

    features = [
        "hour",
        "distance_km",
        "traffic_level",
        "weather_level",
        "priority_level",
        "peak_hour",
    ]

    frame = clean_numeric(
        frame,
        features + ["response_minutes"],
    )

    if (
        (frame["response_minutes"] <= 0).any()
        or (frame["distance_km"] < 0).any()
    ):
        raise ValueError(
            "Invalid response time or distance"
        )

    def baseline(train, test):
        median = float(
            train["response_minutes"].median()
        )

        return np.full(
            len(test),
            median,
        )

    return train_and_evaluate(
        name="response_time",
        frame=frame,
        features=features,
        target="response_minutes",
        time_column="reported_at",
        baseline_function=baseline,
        prediction_unit="minutes",
        prediction_horizon="estimated_at_dispatch",
    )


def train_hospital_capacity():
    frame = load_dataset(
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
    )

    frame["date"] = pd.to_datetime(
        frame["date"],
        errors="coerce",
    )

    frame = frame.dropna(
        subset=["date"]
    )

    numeric_columns = [
        "total_icu_beds",
        "available_beds",
        "admissions_today",
        "discharges_today",
        "next_day_available_beds",
    ]

    frame = clean_numeric(
        frame,
        numeric_columns,
    )

    if (frame[numeric_columns] < 0).any().any():
        raise ValueError(
            "Hospital capacity values cannot be negative"
        )

    if (
        frame["next_day_available_beds"]
        > frame["total_icu_beds"]
    ).any():
        raise ValueError(
            "Available beds exceed hospital capacity"
        )

    frame = frame.sort_values(
        ["hospital_id", "date"]
    ).copy()

    # The generator's next_day_available_beds column is
    # deterministically calculated from the CURRENT day's
    # admissions/discharges.
    #
    # To avoid training on that trivial arithmetic, we use
    # the end-of-day capacity as today's known state and
    # shift the TARGET forward one additional day.
    #
    # This predicts tomorrow's end-of-day availability,
    # without supplying tomorrow's admissions/discharges.

    frame["current_end_available"] = (
        frame["next_day_available_beds"]
    )

    frame["target_tomorrow_end_available"] = (
        frame.groupby("hospital_id")[
            "current_end_available"
        ].shift(-1)
    )

    frame["target_date"] = (
        frame.groupby("hospital_id")[
            "date"
        ].shift(-1)
    )

    frame["day_of_week"] = (
        frame["date"].dt.dayofweek
    )

    frame = frame.dropna(
        subset=[
            "target_tomorrow_end_available",
            "target_date",
        ]
    )

    # Require consecutive dates for the one-day horizon.
    frame = frame[
        (
            frame["target_date"] - frame["date"]
        ).dt.days == 1
    ].copy()

    features = [
        "total_icu_beds",
        "current_end_available",
        "admissions_today",
        "discharges_today",
        "day_of_week",
    ]

    if frame.empty:
        raise ValueError(
            "No consecutive hospital capacity records"
        )

    def baseline(train, test):
        # Simple persistence forecast:
        # tomorrow's available beds = today's available beds.
        return test[
            "current_end_available"
        ].to_numpy(dtype=float)

    return train_and_evaluate(
        name="hospital_capacity",
        frame=frame,
        features=features,
        target="target_tomorrow_end_available",
        time_column="date",
        baseline_function=baseline,
        prediction_unit="available_icu_beds",
        prediction_horizon="next_day_end_of_day",
    )


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Train experimental models "
            "on fictional healthcare data."
        )
    )

    parser.add_argument(
        "--model",
        choices=[
            "all",
            "demand",
            "response_time",
            "hospital_capacity",
        ],
        default="all",
    )

    args = parser.parse_args()

    DATA.mkdir(
        parents=True,
        exist_ok=True,
    )

    trainers = {
        "demand": train_demand,
        "response_time": train_response_time,
        "hospital_capacity": train_hospital_capacity,
    }

    if args.model == "all":
        selected = trainers
    else:
        selected = {
            args.model: trainers[args.model]
        }

    reports = {}

    for name, trainer in selected.items():
        reports[name] = trainer()

    report_path = (
        DATA / "model_evaluation.json"
    )

    report = {
        "modelVersion": MODEL_VERSION,
        "simulated": True,
        "trainingData": "FICTIONAL",
        "models": reports,
        "note": (
            "Experimental demonstration models. "
            "Not validated for clinical, dispatch, "
            "or hospital operational decisions."
        ),
    }

    report_path.write_text(
        json.dumps(
            report,
            indent=2,
        ) + "\n",
        encoding="utf-8",
    )

    print(
        "\n[PASS] Training and evaluation completed"
    )

    print(
        f"[SAVED] {report_path}"
    )


if __name__ == "__main__":
    main()
