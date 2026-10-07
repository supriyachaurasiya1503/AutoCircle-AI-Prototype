import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

MODEL_PATH = Path(__file__).resolve().parent / "artifacts" / "final_battery_model.joblib"
METRICS_PATH = Path(__file__).resolve().parent / "final_battery_metrics.json"
GROUP = "battery_id"
TARGET = "soh"


def build_history_features(frame):
    """Build features for row t exclusively from observations before t."""
    result = frame.sort_values([GROUP, "cycle"], kind="stable").copy()
    groups = result.groupby(GROUP, sort=False)
    result["initial_capacity"] = groups["capacity"].transform("first")
    result["cycle_index"] = groups.cumcount() + 1
    for column in ("capacity", TARGET, "voltage", "temperature"):
        for lag in (1, 2, 3, 5, 10):
            result[f"{column}_lag_{lag}"] = groups[column].shift(lag)
        previous = groups[column].shift(1)
        for window in (3, 5, 10):
            result[f"{column}_mean_prev_{window}"] = previous.groupby(result[GROUP]).transform(
                lambda values: values.rolling(window, min_periods=1).mean()
            )
            result[f"{column}_std_prev_{window}"] = previous.groupby(result[GROUP]).transform(
                lambda values: values.rolling(window, min_periods=2).std()
            )
        result[f"{column}_ewm_prev"] = previous.groupby(result[GROUP]).transform(
            lambda values: values.ewm(span=5, min_periods=1, adjust=False).mean()
        )
    result["soh_delta_lag_1"] = groups[TARGET].shift(1) - groups[TARGET].shift(2)
    result["capacity_delta_lag_1"] = groups["capacity"].shift(1) - groups["capacity"].shift(2)
    result["capacity_ratio_lag_1"] = result["capacity_lag_1"] / result["initial_capacity"]
    result["soh_slope_prev_5"] = groups[TARGET].shift(1).groupby(result[GROUP]).transform(
        lambda values: values.rolling(5, min_periods=2).apply(
            lambda window: np.polyfit(np.arange(len(window)), window, 1)[0], raw=True
        )
    )
    return result


class FinalBatterySOHModel:
    def __init__(self, model_path=MODEL_PATH, metrics_path=METRICS_PATH):
        self.model_path = Path(model_path)
        self.metrics_path = Path(metrics_path)
        self.artifact = joblib.load(self.model_path) if self.model_path.exists() else None
        self.metrics_data = (
            json.loads(self.metrics_path.read_text(encoding="utf-8"))
            if self.metrics_path.exists()
            else None
        )

    @property
    def is_loaded(self):
        return self.artifact is not None

    def get_metrics(self):
        if self.metrics_data is None:
            raise FileNotFoundError(f"Evaluation metrics not found at {self.metrics_path}")
        return self.metrics_data

    def predict_next(self, battery_id, history, next_cycle=None):
        if self.artifact is None:
            raise FileNotFoundError(f"Trained battery model not found at {self.model_path}")
        if not history:
            raise ValueError("At least one historical cycle is required.")

        observations = pd.DataFrame(history).copy()
        required = {"cycle", "voltage", "temperature", "capacity"}
        missing = required.difference(observations.columns)
        if missing:
            raise ValueError(f"Missing historical fields: {', '.join(sorted(missing))}")
        if observations[list(required)].isna().any().any():
            raise ValueError("Historical cycle values must not contain missing values.")
        observations[GROUP] = str(battery_id)
        observations = observations.sort_values("cycle", kind="stable").reset_index(drop=True)
        if int(observations["cycle"].iloc[0]) != 1:
            raise ValueError("History must include the battery's first recorded cycle for initial-capacity calibration.")
        if observations["cycle"].duplicated().any():
            raise ValueError("Historical cycle numbers must be unique.")
        if (observations["capacity"] <= 0).any():
            raise ValueError("Historical capacity values must be positive.")
        observations[TARGET] = observations["capacity"] / observations["capacity"].iloc[0]

        last_cycle = int(observations["cycle"].iloc[-1])
        next_cycle = last_cycle + 1 if next_cycle is None else int(next_cycle)
        if next_cycle != last_cycle + 1:
            raise ValueError("This model is validated only for exactly one physical cycle ahead.")
        pending = pd.DataFrame([{
            GROUP: str(battery_id),
            "cycle": next_cycle,
            "voltage": np.nan,
            "temperature": np.nan,
            "capacity": np.nan,
            TARGET: np.nan,
        }])
        all_rows = pd.concat([observations, pending], ignore_index=True)
        features = build_history_features(all_rows).iloc[[-1]][self.artifact["feature_columns"]]
        raw_prediction = float(self.artifact["estimator"].predict(features)[0])
        formulation = self.artifact["formulation"]
        if formulation == "delta_soh":
            predicted_soh = float(observations[TARGET].iloc[-1]) + raw_prediction
        elif formulation == "degradation":
            predicted_soh = float(observations[TARGET].iloc[-1]) - raw_prediction
        elif formulation in {"capacity_to_soh", "remaining_capacity_to_soh"}:
            predicted_soh = raw_prediction / float(observations["capacity"].iloc[0])
        elif formulation == "normalized_degradation":
            predicted_soh = 1.0 - raw_prediction
        else:
            predicted_soh = raw_prediction

        protocol = self.get_metrics()["known_battery_future_cycles"]
        exact_cycle_test = protocol["exact_one_physical_cycle_test"]
        return {
            "battery_id": str(battery_id),
            "next_cycle": next_cycle,
            "soh": float(predicted_soh * 100.0),
            "model_used": f"{self.artifact['model_name']} ({formulation})",
            "prediction_mode": "exactly one physical cycle ahead using prior observed cycles",
            "mae_percentage_points": exact_cycle_test["mae_percentage_points"],
            "test_rows": exact_cycle_test["rows"],
            "test_coverage": protocol["exact_one_physical_cycle_coverage"]["fraction"],
        }
