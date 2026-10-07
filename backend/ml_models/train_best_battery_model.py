import json
import math
import html
import sys
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import (
    ExtraTreesRegressor,
    GradientBoostingRegressor,
    HistGradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet, LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

sys.path.insert(0, str(Path(__file__).resolve().parent))
from final_battery_model import build_history_features

try:
    from lightgbm import LGBMRegressor
except ImportError:
    LGBMRegressor = None

SEED = 42
ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = ROOT / "data" / "battery_cycle_level_dataset_CLEAN_FINAL.csv"
RESULTS = ROOT / "results" / "final_battery"
MODEL_DIR = ROOT / "backend" / "ml_models" / "artifacts"
PLOT_DIR = RESULTS / "figures"
for directory in (RESULTS, MODEL_DIR, PLOT_DIR):
    directory.mkdir(parents=True, exist_ok=True)

TARGET = "soh"
GROUP = "battery_id"
FORBIDDEN_FEATURES = {"rul", TARGET, "capacity"}


def json_dump(path, value):
    path.write_text(json.dumps(value, indent=2, default=json_default), encoding="utf-8")


def json_default(value):
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.ndarray,)):
        return value.tolist()
    if isinstance(value, (Path,)):
        return str(value)
    raise TypeError(f"Cannot serialize {type(value).__name__}")


def regression_metrics(actual, predicted):
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    return {
        "mae_fraction": float(mean_absolute_error(actual, predicted)),
        "mae_percentage_points": float(mean_absolute_error(actual, predicted) * 100.0),
        "rmse_fraction": float(math.sqrt(mean_squared_error(actual, predicted))),
        "rmse_percentage_points": float(math.sqrt(mean_squared_error(actual, predicted)) * 100.0),
        "r2": float(r2_score(actual, predicted)) if len(actual) > 1 else None,
        "rows": int(len(actual)),
    }


def inspect_dataset(df):
    df = df.sort_values([GROUP, "cycle"], kind="stable").reset_index(drop=True)
    battery_groups = df.groupby(GROUP, sort=True)
    columns = {}
    for name in df.columns:
        series = df[name]
        unique_values = series.drop_duplicates().tolist()
        if len(unique_values) > 50:
            unique_values = unique_values[:10]
        columns[name] = {
            "dtype": str(series.dtype),
            "missing": int(series.isna().sum()),
            "unique_count": int(series.nunique(dropna=False)),
            "unique_values_or_first_10": unique_values,
        }
    per_battery = battery_groups.agg(
        rows=("cycle", "size"),
        first_cycle=("cycle", "min"),
        last_cycle=("cycle", "max"),
        first_capacity=("capacity", "first"),
        first_soh=(TARGET, "first"),
        last_soh=(TARGET, "last"),
        first_rul=("rul", "first"),
        last_rul=("rul", "last"),
    )
    initial_capacity = battery_groups["capacity"].transform("first")
    capacity_soh = df["capacity"] / initial_capacity
    target_formula_abs_error = (df[TARGET] - capacity_soh).abs()
    max_cycle = battery_groups["cycle"].transform("max")
    rul_matches_future = df["rul"].eq(max_cycle - df["cycle"])
    report = {
        "dataset": str(DATA_PATH.relative_to(ROOT)),
        "shape": {"rows": int(len(df)), "columns": int(len(df.columns))},
        "columns": columns,
        "battery_ids": sorted(df[GROUP].astype(str).unique().tolist()),
        "cycles_per_battery": per_battery.to_dict(orient="index"),
        "numeric_summary": df.select_dtypes(include=np.number).describe(
            percentiles=[0.05, 0.25, 0.5, 0.75, 0.95]
        ).to_dict(),
        "correlations": df.select_dtypes(include=np.number).corr().to_dict(),
        "duplicate_rows": int(df.duplicated().sum()),
        "duplicate_battery_cycle_keys": int(df.duplicated([GROUP, "cycle"]).sum()),
        "target_definition_checks": {
            "soh_scale_min": float(df[TARGET].min()),
            "soh_scale_max": float(df[TARGET].max()),
            "capacity_over_first_capacity_mae_fraction": float(target_formula_abs_error.mean()),
            "capacity_over_first_capacity_max_abs_error_fraction": float(target_formula_abs_error.max()),
            "capacity_over_first_capacity_exact_all_rows": bool(target_formula_abs_error.max() < 1e-12),
            "rul_equals_max_cycle_minus_cycle_rows": int(rul_matches_future.sum()),
            "rul_equals_max_cycle_minus_cycle_fraction": float(rul_matches_future.mean()),
            "interpretation": "SOH is capacity divided by the battery's first recorded capacity. RUL equals final observed cycle minus current cycle and leaks future information.",
        },
        "unavailable_raw_signals": ["current", "charge_duration", "discharge_duration", "energy", "power"],
    }
    json_dump(RESULTS / "dataset_audit.json", report)
    return df, report


def model_candidates():
    candidates = {
        "linear": LinearRegression(),
        "ridge_1": Ridge(alpha=1.0),
        "ridge_10": Ridge(alpha=10.0),
        "elastic_net": ElasticNet(alpha=0.001, l1_ratio=0.2, max_iter=10000, random_state=SEED),
        "random_forest": RandomForestRegressor(
            n_estimators=250, min_samples_leaf=2, max_features=0.9, n_jobs=-1, random_state=SEED
        ),
        "extra_trees": ExtraTreesRegressor(
            n_estimators=300, min_samples_leaf=1, max_features=0.9, n_jobs=-1, random_state=SEED
        ),
        "gradient_boosting": GradientBoostingRegressor(
            n_estimators=200, learning_rate=0.03, max_depth=2, loss="huber", random_state=SEED
        ),
        "hist_gradient_boosting": HistGradientBoostingRegressor(
            max_iter=200, learning_rate=0.05, l2_regularization=1.0, random_state=SEED
        ),
        "mlp": make_pipeline(
            StandardScaler(),
            MLPRegressor(hidden_layer_sizes=(64, 32), alpha=0.01, max_iter=600,
                         early_stopping=True, random_state=SEED),
        ),
    }
    xgb_configs = [
        {"n_estimators": 250, "max_depth": 2, "learning_rate": 0.03, "subsample": 0.8, "colsample_bytree": 0.8, "min_child_weight": 3, "gamma": 0, "reg_alpha": 0, "reg_lambda": 1},
        {"n_estimators": 500, "max_depth": 3, "learning_rate": 0.03, "subsample": 0.8, "colsample_bytree": 0.9, "min_child_weight": 3, "gamma": 0, "reg_alpha": 0.1, "reg_lambda": 5},
        {"n_estimators": 700, "max_depth": 4, "learning_rate": 0.02, "subsample": 0.9, "colsample_bytree": 0.8, "min_child_weight": 5, "gamma": 0.1, "reg_alpha": 0.5, "reg_lambda": 10},
        {"n_estimators": 400, "max_depth": 5, "learning_rate": 0.05, "subsample": 0.7, "colsample_bytree": 0.7, "min_child_weight": 1, "gamma": 0, "reg_alpha": 0, "reg_lambda": 1},
        {"n_estimators": 800, "max_depth": 2, "learning_rate": 0.01, "subsample": 1.0, "colsample_bytree": 1.0, "min_child_weight": 10, "gamma": 0.2, "reg_alpha": 1, "reg_lambda": 20},
        {"n_estimators": 300, "max_depth": 6, "learning_rate": 0.03, "subsample": 0.85, "colsample_bytree": 0.85, "min_child_weight": 2, "gamma": 0.05, "reg_alpha": 0.05, "reg_lambda": 3},
    ]
    for index, config in enumerate(xgb_configs, start=1):
        candidates[f"xgboost_{index}"] = XGBRegressor(
            objective="reg:squarederror", n_jobs=1, random_state=SEED, **config
        )
    if LGBMRegressor is not None:
        for index, config in enumerate([
            {"n_estimators": 200, "num_leaves": 7, "learning_rate": 0.03, "min_child_samples": 10, "reg_lambda": 1},
            {"n_estimators": 400, "num_leaves": 15, "learning_rate": 0.02, "min_child_samples": 15, "reg_lambda": 5},
            {"n_estimators": 300, "num_leaves": 31, "learning_rate": 0.03, "min_child_samples": 20, "reg_lambda": 10},
        ], start=1):
            candidates[f"lightgbm_{index}"] = LGBMRegressor(
                verbosity=-1, n_jobs=1, random_state=SEED, **config
            )
    return candidates


def safe_pipeline(estimator):
    if hasattr(estimator, "steps"):
        return make_pipeline(
            SimpleImputer(strategy="median"),
            *[clone(step) for _, step in estimator.steps],
        )
    return make_pipeline(SimpleImputer(strategy="median"), clone(estimator))


def split_unseen_batteries(df):
    groups = df[GROUP].to_numpy()
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=SEED)
    train_val_idx, test_idx = next(splitter.split(df, groups=groups))
    train_val = df.iloc[train_val_idx].copy()
    test = df.iloc[test_idx].copy()
    splitter_val = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=SEED)
    train_idx, val_idx = next(splitter_val.split(train_val, groups=train_val[GROUP]))
    return train_val.iloc[train_idx].copy(), train_val.iloc[val_idx].copy(), test


def temporal_splits(df):
    train_parts, val_parts, test_parts = [], [], []
    eligible = []
    excluded = {}
    for battery_id, group in df.groupby(GROUP, sort=True):
        group = group.sort_values("cycle", kind="stable")
        count = len(group)
        if count < 5:
            excluded[str(battery_id)] = {"cycles": int(count), "reason": "fewer than five observations for train/validation/test chronology"}
            continue
        train_end = max(2, int(np.floor(count * 0.60)))
        val_end = max(train_end + 1, int(np.floor(count * 0.80)))
        val_end = min(val_end, count - 1)
        train_parts.append(group.iloc[:train_end])
        val_parts.append(group.iloc[train_end:val_end])
        test_parts.append(group.iloc[val_end:])
        eligible.append(str(battery_id))
    return pd.concat(train_parts), pd.concat(val_parts), pd.concat(test_parts), eligible, excluded


def transform_target(frame, formulation):
    if formulation == "direct_soh":
        return frame[TARGET].to_numpy(dtype=float)
    if formulation in {"capacity_to_soh", "remaining_capacity_to_soh"}:
        return frame["capacity"].to_numpy(dtype=float)
    if formulation == "delta_soh":
        return (frame[TARGET] - frame[f"{TARGET}_lag_1"]).to_numpy(dtype=float)
    if formulation == "degradation":
        return (frame[f"{TARGET}_lag_1"] - frame[TARGET]).to_numpy(dtype=float)
    if formulation == "normalized_degradation":
        return (1.0 - frame[TARGET]).to_numpy(dtype=float)
    raise ValueError(formulation)


def invert_prediction(frame, values, formulation):
    values = np.asarray(values, dtype=float)
    if formulation == "direct_soh":
        return values
    if formulation in {"capacity_to_soh", "remaining_capacity_to_soh"}:
        return values / frame["initial_capacity"].to_numpy(dtype=float)
    if formulation == "delta_soh":
        return frame[f"{TARGET}_lag_1"].to_numpy(dtype=float) + values
    if formulation == "degradation":
        return frame[f"{TARGET}_lag_1"].to_numpy(dtype=float) - values
    if formulation == "normalized_degradation":
        return 1.0 - values
    raise ValueError(formulation)


def tune_candidates(train, val, feature_columns, formulations):
    output = []
    for formulation in formulations:
        valid_train = train.dropna(subset=[TARGET] + ([f"{TARGET}_lag_1"] if formulation in {"delta_soh", "degradation"} else [])).copy()
        valid_val = val.dropna(subset=[TARGET] + ([f"{TARGET}_lag_1"] if formulation in {"delta_soh", "degradation"} else [])).copy()
        y_train = transform_target(valid_train, formulation)
        y_val = valid_val[TARGET].to_numpy(dtype=float)
        for model_name, estimator in model_candidates().items():
            pipeline = safe_pipeline(estimator)
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    pipeline.fit(valid_train[feature_columns], y_train)
                    raw_prediction = pipeline.predict(valid_val[feature_columns])
                prediction = invert_prediction(valid_val, raw_prediction, formulation)
                record = {
                    "formulation": formulation,
                    "model": model_name,
                    **regression_metrics(y_val, prediction),
                }
                output.append((record, pipeline))
            except Exception as exc:
                output.append(({
                    "formulation": formulation,
                    "model": model_name,
                    "error": f"{type(exc).__name__}: {exc}",
                }, None))
    return output


def battery_error_table(predictions):
    records = []
    for battery_id, group in predictions.groupby(GROUP, sort=True):
        actual = group["actual_soh"].to_numpy()
        predicted = group["predicted_soh"].to_numpy()
        records.append({
            GROUP: str(battery_id),
            "MAE_percentage_points": float(mean_absolute_error(actual, predicted) * 100.0),
            "RMSE_percentage_points": float(math.sqrt(mean_squared_error(actual, predicted)) * 100.0),
            "R2": float(r2_score(actual, predicted)) if len(group) > 1 else None,
            "number_of_cycles": int(len(group)),
        })
    return pd.DataFrame(records)


def plot_battery_predictions(predictions):
    battery_ids = sorted(predictions[GROUP].astype(str).unique())
    columns = 4
    panel_width, panel_height = 260, 150
    rows = int(math.ceil(len(battery_ids) / columns))
    width, height = columns * panel_width, rows * panel_height
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#fff"/>',
        '<style>text{font-family:Arial,sans-serif;fill:#202a2d}.axis{stroke:#b9c3c5;stroke-width:1}.actual{fill:none;stroke:#087f5b;stroke-width:2}.predicted{fill:none;stroke:#e67700;stroke-width:2;stroke-dasharray:4 3}</style>',
    ]
    for index, battery_id in enumerate(battery_ids):
        group = predictions[predictions[GROUP].astype(str) == battery_id].sort_values("cycle")
        x0 = (index % columns) * panel_width + 38
        y0 = (index // columns) * panel_height + 28
        plot_width, plot_height = panel_width - 52, panel_height - 56
        cycle_min, cycle_max = float(group["cycle"].min()), float(group["cycle"].max())
        cycle_span = max(1.0, cycle_max - cycle_min)

        def points(column):
            coordinates = []
            for cycle, soh in zip(group["cycle"], group[column]):
                x = x0 + ((float(cycle) - cycle_min) / cycle_span) * plot_width
                y = y0 + plot_height - ((float(soh) * 100.0 - 55.0) / 50.0) * plot_height
                coordinates.append(f"{x:.1f},{y:.1f}")
            return " ".join(coordinates)

        parts.extend([
            f'<text x="{x0}" y="{y0 - 9}" font-size="13" font-weight="700">{html.escape(battery_id)}</text>',
            f'<line class="axis" x1="{x0}" y1="{y0 + plot_height}" x2="{x0 + plot_width}" y2="{y0 + plot_height}"/>',
            f'<line class="axis" x1="{x0}" y1="{y0}" x2="{x0}" y2="{y0 + plot_height}"/>',
            f'<polyline class="actual" points="{points("actual_soh")}"/>',
            f'<polyline class="predicted" points="{points("predicted_soh")}"/>',
            f'<text x="{x0}" y="{y0 + plot_height + 15}" font-size="10">Cycle {int(cycle_min)}-{int(cycle_max)}</text>',
        ])
    parts.extend([
        f'<text x="{width - 220}" y="18" font-size="11" fill="#087f5b">Actual</text>',
        f'<text x="{width - 160}" y="18" font-size="11" fill="#e67700">Predicted</text>',
        '</svg>',
    ])
    (PLOT_DIR / "future_forecast_by_battery.svg").write_text("\n".join(parts), encoding="utf-8")


def evaluate_unseen_groupkfold(frame, feature_columns):
    folds = []
    splitter = GroupKFold(n_splits=5)
    for fold_number, (train_indices, test_indices) in enumerate(
        splitter.split(frame[feature_columns], frame[TARGET], groups=frame[GROUP]),
        start=1,
    ):
        estimator = make_pipeline(
            SimpleImputer(strategy="median"),
            HistGradientBoostingRegressor(
                max_iter=200, learning_rate=0.05, l2_regularization=1.0, random_state=SEED
            ),
        )
        train_fold = frame.iloc[train_indices]
        test_fold = frame.iloc[test_indices]
        estimator.fit(train_fold[feature_columns], train_fold[TARGET])
        prediction = estimator.predict(test_fold[feature_columns])
        folds.append({
            "fold": fold_number,
            "metrics": regression_metrics(test_fold[TARGET], prediction),
            "test_batteries": sorted(test_fold[GROUP].astype(str).unique().tolist()),
        })
    fold_maes = [fold["metrics"]["mae_percentage_points"] for fold in folds]
    return {
        "protocol": "5-fold GroupKFold; each battery is held out once; cycle 1 is excluded; HistGradientBoosting direct SOH regression.",
        "folds": folds,
        "mean_mae_percentage_points": float(np.mean(fold_maes)),
        "std_mae_percentage_points": float(np.std(fold_maes, ddof=1)),
    }


def main():
    df, audit = inspect_dataset(pd.read_csv(DATA_PATH))
    initial_capacity = df.groupby(GROUP)["capacity"].transform("first")
    df["initial_capacity"] = initial_capacity

    # Unseen batteries: hold out whole groups. Current capacity and future-derived RUL are excluded.
    unseen_features = ["cycle", "voltage", "temperature", "initial_capacity"]
    unseen_frame = df[df["cycle"] > 1].copy()
    unseen_groupkfold = evaluate_unseen_groupkfold(unseen_frame, unseen_features)
    unseen_train, unseen_val, unseen_test = split_unseen_batteries(unseen_frame)
    unseen_results = tune_candidates(unseen_train, unseen_val, unseen_features, ["direct_soh"])
    unseen_successes = [(row, model) for row, model in unseen_results if model is not None]
    unseen_best_row, unseen_best_model = min(unseen_successes, key=lambda item: item[0]["mae_fraction"])
    unseen_fit_rows = pd.concat([unseen_train, unseen_val])
    unseen_best_model.fit(unseen_fit_rows[unseen_features], unseen_fit_rows[TARGET])
    unseen_pred = unseen_best_model.predict(unseen_test[unseen_features])
    unseen_metrics = regression_metrics(unseen_test[TARGET], unseen_pred)
    unseen_predictions = unseen_test[[GROUP, "cycle", TARGET]].rename(columns={TARGET: "actual_soh"}).copy()
    unseen_predictions["predicted_soh"] = unseen_pred
    unseen_predictions.to_csv(RESULTS / "unseen_battery_predictions.csv", index=False)
    unseen_val_leaderboard = pd.DataFrame([row for row, _ in unseen_results if "mae_fraction" in row])
    unseen_val_leaderboard.sort_values("mae_fraction").to_csv(RESULTS / "unseen_battery_validation_leaderboard.csv", index=False)

    # Explicit calibration baseline: valid only after observing cycle 1 and current-cycle capacity.
    formula_test = unseen_test.copy()
    formula_prediction = formula_test["capacity"] / formula_test["initial_capacity"]
    calibrated_formula_metrics = regression_metrics(formula_test[TARGET], formula_prediction)

    # Forecast known batteries: all predictors are lagged; the chronological test is the final cycles.
    temporal_df = build_history_features(df)
    temporal_df["prior_cycle"] = temporal_df.groupby(GROUP, sort=False)["cycle"].shift(1)
    forecast_train, forecast_val, forecast_test, eligible_batteries, excluded_batteries = temporal_splits(temporal_df)
    forecast_features = [
        column for column in temporal_df.columns
        if column not in {GROUP, TARGET, "rul", "capacity", "voltage", "temperature", "prior_cycle"}
        and not column.endswith("_lag_0")
    ]
    # Make the no-future-information boundary explicit.
    if any(column in forecast_features for column in ("rul", "capacity", "voltage", "temperature", TARGET)):
        raise RuntimeError("Contemporaneous or future-derived feature entered forecasting model")

    formulations = ["direct_soh", "capacity_to_soh", "remaining_capacity_to_soh", "delta_soh", "degradation", "normalized_degradation"]
    # Broad next-observation results remain a separate diagnostic when cycle labels have gaps.
    forecast_results = tune_candidates(forecast_train, forecast_val, forecast_features, formulations)
    forecast_leaderboard = pd.DataFrame([row for row, _ in forecast_results if "mae_fraction" in row])
    forecast_leaderboard.sort_values("mae_fraction").to_csv(RESULTS / "next_record_validation_leaderboard.csv", index=False)

    exact_train = forecast_train[(forecast_train["cycle"] - forecast_train["prior_cycle"]) == 1].copy()
    exact_val = forecast_val[(forecast_val["cycle"] - forecast_val["prior_cycle"]) == 1].copy()
    exact_test = forecast_test[(forecast_test["cycle"] - forecast_test["prior_cycle"]) == 1].copy()
    exact_results = tune_candidates(exact_train, exact_val, forecast_features, formulations)
    exact_successes = [(row, model) for row, model in exact_results if model is not None and "mae_fraction" in row]
    best_val_row, best_forecast_model = min(exact_successes, key=lambda item: item[0]["mae_fraction"])
    best_formulation = best_val_row["formulation"]
    fit_frame = pd.concat([exact_train, exact_val]).dropna(
        subset=[TARGET] + ([f"{TARGET}_lag_1"] if best_formulation in {"delta_soh", "degradation"} else [])
    )
    best_forecast_model.fit(fit_frame[forecast_features], transform_target(fit_frame, best_formulation))
    forecast_raw_prediction = best_forecast_model.predict(forecast_test[forecast_features])
    forecast_prediction = invert_prediction(forecast_test, forecast_raw_prediction, best_formulation)
    forecast_metrics = regression_metrics(forecast_test[TARGET], forecast_prediction)
    exact_raw_prediction = best_forecast_model.predict(exact_test[forecast_features])
    exact_prediction = invert_prediction(exact_test, exact_raw_prediction, best_formulation)
    exact_cycle_metrics = regression_metrics(exact_test[TARGET], exact_prediction)

    forecast_predictions = forecast_test[[GROUP, "cycle", "prior_cycle", TARGET, "initial_capacity"]].rename(columns={TARGET: "actual_soh"}).copy()
    forecast_predictions["horizon_cycles"] = forecast_predictions["cycle"] - forecast_predictions["prior_cycle"]
    forecast_predictions["predicted_soh"] = forecast_prediction
    forecast_predictions["absolute_error_percentage_points"] = (forecast_predictions["actual_soh"] - forecast_predictions["predicted_soh"]).abs() * 100.0
    forecast_predictions.to_csv(RESULTS / "future_cycle_predictions.csv", index=False)
    per_battery = battery_error_table(forecast_predictions)
    per_battery.to_csv(RESULTS / "future_cycle_per_battery_errors.csv", index=False)
    exact_predictions = forecast_predictions[forecast_predictions["horizon_cycles"] == 1].copy()
    exact_predictions["predicted_soh"] = exact_prediction
    exact_predictions["absolute_error_percentage_points"] = (exact_predictions["actual_soh"] - exact_predictions["predicted_soh"]).abs() * 100.0
    exact_predictions.to_csv(RESULTS / "exact_one_cycle_predictions.csv", index=False)
    battery_error_table(exact_predictions).to_csv(RESULTS / "exact_one_cycle_per_battery_errors.csv", index=False)
    plot_battery_predictions(exact_predictions)

    # Exact capacity-ratio calculation is recorded separately, never described as ML.
    forecast_formula = forecast_test["capacity"] / forecast_test["initial_capacity"]
    forecast_formula_metrics = regression_metrics(forecast_test[TARGET], forecast_formula)
    persistence_val_metrics = regression_metrics(
        forecast_val[TARGET], forecast_val[f"{TARGET}_lag_1"]
    )
    persistence_test_metrics = regression_metrics(
        forecast_test[TARGET], forecast_test[f"{TARGET}_lag_1"]
    )
    exact_cycle_persistence_metrics = regression_metrics(
        exact_test[TARGET], exact_test[f"{TARGET}_lag_1"],
    )
    exact_cycle_persistence_val_metrics = regression_metrics(
        exact_val[TARGET], exact_val[f"{TARGET}_lag_1"],
    )

    exact_leaderboard = pd.DataFrame([row for row, _ in exact_results if "mae_fraction" in row])
    exact_leaderboard.sort_values("mae_fraction").to_csv(RESULTS / "future_cycle_validation_leaderboard.csv", index=False)

    selected_estimator = (
        best_forecast_model.steps[-1][1]
        if hasattr(best_forecast_model, "steps")
        else best_forecast_model
    )
    selected_params = selected_estimator.get_params(deep=False)
    recorded_param_names = (
        "objective", "n_estimators", "max_depth", "learning_rate", "subsample",
        "colsample_bytree", "min_child_weight", "gamma", "reg_alpha", "reg_lambda",
        "random_state", "n_jobs", "loss", "max_features", "min_samples_leaf",
        "num_leaves", "min_child_samples", "alpha",
        "l1_ratio", "hidden_layer_sizes", "max_iter",
    )
    recorded_hyperparameters = {
        name: selected_params[name]
        for name in recorded_param_names
        if name in selected_params and selected_params[name] is not None
    }
    artifact = {
        "estimator": best_forecast_model,
        "model_name": best_val_row["model"],
        "formulation": best_formulation,
        "feature_columns": forecast_features,
        "target_column": TARGET,
        "target_definition": "SOH = current_capacity / first_observed_capacity for the battery",
        "prediction_mode": "exactly one physical cycle ahead for a known battery; predictors use history through cycle t-1",
        "seed": SEED,
        "training_batteries": sorted(forecast_train[GROUP].astype(str).unique().tolist()),
        "validation_batteries": sorted(forecast_val[GROUP].astype(str).unique().tolist()),
        "test_batteries": sorted(forecast_test[GROUP].astype(str).unique().tolist()),
        "hyperparameters": recorded_hyperparameters,
        "validation_metrics": best_val_row,
        "test_metrics": exact_cycle_metrics,
        "next_record_test_metrics": forecast_metrics,
        "scaler": None,
        "imputer": "SimpleImputer(strategy='median')",
        "sequence_length": None,
        "train_rows": int(len(exact_train)),
        "validation_rows": int(len(exact_val)),
        "test_rows": int(len(exact_test)),
        "excluded_short_batteries": excluded_batteries,
    }
    joblib.dump(artifact, MODEL_DIR / "final_battery_model.joblib")
    model_metadata = {key: value for key, value in artifact.items() if key != "estimator"}
    json_dump(MODEL_DIR / "final_battery_model_config.json", model_metadata)

    metrics_payload = {
        "dataset_audit": audit["target_definition_checks"],
        "unseen_battery": {
            "protocol": "GroupShuffleSplit by battery ID (train/validation/test); test rows exclude cycle 1; features cycle, voltage, temperature, and first-cycle capacity; current capacity and RUL excluded.",
            "validation_winner": unseen_best_row,
            "test": unseen_metrics,
            "test_batteries": sorted(unseen_test[GROUP].astype(str).unique().tolist()),
            "test_rows": int(len(unseen_test)),
            "group_kfold_5_fold": unseen_groupkfold,
            "capacity_ratio_calculation_baseline": calibrated_formula_metrics,
            "capacity_ratio_baseline_label": "direct calculation using current capacity; not an ML forecast",
        },
        "known_battery_future_cycles": {
            "protocol": "Within each eligible battery, earliest 60% of observations train, next 20% validate, final 20% test. Final model selection/training/testing use exact consecutive-cycle transitions only. The broader next-record score is separately reported and can include gaps. Test features use only prior cycles.",
            "selected_model": best_val_row["model"],
            "selected_formulation": best_formulation,
            "validation": best_val_row,
            "test": forecast_metrics,
            "exact_one_physical_cycle_test": exact_cycle_metrics,
            "exact_one_physical_cycle_test_batteries": sorted(
                exact_test[GROUP].astype(str).unique().tolist()
            ),
            "exact_one_physical_cycle_coverage": {
                "rows": exact_cycle_metrics["rows"],
                "all_test_rows": forecast_metrics["rows"],
                "fraction": float(exact_cycle_metrics["rows"] / forecast_metrics["rows"]),
            },
            "test_batteries": sorted(forecast_test[GROUP].astype(str).unique().tolist()),
            "eligible_batteries": eligible_batteries,
            "excluded_batteries": excluded_batteries,
            "train_rows": int(len(forecast_train)),
            "validation_rows": int(len(forecast_val)),
            "test_rows": int(len(forecast_test)),
            "exact_one_cycle_train_rows": int(len(exact_train)),
            "exact_one_cycle_validation_rows": int(len(exact_val)),
            "exact_one_cycle_test_rows": int(len(exact_test)),
            "capacity_ratio_calculation_baseline": forecast_formula_metrics,
            "capacity_ratio_baseline_label": "same-cycle capacity calculation; diagnostic only, not a future forecast",
            "persistence_baseline_validation": persistence_val_metrics,
            "persistence_baseline_test": persistence_test_metrics,
            "exact_one_physical_cycle_persistence_baseline_validation": exact_cycle_persistence_val_metrics,
            "exact_one_physical_cycle_persistence_baseline_test": exact_cycle_persistence_metrics,
            "exact_one_cycle_mae_improvement_over_persistence_percentage": float(
                (exact_cycle_persistence_metrics["mae_percentage_points"] - exact_cycle_metrics["mae_percentage_points"])
                / exact_cycle_persistence_metrics["mae_percentage_points"] * 100.0
            ),
            "mae_improvement_over_persistence_percentage": float(
                (persistence_test_metrics["mae_percentage_points"] - forecast_metrics["mae_percentage_points"])
                / persistence_test_metrics["mae_percentage_points"] * 100.0
            ),
        },
        "model_availability": {"catboost": "not installed; not run", "deep_temporal_architectures": "not yet run; tree/linear/MLP validation benchmark selects next step"},
        "claim_under_1_5_percentage_points": bool(exact_cycle_metrics["mae_percentage_points"] < 1.5),
        "final_model_path": str((MODEL_DIR / "final_battery_model.joblib").relative_to(ROOT)),
        "metrics_path": str((RESULTS / "final_battery_metrics.json").relative_to(ROOT)),
    }
    json_dump(MODEL_DIR / "final_battery_metrics.json", metrics_payload)
    json_dump(ROOT / "backend" / "ml_models" / "final_battery_metrics.json", metrics_payload)
    json_dump(RESULTS / "final_battery_metrics.json", metrics_payload)

    report = f"""# Final Battery SOH Model Evaluation

## Dataset and target
- Source: `{DATA_PATH.relative_to(ROOT)}`
- Records: {len(df)}; batteries: {df[GROUP].nunique()}
- SOH is defined in this dataset as `capacity / first recorded capacity for that battery` (verified maximum absolute residual: {audit['target_definition_checks']['capacity_over_first_capacity_max_abs_error_fraction']:.3g} fractional SOH).
- `rul` equals final observed cycle minus current cycle for {audit['target_definition_checks']['rul_equals_max_cycle_minus_cycle_rows']} of {len(df)} rows. It is future-derived and excluded from every learned model.
- No current, charge/discharge duration, energy, or power columns are available.

## Evaluation tracks
### Held-out batteries
Protocol: GroupShuffleSplit by battery ID. The model uses cycle, voltage, temperature, and the first-cycle capacity calibration; current capacity and RUL are excluded. Test rows exclude cycle 1.

- Validation-selected model: {unseen_best_row['model']}
- Test MAE: {unseen_metrics['mae_percentage_points']:.4f} percentage points
- Test RMSE: {unseen_metrics['rmse_percentage_points']:.4f} percentage points
- Test R²: {unseen_metrics['r2']}
- Test batteries: {', '.join(sorted(unseen_test[GROUP].astype(str).unique().tolist()))}
- 5-fold GroupKFold mean MAE: {unseen_groupkfold['mean_mae_percentage_points']:.4f} +/- {unseen_groupkfold['std_mae_percentage_points']:.4f} percentage points

### Known-battery future-cycle rolling forecast
Protocol: For batteries with at least five observations, earliest 60% train, next 20% validate, final 20% test. The full next-record test includes occasional gaps in cycle numbers. The primary API claim is evaluated on exactly one-cycle-ahead rows only. Inputs use measurements through the preceding observation.

- Validation-selected model: {best_val_row['model']}
- Target formulation: {best_formulation}
- Exact-cycle train/validation/test rows: {len(exact_train)}/{len(exact_val)}/{len(exact_test)}
- Test MAE: {forecast_metrics['mae_percentage_points']:.4f} percentage points
- Test RMSE: {forecast_metrics['rmse_percentage_points']:.4f} percentage points
- Test R²: {forecast_metrics['r2']}
- Test rows: {forecast_metrics['rows']}; test batteries: {len(forecast_test[GROUP].unique())}
- Test battery IDs: {', '.join(sorted(forecast_test[GROUP].astype(str).unique().tolist()))}
- Exact one-physical-cycle-ahead MAE: {exact_cycle_metrics['mae_percentage_points']:.4f} percentage points ({exact_cycle_metrics['rows']} of {forecast_metrics['rows']} rows; {exact_cycle_metrics['rows'] / forecast_metrics['rows']:.1%} coverage)
- Exact one-cycle persistence baseline MAE: {exact_cycle_persistence_metrics['mae_percentage_points']:.4f} percentage points

## Frozen selected pipeline
- Estimator: {selected_estimator.__class__.__name__} with median imputation; no scaler.
- Formulation: `{best_formulation}`; predicted degradation is subtracted from the prior observed SOH when selected.
- Target definition: `soh = capacity / first recorded capacity`.
- Features ({len(forecast_features)}): {', '.join(forecast_features)}.
- Temporal history: lag/rolling/EWMA features from prior observations through lag 10; no contemporaneous target, capacity, voltage, temperature, or RUL.
- Sequence length: not applicable; estimator receives engineered tabular history features.
- Seed: {SEED}; exact-cycle train/validation/test rows: {len(exact_train)}/{len(exact_val)}/{len(exact_test)}.
- Training/validation/test battery IDs: {', '.join(eligible_batteries)} (chronological partitions within each battery).
- Hyperparameters: `{json.dumps(recorded_hyperparameters, sort_keys=True)}`.
- Exact-one-cycle validation persistence MAE: {exact_cycle_persistence_val_metrics['mae_percentage_points']:.4f} pp.
- Exact-one-cycle test metrics: MAE {exact_cycle_metrics['mae_percentage_points']:.4f} pp, RMSE {exact_cycle_metrics['rmse_percentage_points']:.4f} pp, R² {exact_cycle_metrics['r2']:.4f}; test battery IDs: {', '.join(sorted(exact_test[GROUP].astype(str).unique().tolist()))}.

## Formula baseline, not a learned model
SOH calculated directly as `current capacity / initial capacity` is measured separately. Its score is not represented as ML performance, and for future forecasting its use of same-cycle capacity is explicitly not a forecast.

## Artifacts
- Selected pipeline: `backend/ml_models/artifacts/final_battery_model.joblib`
- Model config: `backend/ml_models/artifacts/final_battery_model_config.json`
- Metrics: `backend/ml_models/final_battery_metrics.json`
- Audit: `results/final_battery/dataset_audit.json`
- Exact-one-cycle validation leaderboard: `results/final_battery/future_cycle_validation_leaderboard.csv`
- Next-record diagnostic leaderboard: `results/final_battery/next_record_validation_leaderboard.csv`
- Test predictions and per-battery errors: `results/final_battery/exact_one_cycle_predictions.csv`, `results/final_battery/exact_one_cycle_per_battery_errors.csv`
- Per-battery plot: `results/final_battery/figures/future_forecast_by_battery.svg`

## Resume claim
"Under 1.5 percentage points" is {'supported for the held-out, exact-one-physical-cycle-ahead subset' if exact_cycle_metrics['mae_percentage_points'] < 1.5 else 'not achieved on the exact-one-physical-cycle-ahead subset'} (measured MAE: {exact_cycle_metrics['mae_percentage_points']:.4f} percentage points over {exact_cycle_metrics['rows']} rows). The broader next-record score over all {forecast_metrics['rows']} test rows is {forecast_metrics['mae_percentage_points']:.4f} points. Neither result transfers to unseen-battery generalization.
"""
    (ROOT / "docs" / "final_battery_model_report.md").write_text(report, encoding="utf-8")

    print(json.dumps({
        "dataset_rows": len(df),
        "battery_count": int(df[GROUP].nunique()),
        "soh_capacity_formula_max_error": audit["target_definition_checks"]["capacity_over_first_capacity_max_abs_error_fraction"],
        "rul_future_leakage_rows": audit["target_definition_checks"]["rul_equals_max_cycle_minus_cycle_rows"],
        "unseen_battery_test": unseen_metrics,
        "unseen_battery_validation_winner": unseen_best_row,
        "future_cycle_test": forecast_metrics,
        "future_cycle_validation_winner": best_val_row,
        "future_cycle_formula_baseline": forecast_formula_metrics,
        "exact_one_cycle_future_forecast": exact_cycle_metrics,
        "exact_one_cycle_coverage": metrics_payload["known_battery_future_cycles"]["exact_one_physical_cycle_coverage"],
        "under_1_5_future_forecast": metrics_payload["claim_under_1_5_percentage_points"],
        "saved_model": str(MODEL_DIR / "final_battery_model.joblib"),
        "metrics": str(RESULTS / "final_battery_metrics.json"),
    }, indent=2, default=json_default))


if __name__ == "__main__":
    main()
