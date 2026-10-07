# Final Battery SOH Model Evaluation

## Dataset and target
- Source: `data\battery_cycle_level_dataset_CLEAN_FINAL.csv`
- Records: 1415; batteries: 34
- SOH is defined in this dataset as `capacity / first recorded capacity for that battery` (verified maximum absolute residual: 4.44e-16 fractional SOH).
- `rul` equals final observed cycle minus current cycle for 1415 of 1415 rows. It is future-derived and excluded from every learned model.
- No current, charge/discharge duration, energy, or power columns are available.

## Evaluation tracks
### Held-out batteries
Protocol: GroupShuffleSplit by battery ID. The model uses cycle, voltage, temperature, and the first-cycle capacity calibration; current capacity and RUL are excluded. Test rows exclude cycle 1.

- Validation-selected model: hist_gradient_boosting
- Test MAE: 4.8501 percentage points
- Test RMSE: 8.4581 percentage points
- Test R²: 0.5443436257990777
- Test batteries: B0005, B0029, B0030, B0032, B0042, B0050
- 5-fold GroupKFold mean MAE: 5.9075 +/- 3.3383 percentage points

### Known-battery future-cycle rolling forecast
Protocol: For batteries with at least five observations, earliest 60% train, next 20% validate, final 20% test. The full next-record test includes occasional gaps in cycle numbers. The primary API claim is evaluated on exactly one-cycle-ahead rows only. Inputs use measurements through the preceding observation.

- Validation-selected model: gradient_boosting
- Target formulation: degradation
- Exact-cycle train/validation/test rows: 790/270/280
- Test MAE: 0.6952 percentage points
- Test RMSE: 1.4587 percentage points
- Test R²: 0.9837910628526646
- Test rows: 288; test batteries: 23
- Test battery IDs: B0005, B0006, B0007, B0018, B0025, B0026, B0027, B0028, B0029, B0030, B0031, B0032, B0039, B0041, B0042, B0043, B0044, B0045, B0046, B0047, B0048, B0049, B0053
- Exact one-physical-cycle-ahead MAE: 0.5716 percentage points (280 of 288 rows; 97.2% coverage)
- Exact one-cycle persistence baseline MAE: 0.7140 percentage points

## Frozen selected pipeline
- Estimator: GradientBoostingRegressor with median imputation; no scaler.
- Formulation: `degradation`; predicted degradation is subtracted from the prior observed SOH when selected.
- Target definition: `soh = capacity / first recorded capacity`.
- Features (55): cycle, initial_capacity, cycle_index, capacity_lag_1, capacity_lag_2, capacity_lag_3, capacity_lag_5, capacity_lag_10, capacity_mean_prev_3, capacity_std_prev_3, capacity_mean_prev_5, capacity_std_prev_5, capacity_mean_prev_10, capacity_std_prev_10, capacity_ewm_prev, soh_lag_1, soh_lag_2, soh_lag_3, soh_lag_5, soh_lag_10, soh_mean_prev_3, soh_std_prev_3, soh_mean_prev_5, soh_std_prev_5, soh_mean_prev_10, soh_std_prev_10, soh_ewm_prev, voltage_lag_1, voltage_lag_2, voltage_lag_3, voltage_lag_5, voltage_lag_10, voltage_mean_prev_3, voltage_std_prev_3, voltage_mean_prev_5, voltage_std_prev_5, voltage_mean_prev_10, voltage_std_prev_10, voltage_ewm_prev, temperature_lag_1, temperature_lag_2, temperature_lag_3, temperature_lag_5, temperature_lag_10, temperature_mean_prev_3, temperature_std_prev_3, temperature_mean_prev_5, temperature_std_prev_5, temperature_mean_prev_10, temperature_std_prev_10, temperature_ewm_prev, soh_delta_lag_1, capacity_delta_lag_1, capacity_ratio_lag_1, soh_slope_prev_5.
- Temporal history: lag/rolling/EWMA features from prior observations through lag 10; no contemporaneous target, capacity, voltage, temperature, or RUL.
- Sequence length: not applicable; estimator receives engineered tabular history features.
- Seed: 42; exact-cycle train/validation/test rows: 790/270/280.
- Training/validation/test battery IDs: B0005, B0006, B0007, B0018, B0025, B0026, B0027, B0028, B0029, B0030, B0031, B0032, B0039, B0041, B0042, B0043, B0044, B0045, B0046, B0047, B0048, B0049, B0053 (chronological partitions within each battery).
- Hyperparameters: `{"alpha": 0.9, "learning_rate": 0.03, "loss": "huber", "max_depth": 2, "min_samples_leaf": 1, "n_estimators": 200, "random_state": 42, "subsample": 1.0}`.
- Exact-one-cycle validation persistence MAE: 0.7100 pp.
- Exact-one-cycle test metrics: MAE 0.5716 pp, RMSE 1.1323 pp, R² 0.9897; test battery IDs: B0005, B0006, B0007, B0018, B0026, B0027, B0028, B0029, B0030, B0031, B0032, B0039, B0041, B0042, B0043, B0044, B0045, B0046, B0047, B0048, B0049, B0053.

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
"Under 1.5 percentage points" is supported for the held-out, exact-one-physical-cycle-ahead subset (measured MAE: 0.5716 percentage points over 280 rows). The broader next-record score over all 288 test rows is 0.6952 points. Neither result transfers to unseen-battery generalization.
