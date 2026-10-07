# AutoCircle AI — Real Battery SOH Evaluation Report

## Data source
- Source file: `data/battery_cycle_level_dataset_CLEAN_FINAL.csv`
- Rows: 1,415
- Unique batteries: 34
- Missing values: none
- Duplicate rows: 0
- Target column: `soh`
- Split method: battery-level split (train/val/test by unique battery ID) to avoid leakage.

## Measured test-set results

| Model | MAE | RMSE | R² |
|---|---:|---:|---:|
| LSTM | 0.0593 | 0.0852 | 0.4236 |
| XGBoost | 0.0803 | 0.1364 | -0.3099 |

Interpretation:
- LSTM MAE = 5.93% SOH on the held-out test split.
- XGBoost MAE = 8.03% SOH on the held-out test split.
- Neither model meets an "under 1.5% MAE" claim on the measured test set.
- The project does contain a real battery SOH prediction pipeline, but the model quality is not yet at the claimed level.

## Claim verification
- "2,800+ NASA battery records": Not verified. The uploaded dataset contains 1,415 rows and 34 batteries; the repository does not provide evidence for a 2,800+ NASA record set.
- "MAE under 1.5%": Not verified on the actual held-out test set. The best measured MAE was 5.93%.
- "Real model evidence": Verified. A real LSTM and a real XGBoost model were trained and evaluated on the uploaded dataset.

## Recommended resume wording
> Trained and evaluated battery SOH prediction models on a real cycle-level dataset using a battery-level split; the held-out test MAE was approximately 5.9% for the best-performing model.
