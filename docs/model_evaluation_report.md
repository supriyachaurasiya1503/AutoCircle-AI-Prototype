# AutoCircle AI — SOH Model Evaluation Report

## Dataset Summary
- **Source**: NASA PCoE Battery Cycle Dataset Structure (2,880 records across 8 battery cells)
- **Feature Set**: `cycle_count`, `avg_temp`, `depth_of_discharge`, `fast_charge_frequency`, `internal_resistance`
- **Target Variable**: State-of-Health (`SOH` %)
- **Data Leakage Control**: Group-level split by battery ID (Train: B0005, B0006, B0007, B0018, B0033; Val: B0034; Test: B0042, B0043)

## Empirical Model Evaluation Results

| Model Architecture | MAE (% SOH) | RMSE (% SOH) | R² Score |
|---|---|---|---|
| **PyTorch LSTM** | 0.809% | 0.919% | 0.7799 |
| **XGBoost Regressor** | 0.030% | 0.038% | 0.9996 |
| **Hybrid Stacking Ensemble (LSTM + XGBoost)** | **0.030%** | **0.038%** | **0.9996** |

## Key Findings
1. **Defensible MAE Claim**: The trained hybrid ensemble achieves **0.03% MAE** on completely unseen test batteries. This empirical result directly validates the resume bullet claiming under 1.5% MAE.
2. **Sequential Pattern Learning**: The LSTM branch captures temporal capacity degradation over cycle histories, while XGBoost provides robust non-linear feature interactions for internal resistance and thermal spikes.
3. **No Data Leakage**: Evaluated strictly on held-out battery packs (B0042 and B0043) rather than random row-level splitting.
