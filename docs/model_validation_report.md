# Model validation report

## Dataset
- Dataset: synthetic battery cycle dataset generated in code
- Records: 2,880
- Batteries: 8
- Features: `cycle_count`, `avg_temp`, `depth_of_discharge`, `fast_charge_frequency`, `internal_resistance`
- Train size: 5 batteries
- Validation size: 1 battery
- Test size: 2 batteries

## Model architecture
- LSTM: 2-layer PyTorch network with 64 hidden units and dropout regularization
- XGBoost: gradient boosted regressor with 400 trees and tuned tree depth / learning rate
- Ensemble: Ridge meta-learner combining LSTM and XGBoost outputs

## Training configuration
- Sequence length: 5
- Loss: MSE
- Optimizer: Adam
- Learning rate: 0.003
- Batch size: 32
- Epochs: 60
- Early stopping: not implemented in this version

## Final metrics on unseen test batteries

| Model | MAE | RMSE | R² |
|-------|-----|------|----|
| LSTM | 1.638% | 1.669% | 0.2741 |
| XGBoost | 0.030% | 0.038% | 0.9996 |
| Ensemble | 0.030% | 0.038% | 0.9996 |

## Status for the 1.5% claim
The repo supports a verified claim that the XGBoost and ensemble model achieve MAE below 1.5% on the current held-out synthetic test split. The claim is therefore VERIFIED for the current model and dataset, but it must be phrased carefully because the dataset is synthetic and not proven to be an imported NASA raw dataset.
