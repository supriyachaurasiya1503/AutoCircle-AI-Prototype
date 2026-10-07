# Dataset report

## Dataset audit
- Dataset source: generated in `backend/ml_models/train_hybrid_soh.py` using scripted battery degradation logic
- Dataset file: `backend/datasets/nasa_battery_dataset.csv`
- Total records: 2,880
- Unique batteries/cells: 8
- Cycles per battery: 360
- Columns: `battery_id`, `cycle_count`, `avg_temp`, `depth_of_discharge`, `fast_charge_frequency`, `internal_resistance`, `voltage`, `current`, `capacity`, `SOH`
- Target: `SOH`
- Missing values: none detected in the generated file
- Duplicates: none observed in the generated dataset
- Invalid records: none in the generated synthetic dataset
- Outliers: within synthetic ranges; no formal anomaly review was performed against a raw NASA benchmark

## Important caveat
The code generates the battery dataset rather than importing a raw NASA dataset file. That means the repo can support a valid experiment on a synthetic battery-cycle dataset, but the wording should not claim a direct raw NASA dataset import without a documented external file and provenance trail.

## Split logic
The code uses a group-level battery split instead of random row splitting:
- Train: B0005, B0006, B0007, B0018, B0033
- Validation: B0034
- Test: B0042, B0043

This is a sensible design for time-series battery degradation data and prevents leakage across cycles of the same battery.

## Observed statistics
- Mean SOH: 96.56
- Min SOH: 93.10
- Max SOH: 100.00
- Data shape: (2880, 10)
