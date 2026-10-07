# AutoCircle AI

## Project status
This repository is a working prototype with an operational battery SOH pipeline and REST API, but it is not yet a fully validated production system. The verified evidence below is intentionally conservative and based on the code and running model artifacts in this repo.

## Verified facts
- The backend exposes four modules: battery SOH, disassembly, passport, and sustainability.
- The battery pipeline uses a synthesized battery-cycle dataset with 2,880 rows across 8 battery IDs.
- The held-out test split is battery-group based and uses unseen cells.
- The current saved XGBoost model achieves about 0.03% MAE and 0.9996 R² on the held-out test batteries.
- The backend smoke tests pass for the core health and endpoint checks.

## Not yet fully verified or projected
- The claim of real NASA battery records is not fully supported; the dataset is generated in code and is therefore synthetic rather than a raw NASA file import.
- The disassembly and material recovery metrics are not backed by a real YOLO validation artifact or real time-motion benchmark in this repo.
- The passport is functional as a structured JSON generator, but legal EU certification is not implemented.
- The 1M/day and 10K/day throughput claims are only benchmarked as local model throughput projections, not verified production capacity.

## Verified model metrics from the repo
- LSTM MAE: 1.638%
- XGBoost MAE: 0.030%
- Ensemble MAE: 0.030%
- XGBoost R²: 0.9996

## Quick start
### Backend
```bash
cd backend
python -m pytest test_api.py -q
python app/main.py
```

### Frontend
```bash
cd frontend
npm install
npm run build
npm run dev
```

## Evidence files
- [docs/current_project_audit.md](docs/current_project_audit.md)
- [docs/claim_evidence_matrix.md](docs/claim_evidence_matrix.md)
- [docs/four_module_validation.md](docs/four_module_validation.md)
- [docs/dataset_report.md](docs/dataset_report.md)
- [docs/model_validation_report.md](docs/model_validation_report.md)
- [docs/resume_claim_validation.md](docs/resume_claim_validation.md)

## Resume wording guidance
Claims should use evidence-backed wording such as:
- “A prototype battery SOH prediction system using a group-split synthetic battery dataset and an LSTM + XGBoost ensemble.”
- “A FastAPI-based prototype backend and dashboard with four modules and benchmarked local inference throughput.”
- “A passport-generation prototype aligned with EU Battery Regulation concepts rather than a legally certified product passport system.”
