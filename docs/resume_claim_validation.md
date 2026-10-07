# Resume claim validation

This project contains a real working prototype, but several resume claims need to be tightened to match the evidence that is actually present in the repository.

## Summary table

| Resume Claim | Evidence | Actual Metric | Status |
|---|---|---|---|
| Four-module system | Backend routes and frontend navigation | 4 modules implemented | ✅ VERIFIED |
| 2,800+ battery records | Generated synthetic dataset in code | 2,880 rows across 8 cells | 🟡 PARTIALLY VERIFIED |
| LSTM + XGBoost | Saved model files and training script | Model files exist and run | ✅ VERIFIED |
| <1.5% MAE | Held-out test evaluation | 0.030% MAE on synthetic test batteries | ✅ VERIFIED |
| FastAPI | API endpoints and smoke tests | Health and core endpoints pass | ✅ VERIFIED |
| 1M+ data points/day | Local benchmark projection | ~33M assessments/day projected on local CPU | 🔵 PROJECTED |
| YOLOv8 | Disassembly service exists | No real model artifact or evaluation present | ❌ NOT VERIFIED |
| 60% faster disassembly | No physical benchmark present | Not measured | ❌ NOT VERIFIED |
| 92% material recovery | Hard-coded value in service output | Not experimentally validated | ❌ NOT VERIFIED |
| Digital Product Passport | Passport generator works | Structured JSON output generated | ✅ VERIFIED |
| 10K+ assessments/day | Local benchmark projection | ~33M/day projected local benchmark | 🔵 PROJECTED |

## Important interpretation
The battery model claim is the strongest and most defensible result in this repo. The disassembly and recovery claims are not yet supported by a real dataset, trained YOLO weights, or benchmark pipeline. The passport module is functional as a prototype, but the claims around legal compliance and throughput should be framed as values aligned with a prototype or projected benchmark, not as measured production reality.

## Safer resume wording
Use wording like:
- “Built a four-module circular-economy dashboard prototype for battery SOH, disassembly planning, digital passport generation, and sustainability analytics.”
- “Implemented an LSTM + XGBoost ensemble for battery SOH prediction and validated it on a held-out battery-group split with 0.03% MAE on the current synthetic test set.”
- “Created a FastAPI-based prototype backend with benchmarked local inference throughput for battery assessment processing.”
- “Developed an EU Battery Regulation-aligned digital product passport prototype rather than a legally certified compliance product.”
