from functools import lru_cache
from pathlib import Path

from fastapi import APIRouter, HTTPException
import pandas as pd

try:
    from app.schemas.pydantic_models import (
        BatteryPredictRequest, BatteryPredictResponse,
        BatchBatteryPredictRequest, BatchBatteryPredictResponse,
        BatteryForecastRequest, BatteryForecastResponse
    )
    from app.services.battery_service import battery_service, final_battery_model
except ModuleNotFoundError:
    from schemas.pydantic_models import (
        BatteryPredictRequest, BatteryPredictResponse,
        BatchBatteryPredictRequest, BatchBatteryPredictResponse,
        BatteryForecastRequest, BatteryForecastResponse
    )
    from services.battery_service import battery_service, final_battery_model

router = APIRouter(prefix="/api/battery", tags=["Battery SOH"])
DATASET_PATH = Path(__file__).resolve().parents[3] / "data" / "battery_cycle_level_dataset_CLEAN_FINAL.csv"
CONTINUE_EV_MIN_SOH = 80.0
SECOND_LIFE_MIN_SOH = 70.0


@lru_cache(maxsize=1)
def _load_battery_dataset():
    if not DATASET_PATH.exists():
        raise HTTPException(status_code=503, detail="Battery history dataset is unavailable.")
    try:
        return pd.read_csv(DATASET_PATH).sort_values(["battery_id", "cycle"])
    except Exception as exc:
        raise HTTPException(status_code=500, detail="Battery history dataset could not be read.") from exc


@router.get("/batteries")
def list_battery_samples():
    dataset = _load_battery_dataset()
    summary = dataset.groupby("battery_id").agg(
        cycle_count=("cycle", "size"),
        first_cycle=("cycle", "min"),
        last_cycle=("cycle", "max"),
    )
    eligible = summary[(summary["cycle_count"] >= 4) & (summary["first_cycle"] == 1)]
    return {
        "batteries": [
            {
                "battery_id": str(battery_id),
                "cycle_count": int(row["cycle_count"]),
                "last_cycle": int(row["last_cycle"]),
                "observed_cycles": [
                    int(cycle)
                    for cycle in dataset.loc[dataset["battery_id"] == battery_id, "cycle"].tolist()
                ],
            }
            for battery_id, row in eligible.sort_index().iterrows()
        ]
    }


@router.get("/batteries/{battery_id}/history")
def get_battery_history(battery_id: str, through_cycle: int | None = None):
    dataset = _load_battery_dataset()
    battery = dataset[dataset["battery_id"].astype(str) == battery_id].sort_values("cycle")
    if battery.empty:
        raise HTTPException(status_code=404, detail=f"Battery {battery_id} was not found.")
    if through_cycle is not None:
        battery = battery[battery["cycle"] <= through_cycle]
    if len(battery) < 4 or int(battery["cycle"].iloc[0]) != 1:
        raise HTTPException(status_code=422, detail="Selected history must contain at least four rows beginning at cycle 1.")
    history = battery[["cycle", "voltage", "temperature", "capacity"]]
    return {
        "battery_id": battery_id,
        "through_cycle": int(history["cycle"].iloc[-1]),
        "available_cycle_count": int(dataset[dataset["battery_id"].astype(str) == battery_id].shape[0]),
        "history": history.to_dict(orient="records"),
    }

@router.post("/predict", response_model=BatteryPredictResponse)
def predict_soh(req: BatteryPredictRequest):
    try:
        return battery_service.predict_soh(req)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/batch-predict", response_model=BatchBatteryPredictResponse)
def predict_batch(req: BatchBatteryPredictRequest):
    try:
        return battery_service.predict_batch(req)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/forecast", response_model=BatteryForecastResponse)
def forecast_next_cycle(req: BatteryForecastRequest):
    try:
        result = final_battery_model.predict_next(
            battery_id=req.battery_id,
            history=[item.model_dump() for item in req.history],
            next_cycle=req.next_cycle,
        )
        soh_percent = result["soh"]
        if soh_percent >= CONTINUE_EV_MIN_SOH:
            result["routing_decision"] = "Continue EV Use"
            result["recommendation"] = "Predicted SOH is suitable for continued EV use; verify against the next measured cycle."
        elif soh_percent >= SECOND_LIFE_MIN_SOH:
            result["routing_decision"] = "Second-Life Storage"
            result["recommendation"] = "Predicted SOH may suit second-life evaluation; confirm with measured capacity and safety checks."
        else:
            result["routing_decision"] = "Recycle Now"
            result["recommendation"] = "Predicted SOH is low; verify with measured capacity and qualified recycling assessment."
        return result
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

@router.get("/metrics")
def get_model_metrics():
    try:
        metrics = final_battery_model.get_metrics()
        metrics["routing_thresholds"] = {
            "continue_ev_min_soh_percent": CONTINUE_EV_MIN_SOH,
            "second_life_min_soh_percent": SECOND_LIFE_MIN_SOH,
        }
        return metrics
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
