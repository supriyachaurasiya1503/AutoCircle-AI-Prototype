from typing import List, Optional
from pydantic import BaseModel, Field

# --- Battery SOH Models ---
class BatteryPredictRequest(BaseModel):
    cycle_count: int = Field(..., example=800, ge=0, le=3000)
    avg_temp: float = Field(..., example=28.0, ge=10.0, le=60.0)
    depth_of_discharge: float = Field(..., example=70.0, ge=10.0, le=100.0)
    fast_charge_frequency: float = Field(..., example=20.0, ge=0.0, le=100.0)
    internal_resistance: float = Field(..., example=110.0, ge=50.0, le=300.0)

class BatteryPredictResponse(BaseModel):
    soh: float = Field(..., description="Predicted SOH percentage (0-100%)")
    rul_cycles: int = Field(..., description="Estimated Remaining Useful Life in cycles")
    routing_decision: str = Field(..., description="Continue EV | Second-Life Storage | Recycle Now")
    confidence: float = Field(..., description="Confidence score (%)")
    degradation_mode: str = Field(..., description="Primary degradation driver")
    key_insight: str = Field(..., description="ML model technical insight")
    recommendation: str = Field(..., description="Actionable recommendation for battery pack")
    model_used: str = Field(default="PyTorch LSTM + XGBoost Stacking Ensemble")

class BatchBatteryPredictRequest(BaseModel):
    items: List[BatteryPredictRequest]

class BatchBatteryPredictResponse(BaseModel):
    total_processed: int
    processing_time_ms: float
    predictions: List[BatteryPredictResponse]

class BatteryCycleObservation(BaseModel):
    cycle: int = Field(..., ge=1)
    voltage: float
    temperature: float
    capacity: float = Field(..., gt=0)

class BatteryForecastRequest(BaseModel):
    battery_id: str = Field(..., min_length=1)
    history: List[BatteryCycleObservation] = Field(..., min_length=4)
    next_cycle: Optional[int] = Field(default=None, ge=1)

class BatteryForecastResponse(BaseModel):
    battery_id: str
    next_cycle: int
    soh: float = Field(..., description="Predicted SOH percentage")
    model_used: str
    prediction_mode: str
    mae_percentage_points: float
    test_rows: int
    test_coverage: float
    routing_decision: str
    recommendation: str

# --- Disassembly AI Models ---
class ComponentDetection(BaseModel):
    id: str
    name: str
    material: str
    risk_level: str
    color: str
    confidence: float
    bbox: List[int]

class DisassemblyStep(BaseModel):
    step_num: int
    action: str
    component: str
    safety_protocol: str
    estimated_time_sec: int

class DisassemblyResponse(BaseModel):
    components_detected: List[ComponentDetection]
    disassembly_sequence: List[DisassemblyStep]
    estimated_time_mins: float
    material_recovery_rate: float
    safety_alerts: List[str]

# --- Passport Models ---
class PassportGenerateRequest(BaseModel):
    component_type: str = Field(..., example="battery_pack")
    vin: str = Field(..., example="1HGBH41JXMN109186")
    soh: float = Field(..., example=73.4)

class PassportMaterialItem(BaseModel):
    name: str
    percentage: float
    recyclability: str
    svhc: bool

class PassportResponse(BaseModel):
    passport_id: str
    component: str
    vin: str
    manufacture_date: str
    materials: List[PassportMaterialItem]
    soh_percent: float
    routing: str
    second_life_suitability: str
    disposal_instructions: str
    carbon_footprint_kg: float
    blockchain_tx: str
    compliance_badges: List[str]
    qr_code_data: str

# --- Sustainability Models ---
class CarbonAnalyzeRequest(BaseModel):
    vehicles_processed: int = Field(..., example=2000, ge=1)
    second_life_rate: float = Field(..., example=0.6, ge=0.0, le=1.0)
    aluminium_recovery_purity: float = Field(..., example=0.94, ge=0.0, le=1.0)
    renewable_energy_ratio: float = Field(..., example=0.35, ge=0.0, le=1.0)

class CarbonLever(BaseModel):
    name: str
    co2e_saved_tonnes: float
    percentage_contribution: float

class CarbonAnalyzeResponse(BaseModel):
    monthly_co2e_avoided_tonnes: float
    annual_co2e_avoided_tonnes: float
    acct_tokens_earned: int
    token_revenue_usd: float
    carbon_levers: List[CarbonLever]
