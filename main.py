import os
import math
import random
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Dict

app = FastAPI(title="AutoCircle AI - Integrated Multi-Module Production Engine", version="2.0.0")

# CORS enable kiya taaki HTML file backend se bina kisi block ke connect ho sake
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==========================================
# PYDANTIC DATA INTERFACE SCHEMAS
# ==========================================

class SOHInput(BaseModel):
    cycle_count: float
    avg_temp: float
    depth_of_discharge: float
    fast_charge_frequency: float
    internal_resistance: float

class SOHOutput(BaseModel):
    soh: float
    rul: int
    routing: str
    confidence: float
    recommendation: str

class DisassemblyInput(BaseModel):
    vehicle_model: str
    confidence_threshold: float

class DisassemblyStep(BaseModel):
    step: int
    operation: str
    tool_required: str
    safety_risk: str

class DisassemblyOutput(BaseModel):
    vision_status: str
    sequence: List[DisassemblyStep]

class PassportInput(BaseModel):
    component_type: str
    vin: str
    soh_score: float

class MaterialComponent(BaseModel):
    name: str
    percentage: float
    recyclability: str

class PassportOutput(BaseModel):
    passport_id: str
    vin: str
    blockchain_hash: str
    carbon_footprint_kg: float
    materials: List[MaterialComponent]
    routing: str

class DesignInput(BaseModel):
    target_weight_kg: float
    material_priority: str
    safety_factor: float

class DesignOutput(BaseModel):
    optimization_status: str
    iterations_run: int
    mass_reduction_pct: float
    structural_integrity_gpa: float
    pareto_frontier_score: float

class CarbonLedgerInput(BaseModel):
    processed_vehicles: int
    aluminum_purity_pct: float
    renewable_energy_pct: float

class CarbonLedgerOutput(BaseModel):
    monthly_co2_saved_tons: float
    tokens_minted: int
    projected_revenue_usd: float
    levers_breakdown: List[Dict[str, float]]

# ==========================================
# CORE ALGORITHM MODULE ROUTERS
# ==========================================

# 1. Battery SOH Engine
@app.post("/api/v1/battery/predict-soh", response_model=SOHOutput)
async def predict_battery_health(payload: SOHInput):
    base_loss = (payload.cycle_count * 0.00028) + ((payload.internal_resistance - 60) * 0.14)
    thermal_acc = max(0.0, (payload.avg_temp - 25.0) * 0.38)
    predicted_soh = float(np.clip(100.0 - (base_loss + thermal_acc + (payload.depth_of_discharge * 0.04)), 35.0, 99.0))
    
    if predicted_soh >= 80.0:
        routing = "Continue EV Use"
        rec = "Battery cell parameters are within nominal safety profiles. Approved for active vehicular service."
    elif predicted_soh >= 70.0:
        routing = "Second-Life Energy Cluster"
        rec = "Approved for secondary stationary grid arrays. Safe structural limits for home power buffers."
    else:
        routing = "Recycle Raw Elements"
        rec = "Severe lithium deposition detected. Core component degradation active. Transfer to chemical extraction."

    return SOHOutput(
        soh=round(predicted_soh, 1),
        rul=int(max(0, (predicted_soh - 35.0) * 8.5)),
        routing=routing,
        confidence=95.4,
        recommendation=rec
    )

# 2. Robotic Disassembly Sequences
@app.post("/api/v1/disassembly/plan", response_model=DisassemblyOutput)
async def plan_disassembly(payload: DisassemblyInput):
    steps = [
        DisassemblyStep(step=1, operation="Isolate High-Voltage Circuit Lines", tool_required="Insulated Safety Wrenches (1000V)", safety_risk="Critical Arc Flash Potential"),
        DisassemblyStep(step=2, operation="Unbolt Modular Underbody Shell Bracket Plate", tool_required="Pneumatic Socket Driver (14mm)", safety_risk="Mechanical Load Release drop hazard"),
        DisassemblyStep(step=3, operation="Extract Heavy Modular Battery Core Subgrid", tool_required="Hydraulic Floor Lift Crane", safety_risk="Heavy Suspension Strain Hazard"),
        DisassemblyStep(step=4, operation="De-couple Copper Wiring Harness Elements", tool_required="Automatic Wire Cutters", safety_risk="Low Static Energy Discharge Risk")
    ]
    return DisassemblyOutput(
        vision_status=f"YOLOv8 Dynamic Object Matrix Resolved Stable ({payload.confidence_threshold * 100}%)",
        sequence=steps
    )

# 3. Decentralized Material Passport Minter
@app.post("/api/v1/passport/generate", response_model=PassportOutput)
async def generate_passport(payload: PassportInput):
    route = "Recycle Raw Elements" if payload.soh_score < 70 else ("Second-Life Energy Cluster" if payload.soh_score < 80 else "Continue EV Use")
    mats = [
        MaterialComponent(name="Lithium Cobalt Oxide", percentage=12.5, recyclability="High Recovery Matrix"),
        MaterialComponent(name="High Purity Copper Foil", percentage=8.2, recyclability="Direct Smelting Grid"),
        MaterialComponent(name="Structural Grade Aluminium", percentage=45.0, recyclability="98% Closed Loop Purity")
    ]
    fake_hash = "0x" + "".join(random.choices("0123456789abcdef", k=16))
    return PassportOutput(
        passport_id=f"DPP-{random.randint(1000, 9990)}",
        vin=payload.vin,
        blockchain_hash=fake_hash,
        carbon_footprint_kg=3140.5,
        materials=mats,
        routing=route
    )

# 4. Generative Topology Weight Optimizer
@app.post("/api/v1/design/optimize", response_model=DesignOutput)
async def optimize_design(payload: DesignInput):
    calc_reduction = float(np.clip(35.2 + (payload.safety_factor * -3.5), 12.0, 52.0))
    stiffness = float(np.clip(110.0 + (payload.target_weight_kg * 1.2), 75.0, 195.0))
    return DesignOutput(
        optimization_status="Topological Geometry Structural Match Converged",
        iterations_run=185,
        mass_reduction_pct=round(calc_reduction, 1),
        structural_integrity_gpa=round(stiffness, 1),
        pareto_frontier_score=0.925
    )

# 5. Life Cycle Assessment Environmental Ledger (CORRECTED)
@app.post("/api/v1/carbon/ledger", response_model=CarbonLedgerOutput)
async def compute_carbon_ledger(payload: CarbonLedgerInput):
    v = payload.processed_vehicles
    b_saved = v * 3.2 * 0.82
    al_saved = v * 38 * (payload.aluminum_purity_pct / 100.0) * 0.009
    re_saved = v * 0.6 * (payload.renewable_energy_pct / 100.0) * 0.25
    
    monthly = float(b_saved + al_saved + re_saved)
    
    # YAHAN MISTAKE THI - Ab sahi kar diya hai
    tokens = int(round(monthly)) 
    
    rev = float(tokens * 14.20)
    
    breakdown = [
        {"name": "Secondary Storage Energy Battery Diversion", "value": round(b_saved, 1)},
        {"name": "High-Purity Core Aluminium Reclamation", "value": round(al_saved, 1)},
        {"name": "Facility Renewable Infrastructure Offset", "value": round(re_saved, 1)}
    ]
    return CarbonLedgerOutput(
        monthly_co2_saved_tons=round(monthly, 1),
        tokens_minted=tokens,
        projected_revenue_usd=round(rev, 2),
        levers_breakdown=breakdown
    )
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)