import hashlib
import time
try:
    from app.schemas.pydantic_models import (
        PassportGenerateRequest, PassportResponse, PassportMaterialItem
    )
except ModuleNotFoundError:
    from schemas.pydantic_models import (
        PassportGenerateRequest, PassportResponse, PassportMaterialItem
    )

class PassportService:
    def generate_passport(self, req: PassportGenerateRequest) -> PassportResponse:
        passport_id = f"DPP-EU-{req.vin[:6]}-{hashlib.md5(req.vin.encode()).hexdigest()[:6].upper()}"
        
        comp_map = {
            "battery_pack": "EV High-Voltage Battery Pack (NMC 811)",
            "motor": "Permanent Magnet Synchronous Motor",
            "inverter": "Silicon Carbide (SiC) Inverter Module",
            "chassis": "Extruded Aluminium Subframe Structure"
        }
        comp_name = comp_map.get(req.component_type, "Automotive Component")
        
        if req.soh >= 80.0:
            routing = "Continue EV Use"
            suitability = "Optimal residual capacity. Retain in vehicle fleet for primary propulsion."
        elif req.soh >= 70.0:
            routing = "Second-Life Storage"
            suitability = "Excellent candidate for stationary BESS (Battery Energy Storage Systems) with 30-40% operational lifespan extension."
        else:
            routing = "Recycle Now"
            suitability = "Recommended for direct hydrometallurgical recycling line to recover raw battery chemicals."

        materials = [
            PassportMaterialItem(name="Lithium (Li)", percentage=8.2, recyclability="High (Hydrometallurgical)", svhc=False),
            PassportMaterialItem(name="Cobalt (Co)", percentage=4.5, recyclability="High (Solvent Extraction)", svhc=False),
            PassportMaterialItem(name="Nickel (Ni)", percentage=32.0, recyclability="High", svhc=False),
            PassportMaterialItem(name="Aluminium (Al6082)", percentage=28.5, recyclability="High (Closed-loop melt)", svhc=False),
            PassportMaterialItem(name="Copper (Cu)", percentage=12.0, recyclability="High", svhc=False),
            PassportMaterialItem(name="Graphite Anode", percentage=14.8, recyclability="Medium", svhc=False),
        ]
        
        raw_hash_str = f"{passport_id}:{req.vin}:{req.soh}:{time.time()}"
        tx_hash = f"0x{hashlib.sha256(raw_hash_str.encode()).hexdigest()[:16]}"
        
        qr_payload = f"https://autocircle.ai/passport/{passport_id}?vin={req.vin}&soh={req.soh}"

        return PassportResponse(
            passport_id=passport_id,
            component=comp_name,
            vin=req.vin,
            manufacture_date="2022-04-15",
            materials=materials,
            soh_percent=round(req.soh, 1),
            routing=routing,
            second_life_suitability=suitability,
            disposal_instructions="Depack in argon-glove chamber or automated shredder under nitrogen purge.",
            carbon_footprint_kg=4820.0,
            blockchain_tx=f"Polygon Network TX: {tx_hash}",
            compliance_badges=["EU Battery Regulation 2023/1542", "REACH SVHC Screened", "ISO 14040 LCA Aligned", "EPR Certified"],
            qr_code_data=qr_payload
        )

passport_service = PassportService()
