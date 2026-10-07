try:
    from app.schemas.pydantic_models import (
        CarbonAnalyzeRequest, CarbonAnalyzeResponse, CarbonLever
    )
except ModuleNotFoundError:
    from schemas.pydantic_models import (
        CarbonAnalyzeRequest, CarbonAnalyzeResponse, CarbonLever
    )

class SustainabilityService:
    def analyze_carbon(self, req: CarbonAnalyzeRequest) -> CarbonAnalyzeResponse:
        vehicles = req.vehicles_processed
        sl_rate = req.second_life_rate
        al_purity = req.aluminium_recovery_purity
        re_ratio = req.renewable_energy_ratio

        batt_saved = vehicles * sl_rate * 4.2
        al_saved = vehicles * 42.0 * al_purity * 0.0085
        logistics_saved = vehicles * 0.18 * 0.22
        re_saved = vehicles * 0.5 * re_ratio * 0.3

        monthly_co2e = batt_saved + al_saved + logistics_saved + re_saved
        annual_co2e = monthly_co2e * 12.0
        
        tokens = int(round(monthly_co2e))
        revenue = tokens * 14.20

        levers = [
            CarbonLever(name="Battery Second-Life Routing", co2e_saved_tonnes=round(batt_saved, 1), percentage_contribution=round(batt_saved/monthly_co2e*100, 1)),
            CarbonLever(name="High-Purity Aluminium Recovery", co2e_saved_tonnes=round(al_saved, 1), percentage_contribution=round(al_saved/monthly_co2e*100, 1)),
            CarbonLever(name="AI Disassembly Logistics Optimization", co2e_saved_tonnes=round(logistics_saved, 1), percentage_contribution=round(logistics_saved/monthly_co2e*100, 1)),
            CarbonLever(name="Facility Renewable Energy Transition", co2e_saved_tonnes=round(re_saved, 1), percentage_contribution=round(re_saved/monthly_co2e*100, 1)),
        ]

        return CarbonAnalyzeResponse(
            monthly_co2e_avoided_tonnes=round(monthly_co2e, 1),
            annual_co2e_avoided_tonnes=round(annual_co2e, 1),
            acct_tokens_earned=tokens,
            token_revenue_usd=round(revenue, 2),
            carbon_levers=levers
        )

sustainability_service = SustainabilityService()
