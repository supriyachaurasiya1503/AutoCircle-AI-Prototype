from fastapi import APIRouter, HTTPException

try:
    from app.schemas.pydantic_models import CarbonAnalyzeRequest, CarbonAnalyzeResponse
    from app.services.sustainability_service import sustainability_service
except ModuleNotFoundError:
    from schemas.pydantic_models import CarbonAnalyzeRequest, CarbonAnalyzeResponse
    from services.sustainability_service import sustainability_service

router = APIRouter(prefix="/api/sustainability", tags=["Sustainability Analytics"])

@router.post("/analyze", response_model=CarbonAnalyzeResponse)
def analyze_carbon(req: CarbonAnalyzeRequest):
    try:
        return sustainability_service.analyze_carbon(req)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
