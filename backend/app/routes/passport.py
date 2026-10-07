from fastapi import APIRouter, HTTPException

try:
    from app.schemas.pydantic_models import PassportGenerateRequest, PassportResponse
    from app.services.passport_service import passport_service
except ModuleNotFoundError:
    from schemas.pydantic_models import PassportGenerateRequest, PassportResponse
    from services.passport_service import passport_service

router = APIRouter(prefix="/api/passport", tags=["Material Passport"])

@router.post("/generate", response_model=PassportResponse)
def generate_passport(req: PassportGenerateRequest):
    try:
        return passport_service.generate_passport(req)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
