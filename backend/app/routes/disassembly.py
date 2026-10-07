from fastapi import APIRouter, HTTPException

try:
    from app.schemas.pydantic_models import DisassemblyResponse
    from app.services.disassembly_service import disassembly_service
except ModuleNotFoundError:
    from schemas.pydantic_models import DisassemblyResponse
    from services.disassembly_service import disassembly_service

router = APIRouter(prefix="/api/disassembly", tags=["Intelligent Disassembly"])

@router.post("/detect", response_model=DisassemblyResponse)
def detect_and_plan():
    try:
        return disassembly_service.detect_and_plan()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
