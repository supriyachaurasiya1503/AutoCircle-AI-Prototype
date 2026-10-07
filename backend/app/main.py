import os
import sys

# Ensure backend root and app directory are in sys.path so python main.py works from anywhere
current_dir = os.path.dirname(os.path.abspath(__file__))
backend_dir = os.path.dirname(current_dir)

if backend_dir not in sys.path:
    sys.path.insert(0, backend_dir)
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

try:
    from app.routes import battery, disassembly, passport, sustainability
    from app.services.battery_service import battery_service, final_battery_model
except ModuleNotFoundError:
    from routes import battery, disassembly, passport, sustainability
    from services.battery_service import battery_service, final_battery_model

app = FastAPI(
    title="AutoCircle AI — Circular Economy Intelligence Platform API",
    description="Prototype service endpoints for battery SOH prediction, disassembly planning, EU material passport generation, and sustainability analytics.",
    version="2.0.0"
)

# Enable CORS for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API Routers
app.include_router(battery.router)
app.include_router(disassembly.router)
app.include_router(passport.router)
app.include_router(sustainability.router)

@app.get("/health", tags=["Health Check"])
def health_check():
    return {
        "status": "healthy",
        "service": "AutoCircle AI Backend",
        "models_loaded": final_battery_model.is_loaded,
        "engine": "XGBoost rolling SOH forecast + FastAPI"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="127.0.0.1", port=8000, reload=True)
