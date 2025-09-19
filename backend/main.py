# BUILD: 2025-09-19 CORS+Health
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

try:
    from backend.version import BUILD_ID
except Exception:
    BUILD_ID = "dev"

from backend.routes.data import router as data_router
from backend.routes.aggregate import router as aggregate_router
from backend.routes.forecast import router as forecast_router
from backend.routes.upload import router as upload_router
from backend.routes.classical import router as classical_router
from backend.routes.meta import router as meta_router

app = FastAPI(title="TSF Backend", version=BUILD_ID)

# Permissive CORS so the frontend can call us from anywhere
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Routers
app.include_router(data_router)
app.include_router(aggregate_router)
app.include_router(forecast_router)
app.include_router(upload_router)
app.include_router(classical_router)
app.include_router(meta_router)

# Health + root
@app.get("/health")
def health():
    return {"ok": True, "version": BUILD_ID}

@app.get("/")
def root():
    return {"service": "TSF Backend", "version": BUILD_ID}

