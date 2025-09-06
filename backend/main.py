from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import text
from backend.database import engine
import os

# Routers
from backend.routes.upload import router as upload_router
from backend.routes.data import router as data_router
from backend.routes.aggregate import router as agg_router
from backend.routes.forecast import router as forecast_router
from backend.routes import meta

app = FastAPI(title="TSF Backend", version="1.0.9")

# ---- CORS ----
env_origins = os.getenv("ALLOWED_ORIGINS", "").strip()
if env_origins:
    allowed = [o.strip() for o in env_origins.split(",") if o.strip()]
    app.add_middleware(
        CORSMiddleware,
        allow_origins=allowed,
        allow_credentials=True,
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["*"],
    )
else:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=False,
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["*"],
    )

# Feature routes
app.include_router(upload_router)
app.include_router(data_router)
app.include_router(agg_router)
app.include_router(forecast_router)
app.include_router(meta.router)

@app.get("/health")
def health():
    return {"status": "ok", "database": "up"}

@app.get("/version")
def version():
    v = "unknown"
    try:
        with open("VERSION", "r") as f:
            v = f.read().strip()
    except Exception:
        pass
    return {"version": v}
