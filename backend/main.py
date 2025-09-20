# =====================================================================
# File: backend/main.py
# Version: v1.1.4 — 2025-09-20
# Change log:
# - v1.1.4 (2025-09-20): Pairs with routes update to use schema air_quality_demo_data.air_quality_raw.
# - v1.1.3: DSN sanitization for channel_binding.
# - v1.1.2: Jinja removed; inline HTML.
# - v1.1.1: Absolute imports; package initializers.
# - v1.1.0: Added forms router.
# - v1.0.0: Initial bootstrap.
# =====================================================================

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.routes.forms_classical import router as forms_classical_router

app = FastAPI(title="TSF Backend", version="1.1.4")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"],
)

app.include_router(forms_classical_router, prefix="/forms", tags=["forms"])

@app.get("/", tags=["health"])
def root():
    return {"ok": True, "app": "TSF Backend", "version": app.version}
