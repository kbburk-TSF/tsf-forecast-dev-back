# =====================================================================
# File: backend/main.py
# Version: v1.1.0 — 2025-09-20
# Change log:
# - v1.1.0 (2025-09-20): Add Jinja2 templates, static mount, and register
#   forms_classical router to serve HTML form and handle CSV generation.
# - v1.0.0: Initial FastAPI app bootstrap (health/debug, APIs, CORS).
# =====================================================================

import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

# Routers (existing)
# from routes.upload import router as upload_router
# from routes.aggregate import router as aggregate_router
# from routes.meta import router as meta_router
# from routes.debug import router as debug_router
# from routes.data import router as data_router
# from routes.forecast import router as forecast_router

# New forms router
from routes.forms_classical import router as forms_classical_router

app = FastAPI(title="TSF Backend", version="1.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")
if os.path.isdir(STATIC_DIR):
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

# app.include_router(upload_router, prefix="/upload", tags=["upload"])
# app.include_router(aggregate_router, prefix="/aggregate", tags=["aggregate"])
# app.include_router(meta_router, prefix="/meta", tags=["meta"])
# app.include_router(debug_router, prefix="/debug", tags=["debug"])
# app.include_router(forecast_router, prefix="/forecast", tags=["forecast"])
# app.include_router(data_router, prefix="/data", tags=["data"])

app.include_router(forms_classical_router, prefix="/forms", tags=["forms"])

@app.get("/", tags=["health"])
def root():
    return {"ok": True, "app": "TSF Backend", "version": app.version}
