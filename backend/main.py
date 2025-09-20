# =====================================================================
# File: backend/main.py
# Version: v1.2.1 — 2025-09-20
# Change log:
# - v1.2.1: Patch quoting bug in forms_classical._options().
# - v1.2.0: DB-backed form; aggregation + CSV save/download.
# =====================================================================

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.routes.forms_classical import router as forms_classical_router

app = FastAPI(title="TSF Backend", version="1.2.1")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"],
)

app.include_router(forms_classical_router, prefix="/forms", tags=["forms"])

@app.get("/", tags=["health"])
def root():
    return {"ok": True, "app": "TSF Backend", "version": app.version}
