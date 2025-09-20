# =====================================================================
# File: backend/main.py
# Version: v1.2.0 — 2025-09-20
# Change log:
# - v1.2.0 (2025-09-20): Backend form wired to DB drop-downs for
#   'Parameter Name' and 'State Name'; generates CSV via AVG(Arithmetic Mean)
#   grouped by 'Date Local'. Saves to staging_historical and downloads.
# - v1.1.x: Prior iterations (imports fix, jinja-free rendering).
# - v1.0.0: Initial bootstrap.
# =====================================================================

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.routes.forms_classical import router as forms_classical_router

app = FastAPI(title="TSF Backend", version="1.2.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"],
)

# Forms
app.include_router(forms_classical_router, prefix="/forms", tags=["forms"])

@app.get("/", tags=["health"])
def root():
    return {"ok": True, "app": "TSF Backend", "version": app.version}
