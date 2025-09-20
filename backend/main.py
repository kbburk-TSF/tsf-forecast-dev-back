# =====================================================================
# File: backend/main.py
# Version: v1.1.1 — 2025-09-20
# Change log:
# - v1.1.1 (2025-09-20): Fix imports to use absolute package path
#   'backend.routes.forms_classical'. Add package initializers.
# - v1.1.0 (2025-09-20): Add templates/static and forms router.
# - v1.0.0: Initial FastAPI bootstrap.
# =====================================================================

import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

# New forms router (absolute import under the 'backend' package)
from backend.routes.forms_classical import router as forms_classical_router

app = FastAPI(title="TSF Backend", version="1.1.1")

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

# Register forms
app.include_router(forms_classical_router, prefix="/forms", tags=["forms"])

@app.get("/", tags=["health"])
def root():
    return {"ok": True, "app": "TSF Backend", "version": app.version}
