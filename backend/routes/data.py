# Data routes (DB binding for Air Quality Demo)
from fastapi import APIRouter, HTTPException, Query
from backend.database import engine
from sqlalchemy import text

router = APIRouter(prefix="/data", tags=["data"])

DB_SCHEMA = "air_quality_demo_data"  # database (Render connects to this DB via DATABASE_URL)
TABLE = "air_quality_raw"            # table in public schema

@router.get("/{db}/targets")
def get_targets(db: str):
    if db not in ("air_quality_demo_data", "demo_air_quality"):
        raise HTTPException(status_code=400, detail=f"Unknown database {db}")
    q = text(f"""
        SELECT DISTINCT "Parameter Name" AS target
        FROM {TABLE}
        ORDER BY 1
    """)
    with engine.connect() as conn:
        rows = conn.execute(q).fetchall()
    return [r[0] for r in rows]

@router.get("/{db}/filters")
def get_filters(db: str, target: str = Query(...)):
    if db not in ("air_quality_demo_data", "demo_air_quality"):
        raise HTTPException(status_code=400, detail=f"Unknown database {db}")
    q = text(f"""
        SELECT DISTINCT "State Name"
        FROM {TABLE}
        WHERE "Parameter Name" = :target
        ORDER BY 1
    """)
    with engine.connect() as conn:
        states = [r[0] for r in conn.execute(q, {{'target': target}}).fetchall()]
    return {{"states": states}}
