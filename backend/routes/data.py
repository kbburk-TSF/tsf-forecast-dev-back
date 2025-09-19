# BUILD: pinpoint routes
from fastapi import APIRouter, HTTPException, Query
from sqlalchemy import text
from backend.database import engine

router = APIRouter(prefix="/data", tags=["data"])

VALID_DBS = {"air_quality_demo_data", "demo_air_quality"}
TABLE = 'air_quality_raw'  # public schema

@router.get("/{db}/targets")
def get_targets(db: str):
    print(f"[DATA] /targets db={db}")
    if db not in VALID_DBS:
        raise HTTPException(status_code=404, detail=f"Unknown database {db}")
    sql = text('SELECT DISTINCT "Parameter Name" AS target FROM ' + TABLE + ' WHERE "Parameter Name" IS NOT NULL ORDER BY 1')
    with engine.connect() as conn:
        rows = conn.execute(sql).all()
    return {"targets": [r.target for r in rows]}

@router.get("/{db}/filters")
def get_filters(db: str, target: str = Query(..., description="Value for Parameter Name")):
    print(f"[DATA] /filters db={db} target={target}")
    if db not in VALID_DBS:
        raise HTTPException(status_code=404, detail=f"Unknown database {db}")
    sql = text('SELECT DISTINCT "State Name" AS state FROM ' + TABLE + ' WHERE "Parameter Name" = :target AND "State Name" IS NOT NULL ORDER BY 1')
    with engine.connect() as conn:
        rows = conn.execute(sql, {"target": target}).all()
    return {"filters": {"states": [r.state for r in rows] }}
