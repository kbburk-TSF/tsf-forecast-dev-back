# BUILD: 2025-09-19 exact JSON contract for App.jsx
from fastapi import APIRouter, HTTPException, Query
from sqlalchemy import text
from backend.database import engine

router = APIRouter(prefix="/data", tags=["data"])

VALID_DBS = {"air_quality_demo_data", "demo_air_quality"}
TABLE = 'air_quality_raw'  # public schema

@router.get("/{db}/targets")
def get_targets(db: str):
    if db not in VALID_DBS:
        raise HTTPException(status_code=404, detail=f"Unknown database {db}")
    sql = text('SELECT DISTINCT "Parameter Name" AS target FROM ' + TABLE + ' WHERE "Parameter Name" IS NOT NULL ORDER BY 1')
    with engine.connect() as conn:
        rows = conn.execute(sql).all()
    return {"targets": [r.target for r in rows]}

@router.get("/{db}/filters")
def get_filters(db: str, target: str = Query(..., alias="target", description="Value for Parameter Name")):
    if db not in VALID_DBS:
        raise HTTPException(status_code=404, detail=f"Unknown database {db}")
    sql = text('SELECT DISTINCT "State Name" AS state FROM ' + TABLE + ' WHERE "Parameter Name" = :target AND "State Name" IS NOT NULL ORDER BY 1')
    with engine.connect() as conn:
        rows = conn.execute(sql, {"target": target}).all()
    return {"filters": {"states": [r.state for r in rows] }}
