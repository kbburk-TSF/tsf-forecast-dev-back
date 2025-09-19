# CLEAN DATA ROUTES
from fastapi import APIRouter, HTTPException, Query
from sqlalchemy import text
from backend.database import engine

router = APIRouter(prefix="/data", tags=["data"])

VALID_DBS = {"air_quality_demo_data", "demo_air_quality"}

CANDIDATE_TABLES = [
    'air_quality_raw',                # preferred
    'air_quality_daily',              # fallback candidates (unknown schema)
    'public.air_quality_raw',
    'public.air_quality_daily',
]

def find_first_existing_table(conn):
    for t in CANDIDATE_TABLES:
        try:
            conn.execute(text(f"SELECT 1 FROM {t} LIMIT 1"))
            return t
        except Exception:
            continue
    return None

@router.get("/{db}/targets")
def get_targets(db: str):
    if db not in VALID_DBS:
        raise HTTPException(status_code=404, detail=f"Unknown database {db}")
    with engine.connect() as conn:
        table = find_first_existing_table(conn)
        if not table:
            return {"targets": [], "note": "no candidate table found in CANDIDATE_TABLES"}
        sql = text(f'SELECT DISTINCT "Parameter Name" AS target FROM {table} WHERE "Parameter Name" IS NOT NULL ORDER BY 1')
        try:
            rows = conn.execute(sql).all()
            return {"targets": [r.target for r in rows], "table": table}
        except Exception as e:
            # Return structured error so frontend shows real cause
            raise HTTPException(status_code=500, detail=f"targets query failed on {table}: {e}")

@router.get("/{db}/filters")
def get_filters(db: str, target: str = Query(..., description='Value of "Parameter Name"')):
    if db not in VALID_DBS:
        raise HTTPException(status_code=404, detail=f"Unknown database {db}")
    with engine.connect() as conn:
        table = find_first_existing_table(conn)
        if not table:
            return {"filters": {"states": []}, "note": "no candidate table found in CANDIDATE_TABLES"}
        sql = text(f'SELECT DISTINCT "State Name" AS state FROM {table} WHERE "Parameter Name" = :target AND "State Name" IS NOT NULL ORDER BY 1')
        try:
            rows = conn.execute(sql, {"target": target}).all()
            return {"filters": {"states": [r.state for r in rows]}, "table": table}
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"filters query failed on {table}: {e}")
