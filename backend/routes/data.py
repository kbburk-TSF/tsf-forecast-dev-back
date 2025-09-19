# BUILD: 2025-09-19 data endpoints for Air Quality Demo
from fastapi import APIRouter, HTTPException, Query
from sqlalchemy import text
try:
    from backend.database import engine
except Exception as e:
    # Fallback: explicit creation if project's database module differs
    raise

router = APIRouter(prefix="/data", tags=["data"])

# Accept both keys for backward compatibility
VALID_DBS = {"air_quality_demo_data", "demo_air_quality"}

TABLE = 'air_quality_raw'  # in public schema

@router.get("/{db}/targets")
def get_targets(db: str):
    if db not in VALID_DBS:
        raise HTTPException(status_code=404, detail=f"Unknown database {db}")
    q = text(f'''
        SELECT DISTINCT "Parameter Name" AS target
        FROM {TABLE}
        WHERE "Parameter Name" IS NOT NULL
        ORDER BY 1
    ''')
    with engine.connect() as conn:
        rows = conn.execute(q).all()
    return [r.target for r in rows]

@router.get("/{db}/filters")
def get_filters(db: str, target: str = Query(..., description="Value of Parameter Name")):
    if db not in VALID_DBS:
        raise HTTPException(status_code=404, detail=f"Unknown database {db}")
    q = text(f'''
        SELECT DISTINCT "State Name" AS state
        FROM {TABLE}
        WHERE "Parameter Name" = :target
        AND "State Name" IS NOT NULL
        ORDER BY 1
    ''')
    with engine.connect() as conn:
        rows = conn.execute(q, {"target": target}).all()
    return {"states": [r.state for r in rows]}
