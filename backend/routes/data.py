from fastapi import APIRouter, Query
from sqlalchemy import text
from backend.database import engine

router = APIRouter(prefix="/data", tags=["data"])

@router.get("/air_quality/last")
def last_rows(limit: int = Query(50, ge=1, le=500)):
    sql = """
    SELECT id, date_local, parameter_name, arithmetic_mean, local_site_name, state_name, county_name, city_name, cbsa_name, created_at
    FROM public.air_quality_raw
    ORDER BY id DESC
    LIMIT :limit
    """
    with engine.begin() as conn:
        rows = conn.execute(text(sql), {"limit": limit}).mappings().all()
    return {"rows": [dict(r) for r in rows]}

@router.get("/air_quality/last_date")
def last_date(state: str, parameter: str):
    sql = """
    SELECT MAX(date_local) AS max_date
    FROM public.air_quality_raw
    WHERE state_name = :state AND parameter_name = :parameter
    """
    with engine.begin() as conn:
        row = conn.execute(text(sql), {"state": state, "parameter": parameter}).first()
    return {"state": state, "parameter": parameter, "last_date": row[0] if row else None}
    
    from fastapi import HTTPException

@router.get("/debug/air_quality/schema_live")
def debug_air_quality_schema_live():
    """
    Returns the live column list from Neon for public.air_quality_raw.
    """
    sql = """
    SELECT column_name, data_type, is_nullable
    FROM information_schema.columns
    WHERE table_schema='public' AND table_name='air_quality_raw'
    ORDER BY ordinal_position
    """
    try:
        with engine.begin() as conn:
            rows = conn.execute(text(sql)).mappings().all()
        return {"table": "public.air_quality_raw", "columns": [dict(r) for r in rows]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

