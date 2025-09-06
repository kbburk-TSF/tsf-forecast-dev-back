from fastapi import APIRouter, Query, HTTPException
from sqlalchemy import text
from backend.database import engine

router = APIRouter(prefix="/data", tags=["data"])

DB_SCHEMA = "demo_air_quality"
TABLE = f"{DB_SCHEMA}.air_quality_raw"

def _safe_query(sql: str, params: dict):
    try:
        with engine.begin() as conn:
            res = conn.execute(text(sql), params).mappings().all()
            return [dict(r) for r in res]
    except Exception as e:
        raise HTTPException(status_code=500, detail={"sql": sql, "params": params, "error": str(e)})

@router.get("/air_quality/last")
def last_rows(limit: int = Query(50, ge=1, le=500)):
    sql = f"""
    SELECT date_local, parameter_name, arithmetic_mean, local_site_name,
           state_name, county_name, city_name, cbsa_name, created_at
    FROM {TABLE}
    ORDER BY date_local DESC, created_at DESC
    LIMIT :limit
    """
    rows = _safe_query(sql, {"limit": limit})
    return {"rows": rows}

@router.get("/air_quality/last_date")
def last_date(state: str, parameter: str):
    sql = f"""
    SELECT MAX(date_local) AS max_date
    FROM {TABLE}
    WHERE state_name = :state AND parameter_name = :parameter
    """
    try:
        with engine.begin() as conn:
            row = conn.execute(text(sql), {"state": state, "parameter": parameter}).first()
            max_date = row[0] if row else None
    except Exception as e:
        raise HTTPException(status_code=500, detail={"sql": sql, "state": state, "parameter": parameter, "error": str(e)})
    return {"state": state, "parameter": parameter, "last_date": max_date}

@router.get("/debug/tables")
def debug_tables():
    sql = """
    SELECT table_schema, table_name
    FROM information_schema.tables
    WHERE table_type='BASE TABLE'
    ORDER BY table_schema, table_name
    """
    with engine.begin() as conn:
        rows = conn.execute(text(sql)).mappings().all()
    return {"tables": [dict(r) for r in rows]}

@router.get("/debug/air_quality/schema_live")
def debug_air_quality_schema_live():
    sql = """
    SELECT column_name, data_type, is_nullable
    FROM information_schema.columns
    WHERE table_schema = :schema AND table_name='air_quality_raw'
    ORDER BY ordinal_position
    """
    with engine.begin() as conn:
        rows = conn.execute(text(sql), {"schema": DB_SCHEMA}).mappings().all()
    return {"table": f"{DB_SCHEMA}.air_quality_raw", "columns": [dict(r) for r in rows]}

@router.get("/debug/air_quality/sample")
def debug_sample(limit: int = Query(5, ge=1, le=50)):
    sql = f"""
    SELECT date_local, parameter_name, arithmetic_mean,
           local_site_name, state_name, county_name, city_name, cbsa_name
    FROM {TABLE}
    LIMIT :limit
    """
    rows = _safe_query(sql, {"limit": limit})
    return {"rows": rows}
