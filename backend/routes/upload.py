from fastapi import APIRouter, UploadFile, File, Query, HTTPException
from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError
from backend.database import engine
import pandas as pd
import io

router = APIRouter(prefix="/upload", tags=["upload"])

DB_SCHEMA = "demo_air_quality"
TABLE = f"{DB_SCHEMA}.air_quality_raw"

HEADER_MAP = {
    "date local": "date_local",
    "date_local": "date_local",
    "date": "date_local",
    "parameter name": "parameter_name",
    "parameter_name": "parameter_name",
    "arithmetic mean": "arithmetic_mean",
    "arithmetic_mean": "arithmetic_mean",
    "local site name": "local_site_name",
    "local_site_name": "local_site_name",
    "state name": "state_name",
    "state_name": "state_name",
    "county name": "county_name",
    "county_name": "county_name",
    "city name": "city_name",
    "city_name": "city_name",
    "cbsa name": "cbsa_name",
    "cbsa_name": "cbsa_name",
}
REQUIRED = ["date_local","parameter_name","arithmetic_mean","state_name"]

@router.post("/air_quality")
def upload_air_quality_csv(
    file: UploadFile = File(...),
    on_conflict: str = Query("ignore", pattern="^(ignore|fail)$")
):
    try:
        raw = file.file.read()
        df = pd.read_csv(io.BytesIO(raw))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Cannot read CSV: {e}")

    df.columns = [HEADER_MAP.get(str(c).strip().lower(), str(c).strip().lower()) for c in df.columns]
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        raise HTTPException(status_code=400, detail=f"Missing required columns: {missing}")

    df = df.copy()
    df["date_local"] = pd.to_datetime(df["date_local"]).dt.date
    df["arithmetic_mean"] = pd.to_numeric(df["arithmetic_mean"], errors="coerce")
    df = df.dropna(subset=["date_local","arithmetic_mean"])
    for opt in ["local_site_name","county_name","city_name","cbsa_name"]:
        if opt not in df.columns:
            df[opt] = None

    inserted = 0
    skipped = 0
    conflict_sql = "ON CONFLICT DO NOTHING" if on_conflict == "ignore" else ""
    cols = ["date_local","parameter_name","arithmetic_mean","local_site_name","state_name","county_name","city_name","cbsa_name"]
    values_template = ",".join([f":{c}" for c in cols])
    insert_sql = f"""
        INSERT INTO {TABLE} ({",".join(cols)})
        VALUES ({values_template})
        {conflict_sql};
    """
    with engine.begin() as conn:
        for _, row in df.iterrows():
            try:
                conn.execute(text(insert_sql), {c: row.get(c) for c in cols})
                inserted += 1
            except SQLAlchemyError:
                if on_conflict == "ignore":
                    skipped += 1
                else:
                    raise

    return {"rows_inserted": inserted, "rows_skipped": skipped, "total": int(inserted + skipped)}
