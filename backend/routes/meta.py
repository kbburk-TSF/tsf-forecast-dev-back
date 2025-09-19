# Meta routes with DB map
from fastapi import APIRouter, HTTPException

router = APIRouter(prefix="/meta", tags=["meta"])

DB_SCHEMA_MAP = {
    "air_quality_demo_data": {
        "table": "air_quality_raw",
        "target_col": "Parameter Name",
        "value_col": "Arithmetic Mean",
        "filters": ["State Name", "County Name", "City Name", "CBSA Name"],
    },
    "demo_air_quality": {  # backward compatibility
        "table": "air_quality_raw",
        "target_col": "Parameter Name",
        "value_col": "Arithmetic Mean",
        "filters": ["State Name", "County Name", "City Name", "CBSA Name"],
    },
}

@router.get("/db/{db}")
def db_meta(db: str):
    cfg = DB_SCHEMA_MAP.get(db)
    if not cfg:
        raise HTTPException(status_code=400, detail=f"Unknown database {db}")
    return cfg
