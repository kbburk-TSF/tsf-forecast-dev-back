from fastapi import APIRouter, HTTPException, Query
from sqlalchemy import text
from backend.database import engine
import pandas as pd

router = APIRouter(prefix="/aggregate", tags=["aggregate"])

DB_SCHEMA = "demo_air_quality"
TABLE = f"{DB_SCHEMA}.air_quality_raw"

@router.get("/state_daily")
def state_daily(state: str, parameter: str, agg: str = Query("mean", pattern="^(mean|sum)$")):
    sql = f"""
    SELECT date_local::date AS date, arithmetic_mean AS value
    FROM {TABLE}
    WHERE state_name = :state AND parameter_name = :parameter
    ORDER BY date_local
    """
    with engine.begin() as conn:
        rows = conn.execute(text(sql), {"state": state, "parameter": parameter}).mappings().all()
    if not rows:
        raise HTTPException(status_code=404, detail="No data for given filters")

    df = pd.DataFrame(rows)
    out = df.groupby("date", as_index=False)["value"].mean() if agg == "mean" else df.groupby("date", as_index=False)["value"].sum()
    out["value"] = out["value"].astype(float)
    return {"state": state, "parameter": parameter, "agg": agg, "series": out.to_dict(orient="records")}
