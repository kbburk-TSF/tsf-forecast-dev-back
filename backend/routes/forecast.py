from fastapi import APIRouter, HTTPException, Query
from sqlalchemy import text
from backend.database import engine
from backend.utils_forecast import forecast_seasonal_naive_dow, forecast_ewma
import pandas as pd

router = APIRouter(prefix="/forecast", tags=["forecast"])

@router.get("/state_daily")
def forecast_state_daily(state: str, parameter: str, h: int = Query(30, ge=1, le=365),
                         agg: str = Query("mean", pattern="^(mean|sum)$"),
                         method: str = Query("seasonal_naive_dow", pattern="^(seasonal_naive_dow|ewma)$")):
    sql = """
    SELECT date_local::date AS date, arithmetic_mean AS value
    FROM public.air_quality_raw
    WHERE state_name = :state AND parameter_name = :parameter
    ORDER BY date_local
    """
    with engine.begin() as conn:
        rows = conn.execute(text(sql), {"state": state, "parameter": parameter}).mappings().all()
    if not rows:
        raise HTTPException(status_code=404, detail="No data for given filters")
    df = pd.DataFrame(rows)
    hist = df.groupby("date", as_index=False)["value"].mean() if agg == "mean" else df.groupby("date", as_index=False)["value"].sum()
    fc = forecast_seasonal_naive_dow(hist, h=h, lookback_weeks=8) if method == "seasonal_naive_dow" else forecast_ewma(hist, h=h, span=14)
    return {
        "state": state,
        "parameter": parameter,
        "agg": agg,
        "method": method,
        "history": hist.assign(date=hist["date"].astype(str)).to_dict(orient="records"),
        "forecast": fc.assign(date=fc["date"].astype(str)).to_dict(orient="records"),
    }
