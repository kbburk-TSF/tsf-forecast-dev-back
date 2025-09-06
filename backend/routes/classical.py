from fastapi import APIRouter, HTTPException, Query, Response
from sqlalchemy import text
from backend.database import engine
import pandas as pd
import numpy as np
from datetime import date
import io

from statsmodels.tsa.holtwinters import ExponentialSmoothing, SimpleExpSmoothing
from pmdarima import auto_arima

router = APIRouter(prefix="/classical", tags=["classical"])

DB_SCHEMA = "demo_air_quality"
TABLE = f"{DB_SCHEMA}.air_quality_raw"

def _load_series(state: str, target_value: str, agg: str):
    sql = f'''
    SELECT "Date Local"::date AS date, "Arithmetic Mean" AS value
    FROM {TABLE}
    WHERE "State Name" = :state AND "Parameter Name" = :parameter
    ORDER BY "Date Local"
    '''
    with engine.begin() as conn:
        rows = conn.execute(text(sql), {"state": state, "parameter": target_value}).mappings().all()
    if not rows:
        raise HTTPException(status_code=404, detail="No data for given filters")
    df = pd.DataFrame(rows)
    if agg == "sum":
        df = df.groupby("date", as_index=False)["value"].sum()
    elif agg == "mean":
        df = df.groupby("date", as_index=False)["value"].mean()
    else:  # count
        df = df.groupby("date", as_index=False)["value"].size().rename(columns={"size":"value"})
    df["date"] = pd.to_datetime(df["date"])
    return df.rename(columns={"date":"DATE","value":"VALUE"}).sort_values("DATE")

def _days_in_month(dt):
    first = dt.replace(day=1)
    if first.month == 12:
        nxt = first.replace(year=first.year+1, month=1)
    else:
        nxt = first.replace(month=first.month+1)
    return (nxt - first).days

def _month_start(dt):
    return dt.replace(day=1)

def _quarter_start(dt):
    q = 3*((dt.month-1)//3)+1
    return dt.replace(month=q, day=1)

def _generate_monthly_forecasts(hist: pd.DataFrame):
    first = _month_start(pd.to_datetime(hist["DATE"].min()))
    start_m = (first + pd.offsets.MonthBegin(1)).normalize()
    end = pd.to_datetime(hist["DATE"].max())
    cur = _month_start(start_m)
    frames = []
    while cur <= end:
        prev_m_start = (cur - pd.offsets.MonthBegin(1)).normalize()
        prev_m_end = (cur - pd.offsets.Day(1)).normalize()
        prev_slice = hist[(hist["DATE"] >= prev_m_start) & (hist["DATE"] <= prev_m_end)]
        if prev_slice.empty:
            cur = (cur + pd.offsets.MonthBegin(1)).normalize()
            continue
        y = prev_slice["VALUE"].astype(float).to_numpy()

        try:
            arima_m = auto_arima(y, error_action="ignore", suppress_warnings=True).predict(n_periods=_days_in_month(cur))
        except Exception:
            arima_m = np.repeat(float(np.nan), _days_in_month(cur))

        try:
            ses = ExponentialSmoothing(y, trend="mul", seasonal=None).fit(optimized=True, use_brute=True)
            ses_m = ses.forecast(_days_in_month(cur))
        except Exception:
            try:
                ses2 = SimpleExpSmoothing(y).fit(optimized=True)
                ses_m = ses2.forecast(_days_in_month(cur))
            except Exception:
                ses_m = np.repeat(float(np.nan), _days_in_month(cur))

        try:
            hwes = ExponentialSmoothing(y, trend="add", damped_trend=True, seasonal=None).fit(optimized=True, use_brute=True)
            hwes_m = hwes.forecast(_days_in_month(cur))
        except Exception:
            hwes_m = np.repeat(float(np.nan), _days_in_month(cur))

        dates = pd.date_range(cur, periods=_days_in_month(cur), freq="D")
        frames.append(pd.DataFrame({
            "DATE": dates,
            "ARIMA-M": arima_m,
            "SES-M": ses_m,
            "HWES-M": hwes_m
        }))
        cur = (cur + pd.offsets.MonthBegin(1)).normalize()

    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=["DATE","ARIMA-M","SES-M","HWES-M"])

def _generate_quarterly_forecasts(hist: pd.DataFrame):
    first_q = _quarter_start(pd.to_datetime(hist["DATE"].min()))
    start_q = (first_q + pd.offsets.QuarterBegin(startingMonth=first_q.month) + pd.offsets.QuarterBegin()).normalize()
    end = pd.to_datetime(hist["DATE"].max())
    cur = _quarter_start(start_q)
    frames = []
    while cur <= end:
        prev_q_start = (cur - pd.offsets.QuarterBegin(startingMonth=1)).normalize()
        prev_q_end = (cur - pd.offsets.Day(1)).normalize()
        prev_slice = hist[(hist["DATE"] >= prev_q_start) & (hist["DATE"] <= prev_q_end)]
        if prev_slice.empty:
            cur = (cur + pd.offsets.QuarterBegin(startingMonth=1)).normalize()
            continue
        y = prev_slice["VALUE"].astype(float).to_numpy()
        q_days = len(pd.date_range(cur, (cur + pd.offsets.QuarterEnd()), freq="D"))

        try:
            arima_q = auto_arima(y, error_action="ignore", suppress_warnings=True).predict(n_periods=q_days)
        except Exception:
            arima_q = np.repeat(float(np.nan), q_days)

        try:
            ses = ExponentialSmoothing(y, trend="mul", seasonal=None).fit(optimized=True, use_brute=True)
            ses_q = ses.forecast(q_days)
        except Exception:
            try:
                ses2 = SimpleExpSmoothing(y).fit(optimized=True)
                ses_q = ses2.forecast(q_days)
            except Exception:
                ses_q = np.repeat(float(np.nan), q_days)

        try:
            hwes = ExponentialSmoothing(y, trend="add", damped_trend=True, seasonal=None).fit(optimized=True, use_brute=True)
            hwes_q = hwes.forecast(q_days)
        except Exception:
            hwes_q = np.repeat(float(np.nan), q_days)

        dates = pd.date_range(cur, (cur + pd.offsets.QuarterEnd()), freq="D")
        frames.append(pd.DataFrame({
            "DATE": dates,
            "ARIMA-Q": arima_q,
            "SES-Q": ses_q,
            "HWES-Q": hwes_q
        }))
        cur = (cur + pd.offsets.QuarterBegin(startingMonth=1)).normalize()

    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=["DATE","ARIMA-Q","SES-Q","HWES-Q"])

def _compose_instance_name(target_value: str, params: list, forecast_type: str):
    parts = ["F", target_value] + [p for p in params if p] + [forecast_type]
    return "_".join(str(p).replace(" ", "_") for p in parts)

@router.get("/export_target")
def export_target_csv(
    db: str = Query("demo_air_quality"),
    target_value: str = Query(..., description="TARGET VARIABLE (e.g., O3, NO2)"),
    state: str = Query(None),
    county: str = Query(None),
    city: str = Query(None),
    cbsa: str = Query(None),
    agg: str = Query("mean", pattern="^(mean|sum|count)$"),
    forecast_type: str = Query("F", pattern="^(F|S)$")
):
    if db != "demo_air_quality":
        raise HTTPException(status_code=400, detail="Only demo_air_quality supported in demo.")
    df = _load_series(state or "", target_value, agg)
    params = [p for p in [state, county, city, cbsa] if p]
    fname_base = _compose_instance_name(target_value, params, forecast_type)
    buf = io.StringIO()
    df.to_csv(buf, index=False)  # DATE, VALUE
    return Response(content=buf.getvalue(), media_type="text/csv",
                    headers={"Content-Disposition": f'attachment; filename="{fname_base}.csv"'})

@router.get("/export_classical")
def export_classical_csv(
    db: str = Query("demo_air_quality"),
    target_value: str = Query(..., description="TARGET VARIABLE (e.g., O3, NO2)"),
    state: str = Query(...),
    agg: str = Query("mean", pattern="^(mean|sum)$"),
    forecast_type: str = Query("F", pattern="^(F|S)$")
):
    if db != "demo_air_quality":
        raise HTTPException(status_code=400, detail="Only demo_air_quality supported in demo.")
    hist = _load_series(state, target_value, agg)
    m = _generate_monthly_forecasts(hist)
    q = _generate_quarterly_forecasts(hist)
    out = pd.DataFrame({"DATE": pd.to_datetime(hist["DATE"])})
    out = out.merge(hist, on="DATE", how="outer")
    out = out.merge(m, on="DATE", how="outer")
    out = out.merge(q, on="DATE", how="outer")
    out = out.sort_values("DATE")
    out["DATE"] = out["DATE"].dt.strftime("%Y-%m-%d")
    out = out[["DATE","VALUE","ARIMA-M","ARIMA-Q","SES-M","SES-Q","HWES-M","HWES-Q"]]
    params = [p for p in [state] if p]
    input_name = _compose_instance_name(target_value, params, forecast_type)
    output_name = input_name[2:] if input_name.startswith("F_") else input_name
    buf = io.StringIO()
    out.to_csv(buf, index=False)
    return Response(content=buf.getvalue(), media_type="text/csv",
                    headers={"Content-Disposition": f'attachment; filename="{output_name}.csv"'})
