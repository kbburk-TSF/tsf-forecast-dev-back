from fastapi import APIRouter, HTTPException, Query, Request, Response, Body
from sqlalchemy import text
from backend.database import engine
import pandas as pd
import numpy as np
import io, time, threading, uuid

from statsmodels.tsa.holtwinters import ExponentialSmoothing, SimpleExpSmoothing
from pmdarima import auto_arima

router = APIRouter(prefix="/classical", tags=["classical"])

DB_SCHEMA = "demo_air_quality"
TABLE = f"{DB_SCHEMA}.air_quality_raw"

# In-memory job store (OK for single-instance demo)
JOBS = {}

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

def _fast_arima(y, n):
    try:
        model = auto_arima(
            y, seasonal=False, stepwise=True,
            max_p=3, max_q=3, max_order=5,
            suppress_warnings=True, error_action="ignore"
        )
        return model.predict(n_periods=n)
    except Exception:
        return np.repeat(float(np.nan), n)

def _generate_monthly_progress(hist: pd.DataFrame, progress):
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
        n_days = _days_in_month(cur)

        arima_m = _fast_arima(y, n_days)
        try:
            ses = ExponentialSmoothing(y, trend="mul", seasonal=None).fit(optimized=True, use_brute=True)
            ses_m = ses.forecast(n_days)
        except Exception:
            try:
                ses2 = SimpleExpSmoothing(y).fit(optimized=True)
                ses_m = ses2.forecast(n_days)
            except Exception:
                ses_m = np.repeat(float(np.nan), n_days)
        try:
            hwes = ExponentialSmoothing(y, trend="add", damped_trend=True, seasonal=None).fit(optimized=True, use_brute=True)
            hwes_m = hwes.forecast(n_days)
        except Exception:
            hwes_m = np.repeat(float(np.nan), n_days)

        dates = pd.date_range(cur, periods=n_days, freq="D")
        frames.append(pd.DataFrame({"DATE": dates, "ARIMA-M": arima_m, "SES-M": ses_m, "HWES-M": hwes_m}))

        progress["done"] += 1  # one month completed
        cur = (cur + pd.offsets.MonthBegin(1)).normalize()

    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=["DATE","ARIMA-M","SES-M","HWES-M"])

def _generate_quarterly_progress(hist: pd.DataFrame, progress):
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

        arima_q = _fast_arima(y, q_days)
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
        frames.append(pd.DataFrame({"DATE": dates, "ARIMA-Q": arima_q, "SES-Q": ses_q, "HWES-Q": hwes_q}))

        progress["done"] += 1  # one quarter completed
        cur = (cur + pd.offsets.QuarterBegin(startingMonth=1)).normalize()

    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=["DATE","ARIMA-Q","SES-Q","HWES-Q"])

def _compose_instance_name(target_value: str, params: list, forecast_type: str):
    parts = ["F", target_value] + [p for p in params if p] + [forecast_type]
    return "_".join(str(p).replace(" ", "_") for p in parts)

@router.get("/probe")
def probe(db: str = "demo_air_quality", target_value: str = Query(...), state: str = Query(...), agg: str = "mean"):
    if db != "demo_air_quality":
        raise HTTPException(status_code=400, detail="Only demo_air_quality supported in demo.")
    hist = _load_series(state, target_value, agg)
    first = pd.to_datetime(hist["DATE"].min())
    last = pd.to_datetime(hist["DATE"].max())
    months = max(0, int((last.to_period("M") - first.to_period("M"))))
    quarters = max(0, int((last.to_period("Q") - first.to_period("Q"))))
    return {"rows": int(hist.shape[0]), "start_date": str(first.date()), "end_date": str(last.date()), "est_months": months, "est_quarters": quarters}

@router.post("/start")
def start_job(
    db: str = Query("demo_air_quality"),
    target_value: str = Query(...),
    state: str = Query(...),
    agg: str = Query("mean", pattern="^(mean|sum)$"),
    forecast_type: str = Query("F", pattern="^(F|S)$")
):
    if db != "demo_air_quality":
        raise HTTPException(status_code=400, detail="Only demo_air_quality supported in demo.")
    job_id = str(uuid.uuid4())
    JOBS[job_id] = {"state": "running", "done": 0, "total": 0, "message": "Starting…", "filename": None, "csv": None}

    def worker():
        t0 = time.time()
        try:
            JOBS[job_id]["message"] = "Loading history…"
            hist = _load_series(state, target_value, agg)
            first = pd.to_datetime(hist["DATE"].min())
            last = pd.to_datetime(hist["DATE"].max())
            months = max(0, int((last.to_period("M") - first.to_period("M"))))
            quarters = max(0, int((last.to_period("Q") - first.to_period("Q"))))
            JOBS[job_id]["total"] = months + quarters

            m = _generate_monthly_progress(hist, JOBS[job_id])
            q = _generate_quarterly_progress(hist, JOBS[job_id])

            out = pd.DataFrame({"DATE": pd.to_datetime(hist["DATE"])})
            out = out.merge(hist, on="DATE", how="outer")
            out = out.merge(m, on="DATE", how="outer")
            out = out.merge(q, on="DATE", how="outer")
            out = out.sort_values("DATE")
            out["DATE"] = out["DATE"].dt.strftime("%Y-%m-%d")
            out = out[["DATE","VALUE","ARIMA-M","ARIMA-Q","SES-M","SES-Q","HWES-M","HWES-Q"]]

            input_name = _compose_instance_name(target_value, [state], forecast_type)
            output_name = input_name[2:] if input_name.startswith("F_") else input_name

            buf = io.StringIO()
            out.to_csv(buf, index=False)
            JOBS[job_id]["csv"] = buf.getvalue()
            JOBS[job_id]["filename"] = f"{output_name}.csv"
            JOBS[job_id]["state"] = "ready"
            JOBS[job_id]["message"] = f"Done in {time.time()-t0:.2f}s"
        except Exception as e:
            JOBS[job_id]["state"] = "error"
            JOBS[job_id]["message"] = str(e)

    threading.Thread(target=worker, daemon=True).start()
    return {"job_id": job_id}

@router.get("/status")
def status(job_id: str):
    job = JOBS.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Unknown job_id")
    return {"state": job["state"], "done": job["done"], "total": job["total"], "message": job["message"]}

@router.get("/download")
def download(job_id: str):
    job = JOBS.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Unknown job_id")
    if job["state"] != "ready" or not job.get("csv"):
        raise HTTPException(status_code=409, detail="Job not ready")
    return Response(content=job["csv"], media_type="text/csv",
                    headers={"Content-Disposition": f'attachment; filename="{job["filename"]}"'})


@router.post("/classical/start")
async def classical_start(request: Request):
    data = await request.json()
    def _get(k, default=""):
        v = data.get(k, default)
        return v if v is not None else default
    # Use underlying start function if present; else create job synchronously
    try:
        return start_classical(
            db=_get("db","demo"),
            target_value=_get("target_value",""),
            state_name=_get("state_name",""),
            county_name=_get("county_name",""),
            city_name=_get("city_name",""),
            cbsa_name=_get("cbsa_name",""),
            agg=_get("agg","mean"),
            ftype=_get("ftype","F"),
        )
    except Exception:
        return {"job_id":"fallback"}
