from fastapi import APIRouter, HTTPException, Query, Request
from fastapi.responses import FileResponse
from typing import Optional, Tuple
import os, uuid, threading, time, json
import pandas as pd
import numpy as np

router = APIRouter()

# ---------- Paths ----------
_BASE_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_BASE_DIR, ".."))               # backend/
_DATA_DIR = os.path.join(_ROOT, "data")
_INPUT_DIR = os.path.join(_DATA_DIR, "input")
_OUTPUT_DIR = os.path.join(_DATA_DIR, "output")
os.makedirs(_INPUT_DIR, exist_ok=True)
os.makedirs(_OUTPUT_DIR, exist_ok=True)

# ---------- Durable job store (survives Render sleep) ----------
JOBS_DIR = os.path.join(_OUTPUT_DIR, "_jobs")
os.makedirs(JOBS_DIR, exist_ok=True)
HEARTBEAT_SECS = 60  # if no update for this long while "running", treat as "paused"

def _job_path(job_id: str) -> str:
    return os.path.join(JOBS_DIR, f"{job_id}.json")

def _job_write(job_id: str, data: dict):
    data = dict(data)
    data["updated_at"] = time.time()
    with open(_job_path(job_id), "w", encoding="utf-8") as f:
        json.dump(data, f)

def _job_read(job_id: str) -> Optional[dict]:
    p = _job_path(job_id)
    if not os.path.exists(p):
        return None
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)

def _maybe_pause_if_stale(job: dict) -> dict:
    if job.get("state") == "running":
        last = float(job.get("updated_at", 0))
        if time.time() - last > HEARTBEAT_SECS:
            # Mark paused but keep last progress fields
            job = {**job, "state": "paused", "message": "Worker paused (stale heartbeat)"}
    return job

# ---------- Utils ----------
def _clean(x: Optional[str]) -> Optional[str]:
    if x is None:
        return None
    s = str(x).strip()
    if not s:
        return None
    return s.replace("/", "-").replace("\\", "-")

def _compose_instance_name(
    target_value: str,
    state: Optional[str] = "",
    county_name: Optional[str] = "",
    city_name: Optional[str] = "",
    cbsa_name: Optional[str] = "",
    ftype: str = "F",
) -> str:
    # Output (spec): [TARGET_VALUE]_[STATE]_[COUNTY]_[CITY]_[CBSA]_[TYPE].csv  (no "F_" prefix on output)
    parts = [
        _clean(target_value),
        _clean(state),
        _clean(county_name),
        _clean(city_name),
        _clean(cbsa_name),
        _clean(ftype or "F"),
    ]
    parts = [p for p in parts if p]
    return "_".join(parts)

def _load_series(
    db: str,
    target_value: str,
    state: Optional[str],
    county_name: Optional[str],
    city_name: Optional[str],
    cbsa_name: Optional[str],
    agg: str,
    ftype: str,
) -> pd.DataFrame:
    # 1) INPUT/F_[...]  2) INPUT/target.csv  3) synthesize demo
    inst = _compose_instance_name(target_value, state, county_name, city_name, cbsa_name, ftype)
    candidate = os.path.join(_INPUT_DIR, f"F_{inst}.csv")
    if os.path.exists(candidate):
        df = pd.read_csv(candidate)
    else:
        generic = os.path.join(_INPUT_DIR, "target.csv")
        if os.path.exists(generic):
            df = pd.read_csv(generic)
        else:
            dates = pd.date_range("2023-01-01", periods=360, freq="D")
            vals = (np.sin(np.arange(len(dates)) / 17.0) * 12 + 60 + np.random.randn(len(dates)) * 2).round(3)
            df = pd.DataFrame({"DATE": dates, "VALUE": vals})

    # Normalize to DATE, VALUE
    cols = {c.lower().strip(): c for c in df.columns}
    if "date" in cols and "value" in cols:
        df = df[[cols["date"], cols["value"]]].copy()
        df.columns = ["DATE", "VALUE"]
    elif "date local" in cols and "arithmetic mean" in cols:
        tmp = df[[cols["date local"], cols["arithmetic mean"]]].copy()
        tmp.columns = ["DATE", "VALUE"]
        tmp["DATE"] = pd.to_datetime(tmp["DATE"])
        df = tmp.groupby("DATE", as_index=False)["VALUE"].mean()
    else:
        up = [c.upper().replace(" ", "_") for c in df.columns]
        df.columns = up
        dcol = next((c for c in up if "DATE" in c), None)
        vcol = next((c for c in up if c in ("VALUE", "VALUES", "ARITHMETIC_MEAN", "MEAN")), None)
        if not (dcol and vcol):
            raise ValueError("Could not identify DATE/VALUE columns in input.")
        df = df[[dcol, vcol]].copy()
        df.columns = ["DATE", "VALUE"]

    df["DATE"] = pd.to_datetime(df["DATE"])
    df = df.sort_values("DATE").dropna(subset=["VALUE"])
    return df

def _monthly_quarterly(df: pd.DataFrame) -> Tuple[pd.Series, pd.Series]:
    m = df.set_index("DATE")["VALUE"].resample("MS").mean()
    q = df.set_index("DATE")["VALUE"].resample("QS").mean()
    return m, q

def _walk_forward(series: pd.Series, model_fn) -> pd.Series:
    preds = []
    vals = series.values
    idx = series.index
    for i in range(1, len(vals)):
        hist = series.iloc[:i]
        try:
            preds.append((idx[i], float(model_fn(hist))))
        except Exception:
            preds.append((idx[i], float(hist.iloc[-1])))
    return pd.Series({d: v for d, v in preds}, name="pred")

def _arima_auto(hist: pd.Series) -> float:
    try:
        import pmdarima as pm
        model = pm.auto_arima(hist, seasonal=False, suppress_warnings=True, error_action="ignore", stepwise=True)
        return float(model.predict(1)[0])
    except Exception:
        return float(hist.iloc[-1])

def _ses_mul(hist: pd.Series) -> float:
    try:
        from statsmodels.tsa.holtwinters import SimpleExpSmoothing
        fit = SimpleExpSmoothing(hist, initialization_method="estimated").fit(optimized=True)
        return float(fit.forecast(1)[0])
    except Exception:
        span = max(2, min(12, max(2, len(hist) // 2)))
        return float(hist.ewm(span=span, adjust=False).mean().iloc[-1])

def _hwes_add_damped(hist: pd.Series) -> float:
    try:
        from statsmodels.tsa.holtwinters import ExponentialSmoothing
        fit = ExponentialSmoothing(
            hist, trend="add", damped_trend=True, seasonal=None, initialization_method="estimated"
        ).fit(optimized=True)
        return float(fit.forecast(1)[0])
    except Exception:
        span = max(5, min(24, len(hist)))
        return float(hist.ewm(span=span, adjust=False).mean().iloc[-1])

def _gen_classical(df: pd.DataFrame) -> pd.DataFrame:
    # Output columns (spec)
    # DATE, VALUE, ARIMA-M, ARIMA-Q, SES-M, SES-Q, HWES-M, HWES-Q
    m, q = _monthly_quarterly(df)
    arima_m = _walk_forward(m, _arima_auto)
    arima_q = _walk_forward(q, _arima_auto)
    ses_m   = _walk_forward(m, _ses_mul)
    ses_q   = _walk_forward(q, _ses_mul)
    hwes_m  = _walk_forward(m, _hwes_add_damped)
    hwes_q  = _walk_forward(q, _hwes_add_damped)

    d = df[["DATE", "VALUE"]].copy()
    d["DATE"] = pd.to_datetime(d["DATE"])

    def fill_days(series_pred: pd.Series, freq: str) -> pd.Series:
        out = pd.Series(index=pd.to_datetime(d["DATE"].unique()), dtype="float64")
        for ts, val in series_pred.items():
            if freq == "MS":
                end = (ts + pd.offsets.MonthEnd(0)).normalize()
            else:
                end = (ts + pd.offsets.QuarterEnd(0)).normalize()
            rng = pd.date_range(ts, end, freq="D")
            out.loc[rng] = float(val)
        return out

    out = d.copy()
    out["ARIMA-M"] = fill_days(arima_m, "MS").reindex(out["DATE"]).values
    out["ARIMA-Q"] = fill_days(arima_q, "QS").reindex(out["DATE"]).values
    out["SES-M"]   = fill_days(ses_m,   "MS").reindex(out["DATE"]).values
    out["SES-Q"]   = fill_days(ses_q,   "QS").reindex(out["DATE"]).values
    out["HWES-M"]  = fill_days(hwes_m,  "MS").reindex(out["DATE"]).values
    out["HWES-Q"]  = fill_days(hwes_q,  "QS").reindex(out["DATE"]).values
    return out.ffill()

# ---------- Param helpers (accept both names; U.S. state is optional) ----------
def _coalesce_state(state: Optional[str], state_name: Optional[str]) -> str:
    return (state or state_name or "").strip()

# ================== Endpoints ==================

@router.get("/classical/probe")
def classical_probe(
    db: str,
    target_value: str,
    state: Optional[str] = Query(default=""),
    state_name: Optional[str] = Query(default=""),
    county_name: Optional[str] = Query(default=""),
    city_name: Optional[str] = Query(default=""),
    cbsa_name: Optional[str] = Query(default=""),
    agg: str = "mean",
    ftype: str = "F",
):
    _state = _coalesce_state(state, state_name)
    df = _load_series(db, target_value, _state, county_name, city_name, cbsa_name, agg, ftype)
    return {
        "rows": int(len(df)),
        "start_date": df["DATE"].min().strftime("%Y-%m-%d"),
        "end_date": df["DATE"].max().strftime("%Y-%m-%d"),
    }

@router.post("/classical/start")
async def classical_start(
    request: Request,
    db: Optional[str] = None,
    target_value: Optional[str] = None,
    state: Optional[str] = None,
    state_name: Optional[str] = None,
    county_name: Optional[str] = None,
    city_name: Optional[str] = None,
    cbsa_name: Optional[str] = None,
    agg: Optional[str] = "mean",
    ftype: Optional[str] = "F",
):
    """
    Accepts either JSON body or query params. Minimal required: db + target_value.
    """
    body = {}
    try:
        if request.headers.get("content-type", "").lower().startswith("application/json"):
            body = await request.json()
    except Exception:
        body = {}

    def pick(key: str, *aliases, default: Optional[str] = ""):
        for k in (key, *aliases):
            if k in body and body[k] is not None and str(body[k]).strip() != "":
                return str(body[k]).strip()
        return (locals().get(key) or default)

    _db     = pick("db", default="demo")              if db is None           else db
    _target = pick("target_value")                    if target_value is None else target_value
    _state  = (pick("state") or pick("state_name"))   if (state is None and state_name is None) else (state or state_name or "")
    _county = pick("county_name")                     if county_name is None  else county_name
    _city   = pick("city_name")                       if city_name is None    else city_name
    _cbsa   = pick("cbsa_name")                       if cbsa_name is None    else cbsa_name
    _agg    = pick("agg", default="mean")             if agg is None          else agg
    _ftype  = pick("ftype", default="F")              if ftype is None        else ftype

    if not _target:
        raise HTTPException(status_code=422, detail="target_value is required")

    job_id = str(uuid.uuid4())
    _job_write(job_id, {"state": "queued", "message": "Queued", "percent": 0, "done": 0, "total": 1})

    def _runner():
        try:
            _job_write(job_id, {"state": "running", "message": "Loading data…", "percent": 10, "done": 0, "total": 1})
            df = _load_series(_db, _target, _state, _county, _city, _cbsa, _agg, _ftype)
            _job_write(job_id, {"state": "running", "message": "Fitting classical models…", "percent": 60, "done": 0, "total": 1})
            out = _gen_classical(df)

            fname = _compose_instance_name(_target, _state, _county, _city, _cbsa, _ftype) + ".csv"
            out_path = os.path.join(_OUTPUT_DIR, fname)
            out.to_csv(out_path, index=False)

            _job_write(job_id, {"state": "ready", "message": "Completed", "percent": 100, "done": 1, "total": 1, "output_file": out_path})
        except Exception as e:
            _job_write(job_id, {"state": "error", "message": str(e), "percent": 0, "done": 0, "total": 1})

    threading.Thread(target=_runner, daemon=True).start()
    return {"job_id": job_id}

@router.post("/classical/resume")
def classical_resume(job_id: str = Query(...)):
    """
    If a job is 'paused' or its file exists but worker is gone, spawn a new worker to complete it.
    """
    job = _job_read(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    state = job.get("state")
    if state not in ("paused", "running", "queued"):
        return {"job_id": job_id, **job}  # nothing to do

    # If output already exists, flip to ready.
    out_path = job.get("output_file")
    if out_path and os.path.exists(out_path):
        _job_write(job_id, {**job, "state": "ready", "message": "Completed", "percent": 100, "done": 1, "total": 1})
        return {"job_id": job_id, **_job_read(job_id)}

    # We don't have partial checkpoints of modeling; re-run deterministic pipeline.
    # The parameters aren’t stored in the job file (lightweight approach). If needed,
    # you can extend job file to include params. For now just mark 'queued' and let client restart.
    _job_write(job_id, {**job, "state": "queued", "message": "Resuming…", "percent": 0, "done": 0, "total": 1})
    return {"job_id": job_id, **_job_read(job_id)}

@router.get("/classical/status")
def classical_status(job_id: str = Query(...)):
    job = _job_read(job_id)
    if not job:
        return {"state": "missing", "message": "Job not found", "done": 0, "total": 1, "percent": 0}
    job = _maybe_pause_if_stale(job)
    return {"job_id": job_id, **job}

@router.get("/classical/download")
def classical_download(job_id: str = Query(...)):
    job = _job_read(job_id)
    if not job or job.get("state") != "ready":
        raise HTTPException(status_code=409, detail="Job not ready")
    path = job.get("output_file")
    if not path or not os.path.exists(path):
        raise HTTPException(status_code=404, detail="Output not found")
    return FileResponse(path, filename=os.path.basename(path), media_type="text/csv")
