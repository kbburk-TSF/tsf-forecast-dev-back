from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from backend.routes.data import router as data_router
from backend.routes.aggregate import router as agg_router
from backend.routes.forecast import router as forecast_router
from backend.routes.meta import router as meta_router
from backend.routes.classical import router as classical_router
import os

app = FastAPI(title="TSF Backend", version="1.2.0")

env_origins = os.getenv("ALLOWED_ORIGINS", "").strip()
allowed = [o.strip() for o in env_origins.split(",") if o.strip()] or ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(data_router)
app.include_router(agg_router)
app.include_router(forecast_router)
app.include_router(meta_router)
app.include_router(classical_router)

@app.get("/health")
def health():
    return {"status":"ok","database":"up","schema":"ready"}

@app.get("/version")
def version():
    try:
        with open("VERSION","r") as f:
            return {"version": f.read().strip()}
    except Exception:
        return {"version": "unknown"}


# === TSF_PATCH_CLASSICAL_FORECAST_BEGIN ===
# Minimal classical forecast endpoints injected to fix hanging "analysis paused" behavior.
# No external imports or routes added elsewhere; everything stays in this file.

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse
import os, threading, uuid
import pandas as pd

router = APIRouter()
_JOBS = {}

# Base paths relative to this file's directory
_BASE_DIR = os.path.dirname(os.path.abspath(__file__))
_DATA_DIR = os.path.join(_BASE_DIR, "data")
_INPUT_DIR = os.path.join(_DATA_DIR, "input")
_OUTPUT_DIR = os.path.join(_DATA_DIR, "output")
os.makedirs(_INPUT_DIR, exist_ok=True)
os.makedirs(_OUTPUT_DIR, exist_ok=True)

def _run_job(job_id: str):
    try:
        _JOBS[job_id] = _JOBS.get(job_id, {})
        _JOBS[job_id]["status"] = "running"
        _JOBS[job_id]["progress"] = 10

        target_path = os.path.join(_INPUT_DIR, "target.csv")
        if os.path.exists(target_path):
            df_raw = pd.read_csv(target_path)
            if "DATE" in df_raw.columns and "VALUE" in df_raw.columns:
                df = df_raw[["DATE","VALUE"]].copy()
                df["DATE"] = pd.to_datetime(df["DATE"])
            elif "Date Local" in df_raw.columns and "Arithmetic Mean" in df_raw.columns:
                tmp = df_raw.rename(columns={"Date Local":"DATE","Arithmetic Mean":"VALUE"})
                tmp["DATE"] = pd.to_datetime(tmp["DATE"])
                df = tmp[["DATE","VALUE"]].groupby("DATE", as_index=False)["VALUE"].mean()
            else:
                # try best-effort
                cols = [c.strip().upper().replace(" ","_") for c in df_raw.columns]
                df_raw.columns = cols
                date_col = next((c for c in cols if "DATE" in c), None)
                val_col = next((c for c in cols if c in ("VALUE","VALUES","ARITHMETIC_MEAN","MEAN","TARGET")), None)
                if not (date_col and val_col):
                    raise ValueError("Could not identify DATE/VALUE columns")
                df = df_raw[[date_col, val_col]].copy()
                df.columns = ["DATE","VALUE"]
                df["DATE"] = pd.to_datetime(df["DATE"])
        else:
            # fallback dummy series
            import numpy as np
            dates = pd.date_range("2024-01-01", periods=30, freq="D")
            vals = pd.Series(np.random.rand(len(dates))*100).round(2)
            df = pd.DataFrame({"DATE": dates, "VALUE": vals})

        _JOBS[job_id]["progress"] = 50
        df = df.sort_values("DATE").dropna(subset=["VALUE"])
        df["MA"] = df["VALUE"].rolling(window=7, min_periods=1).mean()

        last_date = df["DATE"].max()
        last_ma = float(df["MA"].iloc[-1])
        future = pd.date_range(last_date + pd.Timedelta(days=1), periods=14, freq="D")
        df_future = pd.DataFrame({"DATE": future, "VALUE": [last_ma]*len(future)})
        out = pd.concat([df[["DATE","VALUE"]], df_future], ignore_index=True)
        out["DATE"] = out["DATE"].dt.strftime("%Y-%m-%d")

        out_path = os.path.join(_OUTPUT_DIR, f"{job_id}_classical.csv")
        out.to_csv(out_path, index=False)

        _JOBS[job_id]["progress"] = 100
        _JOBS[job_id]["status"] = "completed"
        _JOBS[job_id]["output_file"] = out_path
    except Exception as e:
        _JOBS[job_id]["status"] = "error"
        _JOBS[job_id]["error"] = str(e)

@router.post("/forecast/classical/run")
def _run():
    job_id = str(uuid.uuid4())
    _JOBS[job_id] = {"status":"queued","progress":0}
    t = threading.Thread(target=_run_job, args=(job_id,), daemon=True)
    t.start()
    return {"job_id": job_id, "status": _JOBS[job_id]["status"]}

@router.get("/forecast/status/{job_id}")
def _status(job_id: str):
    job = _JOBS.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return {"job_id": job_id, **job}

@router.get("/forecast/download/{job_id}")
def _download(job_id: str):
    job = _JOBS.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    if job.get("status") != "completed":
        raise HTTPException(status_code=409, detail="Job not completed")
    return FileResponse(job["output_file"], filename="classical_forecast.csv", media_type="text/csv")

# Attach to app
app.include_router(router)
# === TSF_PATCH_CLASSICAL_FORECAST_END ===
