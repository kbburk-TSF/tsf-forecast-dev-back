# =====================================================================
# File: backend/routes/forms_classical.py
# Version: v1.0.0 — 2025-09-20
# Purpose: Serve an HTML form that reads options from the DB and generates
#          a classical forecast CSV; saves to staging_historical and
#          streams the CSV back to the browser.
# =====================================================================

import csv
import os
import io
from datetime import date
from typing import Optional

from fastapi import APIRouter, Form, Request
from fastapi.responses import HTMLResponse, StreamingResponse, JSONResponse
from fastapi.templating import Jinja2Templates

import psycopg
from psycopg.rows import dict_row

TEMPLATES_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "templates")
templates = Jinja2Templates(directory=TEMPLATES_DIR)

STAGING_DIR = os.getenv("STAGING_HISTORICAL_DIR", os.path.join(os.getcwd(), "staging_historical"))
os.makedirs(STAGING_DIR, exist_ok=True)

DATABASE_URL = os.getenv("DATABASE_URL")

router = APIRouter()

def db_conn():
    if not DATABASE_URL:
        raise RuntimeError("DATABASE_URL is not set")
    return psycopg.connect(DATABASE_URL, row_factory=dict_row)

def list_parameters_and_states():
    params, states, date_min, date_max = [], [], None, None
    with db_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT DISTINCT parameter FROM demo_air_quality.air_quality_raw
                WHERE parameter IS NOT NULL
                ORDER BY parameter;
            """)
            params = [r["parameter"] for r in cur.fetchall()]
            cur.execute("""
                SELECT DISTINCT state_code FROM demo_air_quality.air_quality_raw
                WHERE state_code IS NOT NULL
                ORDER BY state_code;
            """)
            states = [r["state_code"] for r in cur.fetchall()]
            cur.execute("""
                SELECT MIN(date_local) AS dmin, MAX(date_local) AS dmax
                FROM demo_air_quality.air_quality_raw;
            """)
            row = cur.fetchone()
            if row:
                date_min = row["dmin"]
                date_max = row["dmax"]
    return params, states, date_min, date_max

def fetch_series_history(parameter: str, state_code: str, start_date: Optional[str], end_date: Optional[str]):
    sql = """
        SELECT date_local::date AS date, arithmetic_mean::float8 AS value
        FROM demo_air_quality.air_quality_raw
        WHERE parameter = %s
          AND state_code = %s
          AND (%s IS NULL OR date_local::date >= %s::date)
          AND (%s IS NULL OR date_local::date <= %s::date)
        ORDER BY date_local;
    """
    with db_conn() as conn, conn.cursor() as cur:
        cur.execute(sql, (parameter, state_code, start_date, start_date, end_date, end_date))
        return cur.fetchall()

def classical_forecast(series_rows, horizon: int = 30, method: str = "seasonal_naive_dow"):
    import datetime as dt
    from collections import defaultdict

    history = [{"date": r["date"], "value": float(r["value"])} for r in series_rows if r["value"] is not None]
    if not history:
        return []

    by_dow = defaultdict(list)
    last_date = history[-1]["date"]
    for r in history:
        by_dow[r["date"].weekday()].append(r["value"])

    def s_naive(next_date: date) -> float:
        dow = next_date.weekday()
        vals = by_dow.get(dow, None)
        if vals:
            return vals[-1]
        return history[-1]["value"]

    def ewma_forecast(alpha=0.3):
        s = history[0]["value"]
        for r in history[1:]:
            s = alpha * r["value"] + (1 - alpha) * s
        return s

    out = []
    for r in history:
        out.append({"date": r["date"], "value": r["value"], "type": "history"})
    for i in range(1, horizon + 1):
        next_d = last_date + dt.timedelta(days=i)
        if method == "ewma":
            fv = ewma_forecast()
        else:
            fv = s_naive(next_d)
        out.append({"date": next_d, "value": float(fv), "type": "forecast"})
    return out

def build_csv_bytes(rows):
    buf = io.StringIO()
    writer = csv.writer(buf, lineterminator="\n")
    writer.writerow(["date", "value", "type"])
    for r in rows:
        writer.writerow([r["date"], r["value"], r["type"]])
    data = buf.getvalue().encode("utf-8")
    buf.close()
    return data

@router.get("/classical", response_class=HTMLResponse)
def classical_form(request: Request):
    try:
        parameters, states, dmin, dmax = list_parameters_and_states()
    except Exception as e:
        return templates.TemplateResponse(
            "forms/classical.html",
            {
                "request": request,
                "parameters": [],
                "states": [],
                "date_min": None,
                "date_max": None,
                "error": str(e),
            },
            status_code=200,
        )

    return templates.TemplateResponse(
        "forms/classical.html",
        {
            "request": request,
            "parameters": parameters,
            "states": states,
            "date_min": dmin.isoformat() if dmin else None,
            "date_max": dmax.isoformat() if dmax else None,
            "error": None,
        },
    )

@router.post("/classical/generate")
def classical_generate(
    parameter: str = Form(...),
    state_code: str = Form(...),
    method: str = Form("seasonal_naive_dow"),
    horizon: int = Form(30),
    start_date: Optional[str] = Form(None),
    end_date: Optional[str] = Form(None),
):
    try:
        history = fetch_series_history(parameter, state_code, start_date, end_date)
        rows = classical_forecast(history, horizon=horizon, method=method)
        if not rows:
            return JSONResponse({"error": "No data found for selection."}, status_code=400)

        csv_bytes = build_csv_bytes(rows)

        safe_param = "".join(c for c in parameter if c.isalnum() or c in ("-", "_")).strip("_")
        safe_state = "".join(c for c in state_code if c.isalnum() or c in ("-", "_")).strip("_")
        filename = f"classical_{safe_param}_{safe_state}.csv"
        file_path = os.path.join(STAGING_DIR, filename)

        with open(file_path, "wb") as f:
            f.write(csv_bytes)

        return StreamingResponse(
            io.BytesIO(csv_bytes),
            media_type="text/csv",
            headers={"Content-Disposition": f'attachment; filename="{filename}"'},
        )
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)
