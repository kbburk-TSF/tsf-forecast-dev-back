# =====================================================================
# File: backend/routes/forms_classical.py
# Version: v1.0.4 — 2025-09-20
# Changes:
# - v1.0.4: Use air_quality_demo_data.air_quality_raw (not demo_air_quality).
# - v1.0.3: DSN sanitization for channel_binding.
# - v1.0.2: Inline HTML (no Jinja).
# =====================================================================

import csv
import os
import io
from datetime import date
from typing import Optional, List
from urllib.parse import urlsplit, urlunsplit, parse_qsl, urlencode

from fastapi import APIRouter, Form, Request
from fastapi.responses import HTMLResponse, StreamingResponse, JSONResponse

import psycopg
from psycopg.rows import dict_row

STAGING_DIR = os.getenv("STAGING_HISTORICAL_DIR", os.path.join(os.getcwd(), "staging_historical"))
os.makedirs(STAGING_DIR, exist_ok=True)

DATABASE_URL = os.getenv("DATABASE_URL")

router = APIRouter()

def _sanitize_conninfo(dsn: str) -> str:
    if not dsn:
        return dsn
    if '://' in dsn:
        try:
            parts = urlsplit(dsn)
            q = dict(parse_qsl(parts.query, keep_blank_values=True))
            if 'channel_binding' in q:
                q.pop('channel_binding', None)
            new_query = urlencode(q, doseq=True)
            return urlunsplit((parts.scheme, parts.netloc, parts.path, new_query, parts.fragment))
        except Exception:
            pass
    tokens = []
    for tok in dsn.split():
        if not tok.lower().startswith('channel_binding='):
            tokens.append(tok)
    return ' '.join(tokens)

def db_conn():
    if not DATABASE_URL:
        raise RuntimeError("DATABASE_URL is not set")
    clean = _sanitize_conninfo(DATABASE_URL)
    return psycopg.connect(clean, row_factory=dict_row)

SCHEMA_TABLE = "air_quality_demo_data.air_quality_raw"

def list_parameters_and_states():
    params, states, date_min, date_max = [], [], None, None
    with db_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(f"""
                SELECT DISTINCT parameter FROM {SCHEMA_TABLE}
                WHERE parameter IS NOT NULL
                ORDER BY parameter;
            """)
            params = [r["parameter"] for r in cur.fetchall()]
            cur.execute(f"""
                SELECT DISTINCT state_code FROM {SCHEMA_TABLE}
                WHERE state_code IS NOT NULL
                ORDER BY state_code;
            """)
            states = [r["state_code"] for r in cur.fetchall()]
            cur.execute(f"""
                SELECT MIN(date_local) AS dmin, MAX(date_local) AS dmax
                FROM {SCHEMA_TABLE};
            """)
            row = cur.fetchone()
            if row:
                date_min = row["dmin"]
                date_max = row["dmax"]
    return params, states, date_min, date_max

def fetch_series_history(parameter: str, state_code: str, start_date: Optional[str], end_date: Optional[str]):
    sql = f"""
        SELECT date_local::date AS date, arithmetic_mean::float8 AS value
        FROM {SCHEMA_TABLE}
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
        fv = ewma_forecast() if method == "ewma" else s_naive(next_d)
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

def _options(opts: List[str]) -> str:
    return "\n".join([f'<option value="{o}">{o}</option>' for o in opts])

@router.get("/classical", response_class=HTMLResponse)
def classical_form(request: Request):
    try:
        parameters, states, dmin, dmax = list_parameters_and_states()
        date_min = dmin.isoformat() if dmin else ""
        date_max = dmax.isoformat() if dmax else ""
        html = f"""<!doctype html>
<html><head><meta charset='utf-8'><title>Classical Forecast — Backend Form</title>
<meta name='viewport' content='width=device-width, initial-scale=1' />
<style>
body {{ font-family: system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial, sans-serif; margin: 24px; }}
.wrap {{ max-width: 760px; margin: 0 auto; }}
form {{ display: grid; grid-template-columns: 1fr 1fr; gap: 16px; }}
label {{ display: block; font-weight: 600; margin-bottom: 6px; }}
input, select {{ width: 100%; padding: 8px; border: 1px solid #ccc; border-radius: 6px; }}
.full {{ grid-column: 1 / -1; }}
.actions {{ grid-column: 1 / -1; display: flex; gap: 12px; }}
button {{ padding: 10px 16px; border: 0; border-radius: 6px; cursor: pointer; }}
.primary {{ background: #1f6feb; color: #fff; }}
.muted {{ background: #eee; }}
</style></head>
<body><div class='wrap'>
<h1>Classical Forecast — Backend Form</h1>
<form method='post' action='/forms/classical/generate'>
  <div>
    <label for='parameter'>Parameter</label>
    <select id='parameter' name='parameter' required>
      {_options(parameters)}
    </select>
  </div>
  <div>
    <label for='state_code'>State</label>
    <select id='state_code' name='state_code' required>
      {_options(states)}
    </select>
  </div>
  <div>
    <label for='start_date'>Start date</label>
    <input type='date' id='start_date' name='start_date' min='{date_min}' max='{date_max}' />
  </div>
  <div>
    <label for='end_date'>End date</label>
    <input type='date' id='end_date' name='end_date' min='{date_min}' max='{date_max}' />
  </div>
  <div>
    <label for='method'>Method</label>
    <select id='method' name='method'>
      <option value='seasonal_naive_dow'>Seasonal Naive (DOW)</option>
      <option value='ewma'>EWMA</option>
    </select>
  </div>
  <div>
    <label for='horizon'>Horizon (days)</label>
    <input type='number' id='horizon' name='horizon' value='30' min='1' max='365' />
  </div>
  <div class='actions'>
    <button class='primary' type='submit'>Generate CSV</button>
    <a class='muted' href='/forms/classical'><button type='button' class='muted'>Reset</button></a>
  </div>
  <div class='full'>
    <p>On submit, the backend queries history, generates a classical forecast,
    saves a CSV into <code>staging_historical</code>, and downloads it.</p>
  </div>
</form>
</div></body></html>"""
        return HTMLResponse(content=html, status_code=200)
    except Exception as e:
        return HTMLResponse(content=f"<pre>Database error: {e}</pre>", status_code=200)

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
