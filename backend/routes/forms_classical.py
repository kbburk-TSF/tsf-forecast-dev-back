# =====================================================================
# File: backend/routes/forms_classical.py
# Version: v1.1.0 — 2025-09-20
# Purpose: Serve an HTML form (no Jinja) with DB‑backed drop-downs:
#   - Target => "Parameter Name"
#   - Filter => "State Name"
# Aggregates mean of "Arithmetic Mean" by "Date Local", writes CSV to
# staging_historical, and streams the file back to the browser.
# =====================================================================

import csv
import os
import io
from typing import Optional, List
from urllib.parse import urlsplit, urlunsplit, parse_qsl, urlencode

from fastapi import APIRouter, Form, Request
from fastapi.responses import HTMLResponse, StreamingResponse, JSONResponse

import psycopg
from psycopg.rows import dict_row

# Where to save CSVs to trigger the pipeline
STAGING_DIR = os.getenv("STAGING_HISTORICAL_DIR", os.path.join(os.getcwd(), "staging_historical"))
os.makedirs(STAGING_DIR, exist_ok=True)

DATABASE_URL = os.getenv("DATABASE_URL")

router = APIRouter()

# ---- Connection helpers ------------------------------------------------

def _sanitize_conninfo(dsn: str) -> str:
    """Remove 'channel_binding' if present (URL or DSN form)."""
    if not dsn:
        return dsn
    if '://' in dsn:
        try:
            parts = urlsplit(dsn)
            q = dict(parse_qsl(parts.query, keep_blank_values=True))
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

TABLE = "air_quality_raw"
COL_PARAM = '"Parameter Name"'
COL_STATE = '"State Name"'
COL_DATE  = '"Date Local"'
COL_VALUE = '"Arithmetic Mean"'

# ---- DB queries --------------------------------------------------------

def list_parameters() -> List[str]:
    sql = f"""
        SELECT DISTINCT {COL_PARAM} AS p
        FROM {TABLE}
        WHERE {COL_PARAM} IS NOT NULL
        ORDER BY p
    """
    with db_conn() as conn, conn.cursor() as cur:
        cur.execute(sql)
        return [r["p"] for r in cur.fetchall()]

def list_states_for_param(param: Optional[str]) -> List[str]:
    if not param:
        sql = f"""
            SELECT DISTINCT {COL_STATE} AS s
            FROM {TABLE}
            WHERE {COL_STATE} IS NOT NULL
            ORDER BY s
        """
        args = ()
    else:
        sql = f"""
            SELECT DISTINCT {COL_STATE} AS s
            FROM {TABLE}
            WHERE {COL_PARAM} = %s AND {COL_STATE} IS NOT NULL
            ORDER BY s
        """
        args = (param,)
    with db_conn() as conn, conn.cursor() as cur:
        cur.execute(sql, args)
        return [r["s"] for r in cur.fetchall()]

def date_bounds() -> Optional[dict]:
    sql = f"""
        SELECT MIN({COL_DATE})::date AS dmin, MAX({COL_DATE})::date AS dmax
        FROM {TABLE}
    """
    with db_conn() as conn, conn.cursor() as cur:
        cur.execute(sql)
        row = cur.fetchone()
        if row:
            return {"min": row["dmin"], "max": row["dmax"]}
        return None

def aggregate_series(param: str, state: str):
    sql = f"""
        SELECT
          {COL_DATE}::date AS date,
          AVG({COL_VALUE})::float8 AS value
        FROM {TABLE}
        WHERE {COL_PARAM} = %s
          AND {COL_STATE} = %s
        GROUP BY {COL_DATE}
        ORDER BY {COL_DATE}
    """
    with db_conn() as conn, conn.cursor() as cur:
        cur.execute(sql, (param, state))
        return cur.fetchall()

# ---- HTML helpers ------------------------------------------------------

def _esc_html(s: str) -> str:
    return (s.replace("&","&amp;").replace("<","&lt;").replace(">","&gt;")
             .replace('"',"&quot;").replace("'","&#39;"))

def _options(opts: List[str], selected: Optional[str] = None) -> str:
    out = []
    for o in opts:
        sel = " selected" if selected is not None and o == selected else ""
        out.append(f"<option value="{_esc_html(o)}"{sel}>{_esc_html(o)}</option>")
    return "\n".join(out)

def _page_html(parameters: List[str], states: List[str], dmin: Optional[str], dmax: Optional[str], param_sel: Optional[str]):
    min_attr = dmin or ""
    max_attr = dmax or ""
    return f"""<!doctype html>
<html><head><meta charset='utf-8'><title>Classical Forecast — Backend Form</title>
<meta name='viewport' content='width=device-width, initial-scale=1' />
<style>
body {{ font-family: system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial, sans-serif; margin: 24px; }}
.wrap {{ max-width: 760px; margin: 0 auto; }}
form {{ display: grid; grid-template-columns: 1fr 1fr; gap: 16px; }}
label {{ display: block; font-weight: 600; margin-bottom: 6px; }}
select, input {{ width: 100%; padding: 8px; border: 1px solid #ccc; border-radius: 6px; }}
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
      <label for='param'>Target (Parameter Name)</label>
      <select id='param' name='param' required>
        {_options(parameters, selected=param_sel)}
      </select>
    </div>

    <div>
      <label for='state'>State Name</label>
      <select id='state' name='state' required>
        {_options(states)}
      </select>
    </div>

    <div class='full'>
      <small>Date range available: <strong>{_esc_html(min_attr)}</strong> to <strong>{_esc_html(max_attr)}</strong> (from "{_esc_html(COL_DATE.strip('"'))}")</small>
    </div>

    <div class='actions'>
      <button class='primary' type='submit'>Generate CSV</button>
      <a class='muted' href='/forms/classical'><button type='button' class='muted'>Reset</button></a>
    </div>
  </form>
</div></body></html>"""

def _csv_bytes(rows: List[dict]) -> bytes:
    buf = io.StringIO()
    w = csv.writer(buf, lineterminator="\n")
    w.writerow(["date","value"])
    for r in rows:
        w.writerow([r["date"], r["value"]])
    data = buf.getvalue().encode("utf-8")
    buf.close()
    return data

# ---- Routes ------------------------------------------------------------

@router.get("/classical", response_class=HTMLResponse)
def classical_form(request: Request, param: Optional[str] = None):
    try:
        parameters = list_parameters()
        # State list depends on selected parameter (if provided)
        states = list_states_for_param(param if param else None)
        bounds = date_bounds() or {}
        dmin = bounds.get("min").isoformat() if bounds.get("min") else ""
        dmax = bounds.get("max").isoformat() if bounds.get("max") else ""
        return HTMLResponse(content=_page_html(parameters, states, dmin, dmax, param_sel=param), status_code=200)
    except Exception as e:
        return HTMLResponse(content=f"<pre>Database error: {e}</pre>", status_code=200)

@router.post("/classical/generate")
def classical_generate(
    param: str = Form(...),
    state: str = Form(...),
):
    try:
        rows = aggregate_series(param, state)
        if not rows:
            return JSONResponse({"error": "No data found for that selection."}, status_code=404)

        csv_bytes = _csv_bytes(rows)

        # Filename based on selection (kept consistent with prior behavior)
        def _safe(s: str) -> str:
            return "".join(c for c in s if c.isalnum() or c in ("-", "_")).strip("_")
        filename = f"classical_{_safe(param)}_{_safe(state)}.csv"
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
