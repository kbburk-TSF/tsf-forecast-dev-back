from typing import List, Optional
import os
from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse
import psycopg
from psycopg.rows import tuple_row

router = APIRouter()

DATABASE_URL = os.getenv("DATABASE_URL")
SCHEMA = "air_quality_demo_data"

TABLE_FQ = f'{SCHEMA}."air_quality_raw"'
COL_PARAM = '"Parameter Name"'
COL_STATE = '"State Name"'
COL_DATE  = '"Date Local"'

def get_conn():
    if not DATABASE_URL:
        raise RuntimeError("DATABASE_URL is not set")
    dsn = DATABASE_URL
    if "channel_binding=" in dsn:
        parts = []
        for tok in dsn.split():
            if not tok.lower().startswith("channel_binding="):
                parts.append(tok)
        dsn = " ".join(parts)
    conn = psycopg.connect(dsn, row_factory=tuple_row)
    with conn.cursor() as cur:
        cur.execute(f"SET search_path TO {SCHEMA}, public")
    return conn

def list_parameters() -> List[str]:
    sql = f'''\
        SELECT DISTINCT {COL_PARAM}
        FROM {TABLE_FQ}
        WHERE {COL_PARAM} IS NOT NULL
        ORDER BY {COL_PARAM}
    '''
    with get_conn() as conn, conn.cursor() as cur:
        cur.execute(sql)
        return [r[0] for r in cur.fetchall()]

def list_states_for_param(param: Optional[str]) -> List[str]:
    if param:
        sql = f'''\
            SELECT DISTINCT {COL_STATE}
            FROM {TABLE_FQ}
            WHERE {COL_PARAM} = %s AND {COL_STATE} IS NOT NULL
            ORDER BY {COL_STATE}
        '''
        args = (param,)
    else:
        sql = f'''\
            SELECT DISTINCT {COL_STATE}
            FROM {TABLE_FQ}
            WHERE {COL_STATE} IS NOT NULL
            ORDER BY {COL_STATE}
        '''
        args = ()
    with get_conn() as conn, conn.cursor() as cur:
        cur.execute(sql, args)
        return [r[0] for r in cur.fetchall()]

def esc(s: str) -> str:
    return (s.replace("&","&amp;").replace("<","&lt;").replace(">","&gt;")
            .replace('"','&quot;').replace("'","&#39;"))

def options(items: List[str], selected: Optional[str] = None) -> str:
    out = []
    for it in items:
        sel = " selected" if selected is not None and it == selected else ""
        out.append(f'<option value="{esc(it)}"{sel}>{esc(it)}</option>')
    return "\n".join(out)

HTML_SHELL = r"""<!doctype html>
<html>
<head>
  <meta charset="utf-8" />
  <title>Classical Forecast - Backend Form</title>
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <style>
    body { font-family: system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial, sans-serif; margin: 24px; }
    .wrap { max-width: 760px; margin: 0 auto; }
    form { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; }
    label { display: block; font-weight: 600; margin-bottom: 6px; }
    select { width: 100%; padding: 8px; border: 1px solid #ccc; border-radius: 6px; }
    .actions { grid-column: 1 / -1; display: flex; gap: 12px; }
    button { padding: 10px 16px; border: 0; border-radius: 6px; cursor: pointer; background: #1f6feb; color: #fff; }
    a.button { text-decoration: none; background: #eee; color: #111; padding: 10px 16px; border-radius: 6px; display: inline-block; }
  </style>
</head>
<body>
<div class="wrap">
  <h1>Classical Forecast - Backend Form</h1>

  <form method="post" action="/classical/start">
    <div>
      <label for="parameter">Target (Parameter Name)</label>
      <select id="parameter" name="parameter" required>
        {PARAM_OPTIONS}
      </select>
    </div>

    <div>
      <label for="state">State Name</label>
      <select id="state" name="state" required>
        {STATE_OPTIONS}
      </select>
    </div>

    <div class="actions">
      <button type="submit">Start Classical Job</button>
      <a class="button" href="/forms/classical">Reset</a>
    </div>
  </form>
</div>
</body>
</html>"""

def render_form(parameters: List[str], states: List[str], param_sel: Optional[str]) -> str:
    return HTML_SHELL.format(
        PARAM_OPTIONS=options(parameters, selected=param_sel),
        STATE_OPTIONS=options(states),
    )

@router.get("/classical", response_class=HTMLResponse)
def classical_form(request: Request, parameter: Optional[str] = None):
    try:
        params = list_parameters()
        states = list_states_for_param(parameter if parameter else None)
        return HTMLResponse(render_form(params, states, parameter), status_code=200)
    except Exception as e:
        return HTMLResponse(f"<pre>Database error: {e}</pre>", status_code=200)
