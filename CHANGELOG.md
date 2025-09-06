# Changelog

## v1.2.1 — 2025-09-06
- Fix Render startup error `ModuleNotFoundError: No module named 'psycopg'` by:
  - Adding `psycopg[binary]` to requirements (psycopg3)
  - Hardening `backend/database.py` to normalize the DATABASE_URL and explicitly select psycopg (v3) or psycopg2

## v1.2.0 — 2025-09-06
- NEW classical endpoints: `/classical/export_target` and `/classical/export_classical`
- Spec-compliant filenames and columns; added statsmodels & pmdarima

## v1.0.9 — 2025-09-06
- Metadata endpoints: `/data/{db}/targets` and `/data/{db}/filters`

## v1.0.8 — 2025-09-06
- Align SQL with Neon’s live schema
