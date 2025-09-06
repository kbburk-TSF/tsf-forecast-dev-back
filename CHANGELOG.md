# Changelog

## v1.2.3 — 2025-09-06
- Background jobs for Classical forecasts:
  - `POST /classical/start` → returns `job_id`
  - `GET /classical/status?job_id=…` → live progress (`state/done/total/message`)
  - `GET /classical/download?job_id=…` → CSV when ready
- Keeps existing `/classical/probe` and `/classical/export_*` endpoints.
