# Changelog

## v1.0.0 — 2025-09-06
- Initial backend release for Render + Neon
- CSV upload to `air_quality_raw`
- Daily aggregation endpoint (mean/sum)
- Lightweight forecasting endpoint (seasonal naive by DOW, EWMA)
- Health + Version endpoints
- No UNIQUE index (duplicates allowed)

## v1.0.1 — 2025-09-06
- Add `python-multipart` to requirements to enable file uploads on FastAPI
