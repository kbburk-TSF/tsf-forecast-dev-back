# Changelog

## 2.0.3 – Fix probe/start param handling
- Make `state` **optional** and accept both `state` and `state_name` for `/classical/probe` and `/classical/start`.
- `/classical/start` now accepts **JSON body** or **query params** (both supported).
- Progress reporting unchanged (`state`, `message`, `percent`, `done`, `total`).
- CSV naming unchanged (drops `F_` on output): `[TARGET_VALUE]_[STATE]_[COUNTY]_[CITY]_[CBSA]_[TYPE].csv`.
- CSV columns unchanged (exactly 8 columns): `DATE, VALUE, ARIMA-M, ARIMA-Q, SES-M, SES-Q, HWES-M, HWES-Q`.

## 2.0.2
- (previous) Added acceptance of `state` alongside `state_name`.

## 2.0.1 / 2.0.0
- (previous) Hardening, CORS, and classical-only flow.



## v1.2.3 — 2025-09-06
- Background jobs for Classical forecasts:
  - `POST /classical/start` → returns `job_id`
  - `GET /classical/status?job_id=…` → live progress (`state/done/total/message`)
  - `GET /classical/download?job_id=…` → CSV when ready
- Keeps existing `/classical/probe` and `/classical/export_*` endpoints.
