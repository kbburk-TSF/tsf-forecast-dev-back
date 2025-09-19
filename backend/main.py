# CLEAN MAIN — 2025-09-19T17:59:41.350549
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from starlette.responses import JSONResponse

from backend.routes.data import router as data_router

app = FastAPI(title="TSF Backend (CLEAN)", version="2025-09-19T17:59:41.350549")

# Wide‑open CORS to unblock frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Simple request log
@app.middleware("http")
async def log_requests(request: Request, call_next):
    print(f"[REQ] {request.method} {request.url.path}?{request.url.query}")
    resp = await call_next(request)
    print(f"[RESP] {request.method} {request.url.path} -> {resp.status_code}")
    return resp

@app.get("/health")
def health():
    return {"ok": True, "service": "tsf-backend", "version": "2025-09-19T17:59:41.350549"}

# Wire routes
app.include_router(data_router)
