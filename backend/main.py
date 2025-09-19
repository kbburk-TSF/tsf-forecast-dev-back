# BUILD: 2025-09-19T17:30:23.672847 pinpoint diagnostics
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from starlette.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

try:
    from backend.version import BUILD_ID
except Exception:
    BUILD_ID = "2025-09-19T17:30:23.672847"

from backend.routes.data import router as data_router

app = FastAPI(title="TSF Backend", version=BUILD_ID)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Simple request logger
@app.middleware("http")
async def log_requests(request: Request, call_next):
    print(f"[REQ] {request.method} {request.url.path} qs={request.url.query}")
    resp = await call_next(request)
    print(f"[RESP] {request.method} {request.url.path} -> {resp.status_code}")
    return resp

# Include routers
app.include_router(data_router)

# 404 handler that reveals the path and registered routes
@app.exception_handler(StarletteHTTPException)
async def http_exception_handler(request: Request, exc: StarletteHTTPException):
    if exc.status_code == 404:
        routes = [r.path for r in app.router.routes if hasattr(r, "path")]
        return JSONResponse(
            status_code=404,
            content={"detail": "Not Found", "path": request.url.path, "qs": request.url.query, "routes": routes},
        )
    return JSONResponse(status_code=exc.status_code, content={"detail": str(exc.detail)})

@app.get("/health")
def health():
    return {"ok": True, "version": BUILD_ID}
