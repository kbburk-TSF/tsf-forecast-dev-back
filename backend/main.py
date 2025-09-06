from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import text
from backend.database import engine

# Routers
from backend.routes.upload import router as upload_router
from backend.routes.data import router as data_router
from backend.routes.aggregate import router as agg_router
from backend.routes.forecast import router as forecast_router

app = FastAPI(title="TSF Backend", version="1.0.2")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include feature routes
app.include_router(upload_router)
app.include_router(data_router)
app.include_router(agg_router)
app.include_router(forecast_router)

@app.get("/health")
def health():
    """
    Reports API status + DB connectivity + table status + row count.
    Example:
      {"status":"ok","database":"up","schema":"ready","rows":123}
    """
    db_status = "down"
    schema_status = "missing"
    rows = 0
    try:
        with engine.begin() as conn:
            db_status = "up"
            exists = conn.execute(text("""
                SELECT EXISTS (
                  SELECT 1
                  FROM information_schema.tables
                  WHERE table_schema='public' AND table_name='air_quality_raw'
                )
            """)).scalar()
            if exists:
                schema_status = "ready"
                rows = int(conn.execute(text("SELECT COUNT(*) FROM public.air_quality_raw")).scalar() or 0)
            else:
                schema_status = "missing"
                rows = 0
    except Exception:
        db_status = "down"
        schema_status = "unknown"
        rows = 0

    return {"status": "ok", "database": db_status, "schema": schema_status, "rows": rows}

@app.get("/version")
def version():
    """
    Returns the backend version from the VERSION file.
    """
    v = "unknown"
    try:
        with open("VERSION", "r") as f:
            v = f.read().strip()
    except Exception:
        pass
    return {"version": v}
