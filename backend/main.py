from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.routes.upload import router as upload_router
from backend.routes.data import router as data_router
from backend.routes.aggregate import router as agg_router
from backend.routes.forecast import router as forecast_router

app = FastAPI(title="TSF Backend", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(upload_router)
app.include_router(data_router)
app.include_router(agg_router)
app.include_router(forecast_router)

@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/version")
def version():
    try:
        with open("VERSION", "r") as f:
            v = f.read().strip()
    except Exception:
        v = "unknown"
    return {"version": v}
