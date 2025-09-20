from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.routes.data import router as data_router
from backend.routes.aggregate import router as agg_router
from backend.routes.forecast import router as forecast_router
from backend.routes.meta import router as meta_router
from backend.routes.classical import router as classical_router

app = FastAPI(title="TSF Backend", version="CLASSICAL-STABLE")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount only the canonical routers (no form)
app.include_router(data_router)
app.include_router(agg_router)
app.include_router(forecast_router)
app.include_router(meta_router)
app.include_router(classical_router)

# No root route is defined in canon; 404 at "/" is expected.
