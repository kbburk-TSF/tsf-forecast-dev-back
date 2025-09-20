from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.routes.data import router as data_router
from backend.routes.aggregate import router as agg_router
from backend.routes.forecast import router as forecast_router
from backend.routes.meta import router as meta_router
from backend.routes.classical import router as classical_router
from backend.routes.forms_classical import router as forms_classical_router

app = FastAPI(title="TSF Backend", version="CLASSICAL-STABLE+schema-fix")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(data_router)
app.include_router(agg_router)
app.include_router(forecast_router)
app.include_router(meta_router)
app.include_router(classical_router)

app.include_router(forms_classical_router, prefix="/forms", tags=["forms"])
