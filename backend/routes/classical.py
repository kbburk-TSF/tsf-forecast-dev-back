# Classical route placeholder; CSV generation + staging upload happens in your existing implementation.
from fastapi import APIRouter

router = APIRouter(prefix="/classical", tags=["classical"])

@router.get("/probe")
def probe():
    return {"ok": True}
