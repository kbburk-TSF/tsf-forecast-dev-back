# Debug routes to verify backend wiring at runtime.
from fastapi import APIRouter
import os

router = APIRouter(prefix="/debug", tags=["debug"])

def redact(url: str) -> str:
    if not url:
        return ""
    # hide password between ':' and '@'
    import re
    return re.sub(r":([^:@/]+)@", ":***@", url)

@router.get("/health")
def health():
    return {"ok": True}

@router.get("/env")
def env():
    return {
        "DATABASE_URL": redact(os.getenv("DATABASE_URL", "")),
        "ENGINE_DATABASE_URL": redact(os.getenv("ENGINE_DATABASE_URL", "")),
    }
