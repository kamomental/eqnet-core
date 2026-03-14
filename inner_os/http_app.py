from __future__ import annotations

from fastapi import FastAPI

from .http_router import router

app = FastAPI(title="Inner OS API", version="0.1.0")
app.include_router(router, tags=["inner_os"])

__all__ = ["app"]
