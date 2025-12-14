from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from .routers.api import audits as api_audits
from .routers.api import traces as api_traces
from .routers.html import pages as html_pages

BASE_DIR = Path(__file__).resolve().parent

app = FastAPI(title="EQNet Observer", version="0.1.0")

_templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))
html_pages.templates = _templates

app.mount("/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static")

app.include_router(api_audits.router, prefix="/api", tags=["api"])
app.include_router(api_traces.router, prefix="/api", tags=["api"])
app.include_router(html_pages.router, tags=["html"])


__all__ = ["app"]
