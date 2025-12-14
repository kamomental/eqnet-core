from __future__ import annotations

from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

from ...config import settings
from ...deps import svc
from .cache_headers import disable_cache

router = APIRouter()
templates: Jinja2Templates | None = None


def _tmpl() -> Jinja2Templates:
    if templates is None:  # pragma: no cover - misconfiguration guard
        raise RuntimeError("templates engine is not bound")
    return templates


@router.get("/", response_class=HTMLResponse)
def dashboard(request: Request) -> HTMLResponse:
    audits = svc().list_audits()
    return _tmpl().TemplateResponse("pages/dashboard.html", {"request": request, "audits": audits})


@router.get("/overlay/latest", response_class=HTMLResponse)
def overlay_latest(request: Request) -> HTMLResponse:
    response = _tmpl().TemplateResponse(
        "pages/overlay_latest.html",
        {
            "request": request,
            "poll_s": settings.overlay_poll_seconds,
        },
    )
    disable_cache(response)
    return response


@router.get("/partials/overlay/latest", response_class=HTMLResponse, name="overlay_fragment_latest")
def overlay_fragment_latest(request: Request) -> HTMLResponse:
    model = svc().overlay_model_latest()
    response = _tmpl().TemplateResponse(
        "partials/overlay_fragment.html",
        {
            "request": request,
            "model": model,
        },
    )
    disable_cache(response)
    return response


__all__ = ["router", "templates"]
