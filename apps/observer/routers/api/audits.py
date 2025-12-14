from __future__ import annotations

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse

from ...deps import svc

router = APIRouter()


@router.get("/audits")
def list_audits() -> list[dict]:
    return svc().list_audits()


@router.get("/audits/{date}")
def get_audit(date: str) -> FileResponse:
    path = svc().audit_file_path(date)
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"audit not found: {date}")
    return FileResponse(path, media_type="application/json")


__all__ = ["router"]
