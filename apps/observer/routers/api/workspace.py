from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query

from ...deps import workspace_svc

router = APIRouter()


@router.get("/workspace/{date}/{filename}")
def workspace_page(
    date: str,
    filename: str,
    *,
    offset: int = Query(0, ge=0),
    limit: int = Query(200, ge=1, le=2000),
    turn_id: str | None = None,
):
    try:
        data, next_offset = workspace_svc().page(
            date,
            filename,
            offset=offset,
            limit=limit,
            turn_id=turn_id,
        )
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return {"data": data, "next_offset": next_offset}


__all__ = ["router"]
