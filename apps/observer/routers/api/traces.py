from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query

from ...deps import svc

router = APIRouter()


@router.get("/traces/{date}")
def list_traces(date: str) -> list[dict]:
    return svc().list_traces(date)


@router.get("/traces/{date}/{filename}")
def trace_page(
    date: str,
    filename: str,
    *,
    offset: int = Query(0, ge=0),
    limit: int = Query(200, ge=1, le=2000),
    turn_id: str | None = None,
    turn_id_contains: str | None = None,
) -> dict[str, object]:
    try:
        rows, next_offset = svc().read_trace_page(
            date,
            filename,
            offset=offset,
            limit=limit,
            turn_id=turn_id,
            turn_id_contains=turn_id_contains,
        )
    except FileNotFoundError as exc:  # pragma: no cover - fastapi converts to 404
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return {"data": rows, "next_offset": next_offset}


__all__ = ["router"]
