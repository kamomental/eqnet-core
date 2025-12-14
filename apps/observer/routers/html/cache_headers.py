from __future__ import annotations

from fastapi import Response


def disable_cache(response: Response) -> None:
    """Set headers that force browsers/CEF not to cache dynamic responses."""

    response.headers["Cache-Control"] = "no-store"
    response.headers["Pragma"] = "no-cache"
    response.headers["Expires"] = "0"


__all__ = ["disable_cache"]
