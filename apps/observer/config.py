from __future__ import annotations

from pathlib import Path

try:
    from pydantic_settings import BaseSettings, SettingsConfigDict
except Exception:  # pragma: no cover - fallback for pydantic v1
    from pydantic import BaseSettings  # type: ignore
    SettingsConfigDict = None  # type: ignore


class ObserverSettings(BaseSettings):
    audit_dir: Path = Path("telemetry/audit")
    trace_v1_dir: Path = Path("telemetry/trace_v1")
    diff_dir: Path = Path("reports/compare_loops")

    page_size: int = 200
    max_page_size: int = 2000
    overlay_poll_seconds: int = 3

    if SettingsConfigDict is not None:
        model_config = SettingsConfigDict(
            env_prefix="EQNET_",
            env_file=".env",
            extra="ignore",
        )
    else:
        class Config:  # type: ignore[no-redef]
            env_prefix = "EQNET_"
            env_file = ".env"


settings = ObserverSettings()
