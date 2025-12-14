from __future__ import annotations

from .config import settings
from .services.observer_service import ObserverService
from .stores.audit_store import AuditStore
from .stores.trace_store import TraceStore

_audit_store = AuditStore(settings.audit_dir)
_trace_store = TraceStore(settings.trace_v1_dir)
_service = ObserverService(_audit_store, _trace_store)


def svc() -> ObserverService:
    return _service


__all__ = ["svc"]
