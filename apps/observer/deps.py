from __future__ import annotations

from .config import settings
from .services.observer_service import ObserverService
from .stores.audit_store import AuditStore
from .stores.trace_store import TraceStore
from .stores.workspace_store import WorkspaceStore
from .services.workspace_service import WorkspaceService

_audit_store = AuditStore(settings.audit_dir)
_trace_store = TraceStore(settings.trace_v1_dir)
_workspace_store = WorkspaceStore(settings.trace_v1_dir)
_service = ObserverService(_audit_store, _trace_store)
_workspace_service = WorkspaceService(_workspace_store)


def svc() -> ObserverService:
    return _service


def workspace_svc() -> WorkspaceService:
    return _workspace_service


__all__ = ["svc", "workspace_svc"]
