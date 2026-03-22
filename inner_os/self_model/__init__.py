"""Self / continuity layer contracts."""

from .models import PersonNode, PersonRegistry, SelfState
from .updater import person_registry_from_snapshot, update_self_state, update_person_registry
