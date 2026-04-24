"""Self / continuity layer contracts."""

from .models import PersonNode, PersonRegistry, SelfState
from .self_other_attribution_state import (
    SelfOtherAttributionState,
    coerce_self_other_attribution_state,
    derive_self_other_attribution_state,
)
from .updater import person_registry_from_snapshot, update_self_state, update_person_registry
