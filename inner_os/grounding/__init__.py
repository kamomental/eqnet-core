"""Grounding layer contracts."""

from .models import Affordance, ObservationBundle, GroundedEntity, SymbolGrounding
from .observer import observe
from .affordance_engine import infer_affordances
from .symbol_grounding import ground_symbols, summarize_partner_grounding
