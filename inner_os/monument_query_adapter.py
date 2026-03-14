from __future__ import annotations

from typing import Any, Mapping, Optional, Tuple


class MonumentQueryAdapter:
    def query(self, relational_world: Mapping[str, Any]) -> Tuple[float, Optional[str]]:
        try:
            from eqnet.culture_model import CultureContext, query_matching_monuments
        except Exception:
            return 0.0, None
        ctx = CultureContext(
            culture_tag=str(relational_world.get("culture_id") or "") or None,
            place_id=str(relational_world.get("place_memory_anchor") or relational_world.get("zone_id") or "") or None,
            partner_id=str(relational_world.get("community_id") or "") or None,
            object_id=None,
            object_role=str(relational_world.get("social_role") or "") or None,
            activity_tag=None,
        )
        monuments = query_matching_monuments(context=ctx, top_k=3)
        if not monuments:
            return 0.0, None
        top = monuments[0]
        return float(getattr(top, "salience", 0.0) or 0.0), (str(getattr(top, "kind", "") or "") or None)
