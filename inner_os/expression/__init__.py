"""Expression bridge contracts."""

from .content_policy import derive_content_sequence, derive_content_skeleton, render_content_sequence
from .discourse_shape import DiscourseShape, coerce_discourse_shape, derive_discourse_shape
from .hint_bridge import (
    QUALIA_HINT_VERSION,
    build_expression_hints_from_gate_result,
)
from .interaction_constraints import InteractionConstraints, derive_interaction_constraints
from .models import DialogueContext, ResponsePlan
from .reaction_contract import (
    ReactionContract,
    coerce_reaction_contract,
    derive_reaction_contract,
)
from .repetition_guard import RepetitionGuard, derive_repetition_guard
from .response_planner import render_response
from .llm_bridge import ExpressionAdapter
from .llm_bridge_contract import (
    LLMBridgeContractReview,
    LLMBridgeContractViolation,
    review_llm_bridge_text,
)
from .llm_expression_bridge import (
    LLMExpressionBridgePolicy,
    LLMExpressionRequest,
    build_llm_expression_request,
)
from .speech_act_contract import (
    SPEECH_ACT_LABELS,
    SpeechActAnalysis,
    SpeechActSentence,
    build_speech_act_classification_request,
    speech_act_analysis_from_dict,
)
from .surface_expression_selector import (
    SurfaceExpressionCandidate,
    SurfaceExpressionProfile,
    build_surface_expression_candidates,
    choose_surface_expression,
)
from .surface_context_packet import (
    SurfaceContextPacket,
    build_surface_context_packet,
    coerce_surface_context_packet,
)
from .turn_delta import TurnDelta, derive_turn_delta

__all__ = [
    "DialogueContext",
    "ResponsePlan",
    "ExpressionAdapter",
    "LLMBridgeContractReview",
    "LLMBridgeContractViolation",
    "LLMExpressionBridgePolicy",
    "LLMExpressionRequest",
    "QUALIA_HINT_VERSION",
    "build_expression_hints_from_gate_result",
    "review_llm_bridge_text",
    "build_llm_expression_request",
    "SPEECH_ACT_LABELS",
    "SpeechActAnalysis",
    "SpeechActSentence",
    "build_speech_act_classification_request",
    "speech_act_analysis_from_dict",
    "render_response",
    "derive_content_skeleton",
    "derive_content_sequence",
    "render_content_sequence",
    "DiscourseShape",
    "coerce_discourse_shape",
    "derive_discourse_shape",
    "InteractionConstraints",
    "derive_interaction_constraints",
    "ReactionContract",
    "coerce_reaction_contract",
    "derive_reaction_contract",
    "RepetitionGuard",
    "derive_repetition_guard",
    "SurfaceExpressionCandidate",
    "SurfaceExpressionProfile",
    "build_surface_expression_candidates",
    "choose_surface_expression",
    "SurfaceContextPacket",
    "build_surface_context_packet",
    "coerce_surface_context_packet",
    "TurnDelta",
    "derive_turn_delta",
]
