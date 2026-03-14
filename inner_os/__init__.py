"""Reusable inner-life operating system primitives."""

from .conscious_access import ConsciousAccessCore, ConsciousAccessSnapshot
from .hook_contracts import (  # noqa: F401
    MemoryRecallInput,
    PostTurnUpdateInput,
    PreTurnUpdateInput,
    ResponseGateInput,
)
from .integration_hooks import (  # noqa: F401
    HookState,
    IntegrationHooks,
    MemoryRecallResult,
    PostTurnUpdateResult,
    PreTurnUpdateResult,
    ResponseGateResult,
)
from .physiology import (
    BoundaryCore,
    HeartbeatConfig,
    HeartbeatCore,
    HeartbeatState,
    PainStressCore,
    RecoveryCore,
)
from .relational_world import RelationalWorldCore, RelationalWorldState
from .temporal import TemporalWeightCore, TemporalWeightState

from .memory_core import MemoryCore, MemorySearchHit
from .development_core import DevelopmentCore, DevelopmentState
from .reinterpretation_core import ReinterpretationCore, ReinterpretationSnapshot
from .environment_pressure_core import EnvironmentPressureCore, EnvironmentPressureSnapshot
from .relationship_core import RelationshipCore, RelationshipState
from .personality_core import PersonalityIndexCore, PersonalityIndexState
from .persistence_core import PersistenceCore, PersistenceState

from .terrain_core import AffectiveTerrainCore, TerrainSnapshot


from .simulation_transfer import SimulationEpisode, SimulationTransferCore, TransferredLearning
from .service import InnerOSService
from .http_contract import build_inner_os_manifest
from .http_router import router as http_router
from .http_app import app as http_app
from .memory_records import (
    BaseMemoryRecord,
    ObservedRealRecord,
    ReconstructedRecord,
    VerifiedRecord,
    ExperiencedSimRecord,
    TransferredLearningRecord,
    normalize_memory_record,
)
from .memory_bridge import collect_runtime_memory_candidates, memory_reference_to_record, observed_vision_to_record

from .schemas import (
    INNER_OS_HTTP_MANIFEST_SCHEMA,
    INNER_OS_MEMORY_RECORD_SCHEMA,
    INNER_OS_RECALL_PAYLOAD_SCHEMA,
    INNER_OS_PRE_TURN_INPUT_SCHEMA,
    INNER_OS_PRE_TURN_RESULT_SCHEMA,
    INNER_OS_MEMORY_RECALL_INPUT_SCHEMA,
    INNER_OS_MEMORY_RECALL_RESULT_SCHEMA,
    INNER_OS_RESPONSE_GATE_INPUT_SCHEMA,
    INNER_OS_RESPONSE_GATE_RESULT_SCHEMA,
    INNER_OS_POST_TURN_INPUT_SCHEMA,
    INNER_OS_POST_TURN_RESULT_SCHEMA,
    memory_record_contract,
    recall_payload_contract,
)
