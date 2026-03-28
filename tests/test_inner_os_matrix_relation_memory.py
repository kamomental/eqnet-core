import numpy as np

from inner_os.matrix_relation_memory import (
    MatrixMemoryHeadSpec,
    MatrixRelationMemoryConfig,
    MatrixRelationMemoryCore,
)


def _head_spec(
    *,
    name: str,
    key_projection: list[list[float]],
    value_projection: list[list[float]],
    query_projection: list[list[float]],
    forget_projection: list[float],
    forget_bias: float = 0.0,
    update_scale: float = 1.0,
    read_scale: float = 1.0,
) -> MatrixMemoryHeadSpec:
    return MatrixMemoryHeadSpec(
        name=name,
        key_projection=np.asarray(key_projection, dtype=np.float32),
        value_projection=np.asarray(value_projection, dtype=np.float32),
        query_projection=np.asarray(query_projection, dtype=np.float32),
        forget_projection=np.asarray(forget_projection, dtype=np.float32),
        forget_bias=forget_bias,
        update_scale=update_scale,
        read_scale=read_scale,
    )


def test_matrix_relation_memory_update_and_read_prefers_matching_head() -> None:
    config = MatrixRelationMemoryConfig(
        feature_names=("repair_trace", "bond_protection", "body_load"),
        head_specs=(
            _head_spec(
                name="repair",
                key_projection=[[1.0, 0.0, 0.0], [0.0, 0.5, 0.0]],
                value_projection=[[1.0, 0.0, 0.0], [0.0, 0.0, 0.5]],
                query_projection=[[1.0, 0.0, 0.0], [0.0, 0.5, 0.0]],
                forget_projection=[1.2, 0.0, -0.4],
                forget_bias=-0.2,
            ),
            _head_spec(
                name="body",
                key_projection=[[0.0, 0.0, 1.0], [0.0, 0.4, 0.0]],
                value_projection=[[0.0, 0.0, 1.0], [0.0, 0.6, 0.0]],
                query_projection=[[0.0, 0.0, 1.0], [0.0, 0.4, 0.0]],
                forget_projection=[-0.2, 0.0, 1.0],
                forget_bias=-0.1,
            ),
        ),
    )
    core = MatrixRelationMemoryCore(config)
    state = core.initialize()

    updated = core.update(
        state,
        features={
            "repair_trace": 0.9,
            "bond_protection": 0.4,
            "body_load": 0.1,
        },
    )
    readout = core.read(
        updated,
        query_features={
            "repair_trace": 1.0,
            "bond_protection": 0.2,
            "body_load": 0.0,
        },
    )

    assert updated.step_count == 1
    assert updated.head_state("repair").retain_gate > updated.head_state("body").retain_gate
    assert readout.dominant_head == "repair"
    assert readout.winner_margin > 0.0
    assert readout.head_scores["repair"] > readout.head_scores["body"]
    assert len(readout.combined_vector) == config.value_dim


def test_matrix_relation_memory_heads_can_hold_different_time_scales() -> None:
    config = MatrixRelationMemoryConfig(
        feature_names=("signal", "context"),
        head_specs=(
            _head_spec(
                name="slow",
                key_projection=[[1.0, 0.0], [0.0, 1.0]],
                value_projection=[[1.0, 0.0], [0.0, 1.0]],
                query_projection=[[1.0, 0.0], [0.0, 1.0]],
                forget_projection=[0.0, 0.0],
                forget_bias=2.4,
            ),
            _head_spec(
                name="fast",
                key_projection=[[1.0, 0.0], [0.0, 1.0]],
                value_projection=[[1.0, 0.0], [0.0, 1.0]],
                query_projection=[[1.0, 0.0], [0.0, 1.0]],
                forget_projection=[0.0, 0.0],
                forget_bias=-2.4,
            ),
        ),
    )
    core = MatrixRelationMemoryCore(config)
    initial = core.initialize()
    charged = core.update(initial, features={"signal": 1.0, "context": 0.5})
    decayed = core.update(charged, features={"signal": 0.0, "context": 0.0})

    slow_norm = float(np.linalg.norm(decayed.head_state("slow").matrix))
    fast_norm = float(np.linalg.norm(decayed.head_state("fast").matrix))

    assert charged.head_state("slow").retain_gate > charged.head_state("fast").retain_gate
    assert decayed.head_state("slow").retain_gate > decayed.head_state("fast").retain_gate
    assert slow_norm > fast_norm


def test_matrix_relation_memory_readout_changes_with_query_context() -> None:
    config = MatrixRelationMemoryConfig(
        feature_names=("bond", "insight", "risk"),
        head_specs=(
            _head_spec(
                name="bond",
                key_projection=[[1.0, 0.0, 0.0], [0.0, 0.8, 0.0]],
                value_projection=[[1.0, 0.0, 0.0], [0.0, 0.5, 0.0]],
                query_projection=[[1.0, 0.0, 0.0], [0.0, 0.8, 0.0]],
                forget_projection=[0.8, 0.2, -0.4],
            ),
            _head_spec(
                name="risk",
                key_projection=[[0.0, 0.0, 1.0], [0.0, 0.6, 0.0]],
                value_projection=[[0.0, 0.0, 1.0], [0.0, 0.3, 0.0]],
                query_projection=[[0.0, 0.0, 1.0], [0.0, 0.6, 0.0]],
                forget_projection=[-0.2, 0.1, 1.0],
            ),
        ),
    )
    core = MatrixRelationMemoryCore(config)
    state = core.initialize()
    state = core.update(
        state,
        features={
            "bond": 0.8,
            "insight": 0.7,
            "risk": 0.1,
        },
    )
    state = core.update(
        state,
        features={
            "bond": 0.1,
            "insight": 0.2,
            "risk": 0.9,
        },
    )

    bond_query = core.read(state, query_features={"bond": 1.0, "insight": 0.2, "risk": 0.0})
    risk_query = core.read(state, query_features={"bond": 0.0, "insight": 0.2, "risk": 1.0})

    assert bond_query.dominant_head == "bond"
    assert risk_query.dominant_head == "risk"
    assert bond_query.query_snapshot["risk"] == 0.0
    assert risk_query.query_snapshot["bond"] == 0.0
