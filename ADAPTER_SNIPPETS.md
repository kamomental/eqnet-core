# Qualia Adapter Snippets

Use these paste-ready blocks to wire the qualia scaffolding into the current
EQNet code without waiting on deeper refactors. All names stay local so you can
swap modules later if needed.

------------------------------------------------------------
Section A. Query pressure hook (emot_terrain_lab/mind/inner_replay.py)
------------------------------------------------------------

```python
from eqnet_core.qualia import QueryEngine

self.qualia_query = getattr(self, "qualia_query", QueryEngine())
...
# inside the replay update step
epsilon_vec = replay_state.prediction_error      # existing tensor/array
precision_w = replay_state.precision_weights     # scalar / diag / matrix
u_t = self.qualia_query.compute(epsilon_vec, precision_w)

telemetry.log(
    {
        "qualia/u_t": float(u_t),
        "qualia/epsilon_norm": float(np.linalg.norm(epsilon_vec)),
    }
)
replay_state.query_pressure = u_t
```

------------------------------------------------------------
Section B. Meta divergence + gate (eqnet_core/models/conscious.py)
------------------------------------------------------------

```python
from eqnet_core.qualia import AccessGate, AccessGateConfig, meta_divergence

self.qualia_gate = getattr(self, "qualia_gate", AccessGate(AccessGateConfig()))
...
# during each conscious turn
u_t = workspace.replay.query_pressure
pred_sem = workspace.world_prediction
post_sem = workspace.workspace_state
m_t = meta_divergence(pred_sem, post_sem)
load_t = workspace.load_metric()   # scalar you already compute / expose

safety_flag = workspace.has_critical_alert()
result = self.qualia_gate.decide(
    u_t=u_t,
    m_t=m_t,
    load_t=load_t,
    override=safety_flag,
    reason="safety" if safety_flag else "normal",
)

telemetry.log(
    {
        "qualia/u_t": result["u_t"],
        "qualia/m_t": result["m_t"],
        "qualia/load": result["load_t"],
        "qualia/p_t": result["p_t"],
        "qualia/p_ema": result["p_ema"],
        "qualia/a_t": 1 if result["allow"] else 0,
        "qualia/logit": result["logit"],
        "qualia/theta": result["theta"],
        "qualia/reason": result["reason"],
    }
)

if result["allow"]:
    narrative = self._generate_narrative(turn_state)
else:
    narrative = None
    telemetry.log({"qualia/unconscious_success": 1})
```

------------------------------------------------------------
Section C. Nightly maintenance (controller job / scripts)
------------------------------------------------------------

```python
from eqnet_core.qualia import QualiaGraph

qualia_graph = getattr(self, "qualia_graph", QualiaGraph())
...
# nightly routine
graph_meta = qualia_graph.nightly_update(day_buffer)
access_rate = stats.mean(logger.series("qualia/a_t"))
new_theta = self.qualia_gate.nightly_retune(access_rate)

logger.write(
    "qualia/nightly",
    {
        "graph": graph_meta,
        "access_rate": float(access_rate),
        "theta": float(new_theta),
    },
)
```

------------------------------------------------------------
Section D. Minimal load metric helper (emot_terrain_lab/sim/mini_world.py)
------------------------------------------------------------

```python
def compute_qualia_load(self) -> float:
    active_intents = len(self.workspace.active_intents)
    token_fraction = self.workspace.token_usage / max(1, self.workspace.token_budget)
    return 0.5 * active_intents + 0.5 * token_fraction
```

Expose `compute_qualia_load()` on the workspace object so the conscious stack can
consume it for `L_t`.
