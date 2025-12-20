# Qualia Research Notes

A survey of recent academic work that treats qualia as structural, measurable phenomena, providing anchors for EQNet's virtual DNA and nightly design.

## 1. Programs & References

1. **Qualia Structure Study (Japan, 2023-2028)** - focuses on relational structure of qualia using psychophysics plus mathematical models. <https://qualia-structure.jp/intro/research/>
2. **Unconscious processing vs. conscious access** - Continuous Flash Suppression experiments show subjective brightness processed unconsciously (Consciousness & Cognition, 2025). <https://qualia-structure.jp/news/>
3. **Mathematical phenomenology** - qualia characterized via relational structures and enriched categories (Yoneda style) in JSPS project publications. <https://ci.nii.ac.jp/>
4. **Predictive error coding / query act** - Frontiers 2023 paper treats qualia as "query acts" driven by predictive coding. <https://www.frontiersin.org/articles/10.3389/fpsyg.2023.1109767>
5. **Modeler Schema Theory** - schema agent ignites conscious reports via qualia consistency checks (preprint). <https://arxiv.org/abs/2307.11111>
6. **Algebraic theory of qualia discrimination** - algebraic independence conditions for qualia spaces. <https://arxiv.org/abs/2401.00513>

## 2. Implementation Skeleton for EQNet

| Module | Research Anchor | Role in EQNet |
| --- | --- | --- |
| **QualiaGraph** | Qualia Structure / enriched category | Store prototypes `Q` and similarity matrix `D`; represent each qualia by relation profile `phi(q_i)=D[i,:]`. Nightly pass updates clusters ("qualia periodic table"). |
| **QueryEngine** | Predictive error coding | Compute `u_t = ||Pi_t epsilon_t||` (precision-weighted prediction errors). Use existing shadow/replay errors. |
| **AccessGate** | CFS experiments | Gate conscious reports via `logit = alpha*u_t + gamma*m_t - beta*L_t - theta_t`; allow shadow processing even when narrative is suppressed. |
| **MetaMonitor** | Modeler-Schema | Add meta inconsistency `m_t = Div(WorldModel_t, Post_t)` so schema breaks force access. |

### Steps to Integrate
1. **QualiaGraph component** (eqnet_core/qualia_graph.py placeholder): maintain prototypes, distances, nightly cluster metadata, and telemetry hooks.
2. **QueryEngine hook** (ShadowEstimator or EmotionVector pass): compute/log `u_t` each turn.
3. **AccessGate** (stage before Narrative/TalkMode): use `u_t`, `m_t`, load `L_t`, thresholds `theta_t`, log gating decisions, provide "unconscious success" logging.
4. **MetaMonitor**: reuse existing world-model vs. workspace differences to compute `m_t`; feed gate and telemetry.
5. **Nightly extensions**: update QualiaGraph clusters (distance smoothing or prototype refresh), retune gate thresholds (`theta`, precision) to maintain target access rates, report metrics (`u_t`, gate rate, unconscious success).
6. **Validation plan**: compare narrative frequency vs. internal recognition accuracy with gate on/off, monitor state-dissociation events, track next-day uncertainty recovery.

This keeps qualia relational and dynamic, aligns with current research, and grounds EQNet's control variables without claiming to generate raw qualia.

## 3. Model Core: States + Gate

Keeping the implementation agnostic to any specific package path, the qualia controller is specified by four coupled state holders plus a gate variable:

1. **QualiaGraph**: maintains prototypes `Q = {q_i}` learned from daily experience vectors and a symmetric distance matrix `D in R^{N x N}`. Each qualia is identified by its relational profile `phi(q_i) = D[i,:]`, so equality means sharing identical relation fingerprints rather than coordinates.
2. **QueryEngine**: exposes a scalar query pressure `u_t` derived from the current precision-weighted prediction error. Given an error vector `epsilon_t` (already available inside Shadow / InnerReplay) and a precision operator `Pi_t` (scalar, diagonal, or full matrix), define `u_t = ||Pi_t * epsilon_t||_2`, and log this per turn, optionally split into modality bands.
3. **MetaMonitor**: captures schema breaks as a divergence `m_t` between world-model predictions and the posterior workspace state. Depending on representations, choose KL/JS divergence (if probabilistic) or cosine/L2 (if embeddings). The contract is to publish a bounded scalar.
4. **AccessGate**: produces an access probability `p_t` and a binary allow flag `a_t` for narrative/talk components while allowing the internal processing chain to continue even when closed. The gate uses `logit_t = alpha*u_t + gamma*m_t - beta*L_t - theta_t` with `p_t = sigma(logit_t)`. Set `a_t = 1[p_t >= tau]` or sample Bernoulli for stochastic access. Safety overrides force `a_t=1` regardless of `p_t`.

## 4. EQNet Integration Blueprint (Path-Agnostic)

### 4.1 QualiaGraph Service
- **Inputs**: per-turn emotion / terrain vectors, nightly batches of the same, optional tags (context, agent, scenario).
- **Outputs**: prototype index per turn, relation vectors `phi(q_i)`, and metadata for telemetry (cluster ids, support counts).
- **Operations**: incremental `add_sample(x)`, smoothing update of `D` (EMA or keep prototypes fixed for a week and only refresh distances), nightly re-clustering to regenerate a "qualia periodic table" log.
- **Stability**: avoid sudden prototype drift by mixing old/new distances via `D_new = lambda*D_obs + (1-lambda)*D_old`.

### 4.2 QueryEngine Hook
- **Placement**: right after ShadowEstimator or InnerReplay produces `epsilon_t`. The hook only needs read-only access to existing error tensors.
- **Responsibility**: compute `u_t`, keep optional modality decomposition (extero/intero), and ship `(u_t, epsilon_norm, precision_tag)` into telemetry.
- **Implementation note**: when `Pi_t` is diagonal, multiply elementwise before the norm; fall back to a scalar gain when precision metadata is unavailable to keep the interface simple.

### 4.3 MetaMonitor
- **Placement**: world-model integration step where predictions are compared to workspace state. Provide a thin adapter so that whatever divergence is already computed for learning is re-used as `m_t`.
- **Outputs**: scalar `m_t` plus optional channel-specific divergences (schema, affect, control). Attach the same log record as QueryEngine for cross-correlation.

### 4.4 AccessGate (Narrative Front Door)
- **Inputs**: `(u_t, m_t, L_t, theta_t, safety_override_flag)`.
- **Outputs**: `(p_t, allow_access=a_t, theta_t_next)` and a gating log containing the upstream indices (`q_idx`, load summary).
- **Behavior**:
  - When `allow_access=False`, halt narrative/talk escalation but continue internal completion; tag logs as `unconscious_success` and funnel them to nightly consolidation.
  - Add hysteresis by storing a smoothed gate state `g_t = lambda*g_{t-1} + (1-lambda)*a_t` and only toggling when `g_t` crosses thresholds; this prevents chatter.
  - Provide a force-open path for safety-critical or high-severity diagnostics.

### 4.5 Load Metric `L_t`
Define once per workspace: options include number of unresolved intents, workspace token occupancy, or rolling count of overlapping query modules. Feed a normalized scalar to the gate to keep coefficients interpretable.

## 5. Nightly Homeostasis Loop

1. **QualiaGraph Refresh**
   - Gather the day's experience vectors, run prototype update (k-means / spectral clustering), recompute `D`, and emit metadata (cluster size, exemplar ids) into logs.
   - Apply exponential smoothing when swapping `D` to keep `phi(q_i)` stable.
2. **Gate Retuning**
   - Track average access rate `p_bar` vs. target `p_star`. Update `theta` via `theta_{t+1} = theta_t + eta*(p_bar - p_star)` so excessive talking raises the threshold and vice versa.
   - Optionally adjust precision scaling `Pi` using nightly stats (for example, whiten errors by their empirical covariance).
3. **Log Consolidation**
   - Merge per-turn telemetry into a nightly report: distributions of `u_t, m_t, p_t`, unconscious-success count, forced overrides, and prototype churn.
   - Feed unconscious successes back into skill consolidation pipelines so "silent" wins still improve policy weights.

## 6. Turn Loop Skeleton (Pseudo)
```
x = sense_emotion_vector()                 # structural record
q_idx = qualia_graph.add_sample(x)

epsilon = fetch_prediction_error()         # from Shadow/Replay
Pi = precision_weights()                   # scalar/diag/matrix
nu = query_engine.compute(epsilon, Pi)

pred_sem = world_model_prediction()
post_sem = workspace_semantics()
m = meta_monitor.compute(pred_sem, post_sem)

L = workspace_load()

decision = access_gate.decide(u=nu, m=m, load=L,
                              theta=current_theta,
                              safety_override=is_critical())
log_turn({"u":nu, "m":m, "p":decision.p,
          "allow":decision.allow_access,
          "q_idx":q_idx, "load":L,
          "theta":decision.theta})

if decision.allow_access:
    narrative = generate_narrative()
else:
    narrative = None
    log_unconscious_success()
```
Paths/classes are placeholders so this can slot into whichever EQNet package hosts the relevant responsibilities.

## 7. Validation & Metrics

- **Narrative Rate**: proportion of turns with `allow_access=True`. Expect a drop once the gate activates.
- **Unconscious Success Rate**: internal task completions with `allow_access=False`; should remain high if processing is intact.
- **Misfire Rate**: count cases where `u_t` and `m_t` are high yet the gate stayed closed (or forced open). Monitor as `state_dissociation`.
- **Next-Day Recovery**: measure reduction of average `u_t` after nightly tuning to confirm the glial/detox analogy.
- **A/B Toggle**: run short windows with the gate disabled to verify narrative frequency spikes while recognition accuracy stays similar, mirroring CFS studies.

These metrics make the qualia controller falsifiable: if EQNet's internal success rate drops when narratives are gated, revisit `Pi_t`, `L_t`, or threshold dynamics before touching philosophical claims.

## 8. Implementation Touch Points (Current Repo)

1. **Experience capture** (`eqnet_core/models/emotion.py`): expose the emotion or terrain embedding already computed per turn, feed it into `QualiaGraph.add_sample`, and store the returned prototype index on the turn record so later modules can join logs.
2. **Prediction error extraction** (`emot_terrain_lab/mind/inner_replay.py`): pull the existing Shadow/Replay residuals, pass them through the lightweight `QueryEngine` helper (see `eqnet_qualia_skeleton` for a NumPy-only reference implementation), and push `(u_t, epsilon_norm, precision_tag)` to telemetry.
3. **Meta divergence adapter** (`eqnet_core/models/conscious.py` or whichever module already compares world-model vs. workspace): wrap the current divergence calculation behind a `MetaMonitor.compute(pred, post)` interface so the gate can subscribe without duplicating math.
4. **Narrative interception** (`eqnet_core/models/conscious.py`): before generating narrative/talk actions, call `AccessGate.decide` with `u_t`, `m_t`, load `L_t`, and the latest `theta`. When the gate closes, skip generation, emit an `unconscious_success` log entry, and cache any internal outputs for nightly consolidation.
5. **Load metric plumbing** (`emot_terrain_lab/sim/mini_world.py` and scheduler utilities): publish a scalar load estimate each control tick (for example, count of concurrent workspace intents or buffer occupancy) and feed it into AccessGate along with any emergency override flags.
6. **Nightly driver** (`scripts` or `control` jobs): schedule a nightly task that calls `QualiaGraph.refresh(day_buffer)` and `AccessGate.retune(theta_stats)` so prototype drift and gate thresholds stay bounded. Store nightly summaries under `logs/qualia/` (or similar) for inspection.

Following these steps keeps the documentation aligned with the live codebase: every module listed above already exists in the repository, so wiring the skeleton components only requires small adapters instead of invasive rewrites.
