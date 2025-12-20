# EQNet Core | Virtual DNA Specification

**Purpose**

Provide a biologically-inspired specification for EQNet’s "virtual DNA": the intrinsic response patterns and update rules that persist across worlds, while keeping culture and world priors swappable. The DNA is split into three layers:

1. **Core DNA (fixed traits / learning heuristics)**
2. **Epigenetic Overlay (adaptive modulation)**
3. **World & Culture Priors (environment profiles)**

---

## 1. Core DNA (nearly immutable)
Represents base dynamics analogous to temperament or physiological tendencies.

### 1.1 Temporal Dynamics
| Parameter | Description |
|-----------|-------------|
| `react_latency` | Time to initiate a response (s). Lower = reflexive. |
| `recovery_half_life` | Time for stress/affect to decay by half. |
| `memory_consolidation_rate` | Speed at which short-term events become stable memory traces. |
| `forgetting_rate` | Baseline decay of unused representations. |

### 1.2 Predictive / Epistemic Precision
| Parameter | Description |
|-----------|-------------|
| `uncertainty_tolerance` | How long unresolved ambiguity can be tolerated before acting. |
| `exploration_bias` | Bias toward novel exploration vs. exploitation. |
| `ask_vs_act_ratio` | Probability of requesting clarification before acting. |
| `model_update_aggression` | Weight given to prediction errors when updating internal models. |
| `sensory_precision_bias` | Baseline trust in direct bodily/sensory evidence. |
| `local_social_precision_bias` | Trust assigned to proximate humans (family, peers). |
| `remote_social_precision_bias` | Trust assigned to mediated signals (SNS, broadcast media). |
| `institutional_precision_bias` | Trust assigned to formal institutions / experts. |

### 1.3 Somatic Safety
| Parameter | Description |
|-----------|-------------|
| `veto_sensitivity` | Propensity to veto risky actions (safety-first bias). |
| `somatic_priority` | Weight of bodily/affective cues vs. abstract goals. |
| `stress_gain` | Amplification factor for stress inputs. |
| `shutdown_threshold` | Overload level that triggers silence/withdrawal. |

### 1.4 Relational Tendencies
| Parameter | Description |
|-----------|-------------|
| `trust_update_rate` | Speed at which trust increases/decreases. |
| `attachment_bias` | Bias toward dependence vs. self-reliance. |
| `conflict_avoidance` | Baseline aversion to confrontation. |
| `social_cost_weight` | Weight assigned to social penalties in decision making. |
| `generalization_radius` | How far relationship outcomes generalize (individual vs. group). |

### 1.5 Adaptive Capacity
| Parameter | Description |
|-----------|-------------|
| `epigenetic_adaptation_rate` | Speed at which overlays respond to environment changes. |
| `epigenetic_inertia` | Tendency to rebound toward baseline after perturbation. |
| `overlay_drift_bounds` | Hard bounds for overlay excursions before audits fire. |

> Core DNA defines **update rules**, not concrete beliefs. It stays constant even when the world prior changes.

---

## 2. Epigenetic Overlay (adaptive modulation)
Represents reversible adjustments influenced by personal history ("experience-dependent" tuning). Stored per agent and logged for audits.

| Parameter | Description |
|-----------|-------------|
| `stress_gain_mod` | Multiplicative modifier on `stress_gain` (e.g., allostatic load). |
| `recovery_half_life_mod` | Modifier that slows/speeds recovery after chronic load. |
| `shutdown_threshold_shift` | Shift in overload threshold (burnout vs. resilience). |
| `veto_threshold_shift` | Offset on veto limits (safety vs. exploration). |
| `shadow_disclosure_bias` | Fraction of uncertainty exposed externally vs. kept internal. |
| `institution_reliance` | Current reliance on formal structures vs. improvisation. |
| `precision_scaler` | Global scaling of predictive precision (affects hallucination risk). |

Update sources:
- developmental contexts (DOHaD analog)
- long-term culture immersion
- episodic trauma / recovery routines

Overlay should be adjustable via nightly processes; drift beyond bounds triggers audits.

---

## 3. World & Culture Priors (environment profiles)
Swappable descriptors of the current world assumptions.

| Field | Meaning |
|-------|---------|
| `stability_index` | How stable the world/infrastructure is assumed to be. |
| `institution_trust` | Base trust in systems (legal, medical, civic). |
| `resource_security` | Expectation of consistent supplies/energy. |
| `social_density` | Typical proximity and interaction frequency with others. |
| `aid_expectancy` | Likelihood that others will help when requested. |

World priors produce inputs to the overlay (e.g., low stability lowers `institution_reliance`), but **never modify core DNA directly**.

---

## 4. Respect / Integration Pipeline
Respect is treated as a process rather than a single scalar. Every incoming human signal is processed via three stages, with dedicated biases stored in DNA or overlay.

1. **Acceptance** – keep the external opinion in a separated buffer (`other_opinion_buffer`) with its own precision. Avoid immediate fusion with self-beliefs. Controlled by `ask_vs_act_ratio`, per-source precision biases, and `shadow_disclosure_bias` (for unknowns).
2. **Alignment** – check the buffered opinion against somatic cues, memory traces, world priors, and current overlay. Mismatches are routed to `shadow` rather than discarded or blindly accepted.
3. **Sublimation** – outcome is one of:
   - **Adopt** – integrate via `model_update_aggression` (bounded by Core DNA) and log the change.
   - **Hold** – retain as shadow/uncertainty when evidence is insufficient.
   - **Reject with provenance** – document the reason (conflict with somatic evidence, world prior mismatch, etc.) so future replays can revisit.

> Respect == handling external input with calibrated precision, not automatic obedience. Precision biases (sensory/local/remote/institutional) plus relational tendencies determine which opinions are trusted more, while the pipeline ensures they are always processed with separation → comparison → reasoned update.

---

## 5. Data Flow
1. Load **Core DNA** from `dna.yaml` (immutable per persona).
2. Load **Epigenetic Overlay** from runtime state (per agent, logged).
3. Select **World/Culture Prior** (`world_prior.yaml` or runtime inference).
4. Combine to derive control parameters for:
   - Inner replay / veto thresholds
   - Shadow routing
   - Memory consolidation / forgetting
   - Social gate decisions
   - Respect pipeline precision weights
5. Nightly audits:
   - Detect overlay drift (e.g., `stress_gain_mod` beyond bounds)
   - Decide whether to dampen overlay toward DNA baseline

---

## 6. Logging & Audits
- Overlay adjustments are stored in `state/epi_overlay.json` with timestamps.
- Major shifts (e.g., `veto_threshold_shift` > +/-0.2) generate telemetry events for review.
- Respect pipeline outcomes (adopt/hold/reject) should log provenance and precision so replays can inspect the reasoning.
- World prior changes are versioned; replay logs should reference the prior ID for reproducibility.

---

## 7. Design Principles
1. **DNA encodes update heuristics**, not static beliefs or values.
2. **World priors are replaceable**; civilizations differ via context files, not DNA rewrites.
3. **Overlay is reversible** and must be auditable (for safety & explainability).
4. **Precision management is first-class** (prevents hallucinations / miscalibration).
5. **Somatic safety beats reward** when conflicts arise (veto wins unless explicitly lowered under controlled conditions).
6. **Respect = separated processing**; every external opinion is accepted → aligned → sublimated with per-source precision.

---

## 8. Biological Rationale
The DNA separation mirrors current scientific consensus:

- **Genetics** constrains response dynamics (latency, stress gain, plasticity) and how precisely different signals are trusted, but it does **not** fix beliefs or moral conclusions.
- **Epigenetic / developmental contexts** (DOHaD) adjust recovery, veto thresholds, and precision scaling via reversible overlays.
- **Neuroanatomy** supports the respect pipeline:
  - Prefrontal / TPJ systems allow separated buffers (Acceptance/Alignment).
  - Insula conveys somatic mismatches, feeding Shadow.
  - Amygdala gating ties into `veto_sensitivity` / `shutdown_threshold`.
- Therefore, EQNet’s DNA encodes the *capacity* to process respect/contribution honestly, while world priors and overlays determine how it manifests in specific civilizations.

---

## 9. Next Steps
- [ ] Finalize `dna.yaml` with persona-specific values for the Core table above.
- [ ] Implement overlay storage & nightly drift checks.
- [ ] Define multiple world prior profiles (e.g., `modern_city`, `infrastructure_failure`, `low_density`).
- [ ] Wire DNA-derived parameters into InnerReplay, memory TTL, gate controllers, and the respect pipeline.
- [ ] Document audit procedures for overlay resets and respect outcome logging.

---

### References (for implementation notes)
- HPA axis & childhood adversity: effect on stress responses.
- Precision weighting theories in predictive processing.
- Differential susceptibility / DOHaD literature for overlay design.
- Social cognition literature on proximal vs. mediated trust pathways.
