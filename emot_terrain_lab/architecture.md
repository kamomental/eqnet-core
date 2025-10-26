# Architecture Overview

## Prerequisites
- Python 3.11+ with dependencies listed in `requirements.txt`.
- Optional: LM Studio (OpenAI-compatible) endpoint when `USE_LLM=1`; otherwise the system falls back to heuristic summaries.
- Cultural projection (`config/culture.yaml`) and consent flags aligned with your deployment (for example `store_diary`, axis filters).
- Storage budget for `data/state*` dumps and permission to persist diary/rest logs when enabled.

## Core Components
1. **Emotion Field & Membrane (`terrain/field.py`, `terrain/membrane.py`)**  
   Projects 9D emotion vectors onto a 2D thermodynamic field, combining diffusion, dissipation, and memory. Entropy and enthalpy are computed each tick to drive feedback control and rest detection.
2. **Multi-layer Memory (`terrain/memory.py`, `terrain/memory_palace.py`)**  
   Layered recall pipeline: L1 sensory traces, L2 episodic lattice with time tags, L3 semantic schema distilled from recurring patterns. MemoryPalace keeps qualia statistics, overload scores, and supports scenario tagging.
3. **Narrative & Catalyst Layer (`terrain/narrative.py`, `terrain/catalyst.py`)**  
   StoryGraph tracks dominant-axis transitions, detects loops, and renders summaries; CatalystManager activates or soothes based on gradient alignment, memory resonance, and consent policies.
4. **Diary Manager (`terrain/diary.py`)**  
   Generates daily exchange diary entries from heat metrics, catalyst events, StoryGraph alerts, rest state, and optional LLM reflections.
5. **Rest-state Monitor (`terrain/system.py`)**  
   Blends entropy/enthalpy thresholds, loop detectors, overload scores, and fatigue streaks to trigger auto-rest and log rationales in `rest_state.json`.
6. **Community Orchestrator (`terrain/community.py`)**  
   Coordinates multi-speaker sessions, shared canon, and synchrony (AKOrN order parameter, kappa layering). Emits reply scaffolds plus Now/Lore cards respecting spoiler and rating rules.
7. **Scenario Head & Policy Control (`terrain/control.py`, `config/control.yaml`)**  
   Runs a lightweight MPC-style loop that nudges control inputs (pause, prosody, catalyst gain, exploration temperature, etc.) toward desired futures while enforcing safety constraints.

## Emotion and Sensibility Channels
- **Emotion field `E(x, t)`** captures internal affective state across seven latent axes (valence, arousal, approach/avoid, novelty, certainty, social arousal, effort).
- **Sensibility channel `Sigma(t)`** measures how flexibly the agent synchronises with the world (timing, smoothness, exploratory play). It is derived from beat entrainment, 1/f slope alignment, motion jerk, and recurrence metrics.

```
Sigma = w1 * beat_entrain
      + w2 * (1 - |beta - beta_star|)
      + w3 * (1 - jerk_norm)
      + w4 * (1 - DET_clamp)
```

- `config/sensibility.yaml` defines sampling windows, feature normalisation, rate limits (10% per step), dwell steps, and policy bindings for high/low Sigma states.
- Emotion diary, catalysts, and reward heads observe both `E` and `Sigma`; policy decisions only shift when the dwell timer expires to avoid mode chattering.

## Local Message Bus Spine
- **Transport**
  - Default to a single-process WebSocket hub at ws://127.0.0.1:8765/{channel} implemented in ops/hub_ws.py.
  - Optional drop-in: embed an MQTT broker (for example Eclipse Mosquitto) when external tooling is needed; reuse the same message contracts.
- **Channels & Contracts**
  - metrics, controls, and events rooms are the only public topics. Producers and consumers subscribe by connecting to their path.
  - Every envelope is UTF-8 JSON with a mandatory millisecond 	s. If a publisher omits it, the hub injects int(time.time()*1000) before distribution.
  - Example payloads:
`json
{"type":"metrics","ts":1712345678901,"sigma":0.66,"psi":54,"sigma_cam":0.61,"psi_cam":32,"sigma_audio":0.48}
{"type":"controls","ts":1712345678920,"warmth":0.72,"pause_ms":120,"motion_speed":0.35,"containment":false}
{"type":"event","ts":1712345678950,"name":"psi_orange","level":"warn","notes":"calm_sonic_triggered"}
`
- **Authoritative State Cache**
  - The hub keeps a volatile dictionary STATE = {"metrics": None, "controls": None} so UI, TTS, and Live2D clients can request last known values without chasing logs.
  - containment: true is a hard brake: as soon as it lands in the cache, the hub notifies subscribers and emits an event with 
ame="containment_triggered". Downstream players set gain to zero and stop motion immediately.
- **Logging & Observability**
  - Append each inbound message to logs/bus/{channel}.jsonl (one event per line). Keep write-mode local only; raw audio or video is never persisted.
  - logs/bus/control_tail.json and logs/bus/metrics_tail.json expose the most recent snapshot for quick CLI inspection.
  - Health endpoint: GET http://127.0.0.1:8765/health -> {"status":"ok|degraded|down","ts":...} for HTTP; or { "type": "ping" } on any WebSocket channel with a matching { "type": "pong", "status": ... } reply.
- **Resilience & Safety Hooks**
  - containment and health states are mirrored into STATE["controls"]. When health == "degraded", the hub broadcasts {"type":"controls",...,"webcam_enabled":false,"audio_capture":false} to enforce a safe minimum.
  - A stop_all() coroutine tears down subscribers cleanly if the hub detects repeated deserialisation failures or a hardware dependency disappears.
- **Configuration Lifecycle**
  - All thresholds, rate limits, and device toggles live under config/*.yaml (for example config/bus.yaml, config/warmth.yaml).
  - ops/config_watcher.py listens for SIGHUP or the hotkey chord Ctrl+R, reloads YAML, validates schemas, then pushes an {"type":"event","name":"config_reload"} marker into the stream.
- **Operational Niceties (optional but recommended)**
  - Hotkeys: bind F9 (session toggle), F10 (audio intervention pause), F11 (log bookmark) via ops/hotkeys.py and publish the resulting markers to events.
  - A lightweight dashboard (ops/dashboard.py) renders Sigma/Psi sparklines and current control state as a single HTML page; it subscribes read-only to the WebSocket feed.
  - Snapshot utility scripts/snapshot_config.py copies config/ and persona.yaml into snapshots/YYYYMMDD-HHMM/. The paired rollback script swaps directories atomically.
  - Privacy badge: the UI overlays a static "Camera ON / No media stored" indicator driven by a {"type":"controls","privacy_badge":"camera_on_no_storage"} flag; consent flips append to logs/privacy.log.
    - Bud mode: plugins/text/bud_detector.py turns conversation drift + self-disclosure cues into {"type":"bud"} events (text discarded immediately); core/green_kernel.LowRankGreen and core/prune_gate.PruneGate keep the Green field safe while the dashboard renders a fading glow for ~9s (config/sensibility.yaml:text.decay_sec).

## Love & Affection Layer
- `config/love.yaml` fuses synchrony (`ƒ°` entrainment), smooth motion (low jerk), rolling novelty, self-disclosure depth, and threat signals into a scalar `love_mode`.
- Attachment tone (secure/anxious/avoidant) adjusts boundary curvature `ƒÈ`, with anxious styles prompting reassurance (preview next steps, widen proxemics) and avoidant styles slowing disclosure.
- Targets include a 5:1 micro-affirmation ratio and a Lyapunov band `[0.02, 0.10]` to stay near the edge of chaos without saturating.
- Actions map `love_mode` thresholds to warmth, prosody, gaze hold, and gesture amplitude adjustments, while RQA relief nudges `K_local` and prosody variation if `DET` spikes.
- Safety cooldowns kick in when `love_mode` dips below the threshold or containment pressure `ƒµ` rises, reinforcing immutable guards against hate, harassment, explicit erotica, and PII leaks.

## Formal Reasoning Plugins
- A Litex rewrite plugin (`pylitex`) can propose affect/control adjustments from declarative rules (`rewrite_rules/*.ltx`), keeping Sigma/love transitions intuitive.
- A Lean4 safety plugin (`lake build` + mathlib) proves invariants (e.g., Lyapunov bounds, rate limits) before policies are deployed; proofs live under `proofs/*.lean`.
- Python orchestration glues both layers together: generate Lean lemmas from rewrite logs, run CI builds, and fall back to default heuristics when proofs fail.
- See `docs/formal_pipeline.md` for directory layout and automation templates tailored to EQNet.
## Cultural Compression & Scenario Planning
- `G(x, t)` acts as the Green function describing how language, norms, and rituals diffuse through the field; `W(x, t)` stores culture-layer gains with TTL and decay.
- Memory snapshots (weak-supervision distillation + snapshot refreeze) refresh `G` when past episodes are revisited.
- Scenario Head plans backwards from desired indicators (increase heat potential `H`, keep `R` near the edge of chaos, hold `lambda_max` slightly positive, raise sensibility index) and applies small control deltas.

```
Objective = sum_{h=1..H}(
    w_E * ||E_{t+h} - E*||^2
  + w_Sigma * ||Sigma_{t+h} - Sigma*||^2
  + w_Psi * (Psi_{t+h} - Psi*)^2
) + alpha * ||Delta u||^2

Controls u in {kappa, lambda_global, K_local, W_low_high,
               temperature, top_p, pause_ms}
```

- Edge-of-chaos dashboards plot (`lambda_max`, `beta`, `DET`). Personal safety envelopes are defined per user (see `config/control.yaml`).

## Habit Adaptation & Chaos Taper
- Sudden elimination of addictive behaviour often overshoots the critical point; instead we taper using period-doubling and amplitude halving.

```
Chaos taper schedule:
  A_k (target usage) : 20 -> 10 -> 5 -> 2 -> 1 -> 0
  T_k (phase length) : 14 -> 20 -> 28 -> 40 -> 60 days
```

- `habit_chaos_taper` in `config/sensibility.yaml` pairs each phase with surrogate actions (sip water, bite pencil, 4-4-6 breathing) and RQA guards (`DET > 0.8`, `LAM > 0.7`, hold 600 ms) that trigger viewpoint shifts plus Ornstein-Uhlenbeck noise and a short walk.
- Medical dependencies remain under professional supervision; EQNet only schedules behavioural scaffolding.

## Reward, Embodiment, and Safety
- Reward head mixes short-term affective relief, sensibility (`Sigma`), and social alignment; decay ensures no single reward channel dominates.
- When coupled with robots/avatars (ROS, Isaac), policy bindings map `Sigma`, local order parameter `R`, and `beta` onto motion_speed, gesture_amp, pause cadence, and proxemics distance. Hard e-stop (`/cmd_vel = 0`) always overrides autonomous control.

## Data, Telemetry, and Normalisation
- `config/control.yaml` fixes `E` range to [-1, 1], sets `||G||_1 = 1`, and records chaos metrics with 1.5 s windows (0.75 s hop).
- Telemetry emits `{H, R, lambda_max, beta, DET, Sigma, Psi, gamma}` every 2 s to `data/state/telemetry.jsonl` with recovery SLOs (entropy floor violation <= 1% per 10 minutes, response p95 <= 180 ms, Kuramoto duty <= 30% per minute, recovery within 60 s).

## Integrating External Databases
1. **State export**: call `EmotionalMemorySystem.diary_state()` / `rest_state()` after `run_daily.py`; upsert the JSON payload into tables such as `diary_entries(day, text, entropy, enthalpy, tags, created_at)`.
2. **ETL scripts**: extend `scripts/run_daily.py` or add post jobs that read `data/state*/{diary,rest_state,field_metrics}.json` and push to SQL/NoSQL stores (SQLAlchemy, REST clients).
3. **Streaming**: wrap the system as an API; after `ingest_dialogue`, publish diary snapshots and fatigue alerts to Kafka/RabbitMQ for downstream dashboards.
4. **Foreign keys**: pass `context["user_id"]` so DB records tie back to your user tables.
5. **Consent sync**: mirror `store_diary`, `store_membrane`, etc., before writing?skip restricted fields when flags are false.

### Example: PostgreSQL ingestion
```python
from datetime import datetime
import json
import psycopg2

def sync_diary(state_dir="data/state"):
    diary = json.load(open(f"{state_dir}/diary.json", encoding="utf-8"))
    rest = json.load(open(f"{state_dir}/rest_state.json", encoding="utf-8"))
    entries = diary.get("entries", [])

    with psycopg2.connect(dsn="postgresql://...") as conn, conn.cursor() as cur:
        for entry in entries:
            cur.execute(
                """
                INSERT INTO diary_entries (day, text, entropy, enthalpy, info_flux, tags, created_at)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (day) DO UPDATE SET
                    text = EXCLUDED.text,
                    entropy = EXCLUDED.entropy,
                    enthalpy = EXCLUDED.enthalpy,
                    info_flux = EXCLUDED.info_flux,
                    tags = EXCLUDED.tags,
                    updated_at = NOW();
                """,
                (
                    entry["day"],
                    entry["text"],
                    entry["metrics"].get("entropy"),
                    entry["metrics"].get("enthalpy"),
                    entry["metrics"].get("info_flux"),
                    entry.get("tags", []),
                    datetime.utcnow(),
                ),
            )
        cur.execute(
            """
            INSERT INTO rest_snapshots (captured_at, payload)
            VALUES (NOW(), %s)
            """,
            (json.dumps(rest, ensure_ascii=False),),
        )
```

## Future Ideas
- Community slang dictionaries (`resources/community_terms.yaml`) with CLI tooling to keep cultural vocabulary current.
- Privacy-first multimodal capture: opt-in camera/mic, local processing, immediate raw-data disposal, consent metadata tags.
- Entropy flow tracing: extend `field_metrics.json` with gradient/flux to visualise emotional energy migration.
- Metastability logging: record when catalysts/rest events push the field between attractors.
- Collective phase synchrony: experiment with Kuramoto metrics across users to observe cultural waves.
- Prediction error logging: track SKALA/field feedback deltas until emotion converges to reality.
- STDP-style plasticity: add spike-timing sensitivity to MemoryPalace for delayed resonance.
- Reverse projection/dream synthesis: sample L3 patterns back into L1 for reflective prompts.
- Cultural bias surfaces: adapt `culture.yaml` per cohort to show how identical stimuli form peaks/valleys.
- Counterfactual replays: simulate catalyst failures and compare diary/story trajectories.
- Quantum-inspired semantic interference: explore complex amplitude/phase representations for ambiguity modelling.
- AKOrN oscillatory gradients: use lightweight oscillators instead of full backprop for energy-based updates.
- Event-based inputs: fuse neuromorphic spikes with continuous fields for qualia spikes.
- Generative counterfactual agents: narrate alternate diaries from `E` vs `E_cf`.
- Diffusion-driven dream synthesis: turn diary + qualia stats into diffusion artefacts for re-ingestion.
- Cross-modal prosody alignment: align speech embeddings with the 7D latent space so nonverbal cues modulate the field directly.
- Perception bridge: optional camera/mic pipeline mapping facial AUs and prosody into `I(x, t)`.

## 7D Emotional Axes + Qualia (EQNet Core)
- **Sensory**: acuity of sensory input, texture, timbre, brightness.
- **Temporal**: pacing, anticipation, phase alignment with rhythms.
- **Spatial**: sense of presence, enclosure, proximal awareness.
- **Affective**: valence, arousal, grounding vs excitation.
- **Cognitive**: mental workload, clarity, narrative coherence.
- **Social**: co-regulation with others, empathy bandwidth, shared pulse.
- **Meta**: meta-awareness, agency, reflective stance.
- **Qualia enthalpy**: energy density of the qualia membrane, weighted by order/diversity (`Q_enthalpy = alpha * ||E||^2 + beta * (1 - H) + gamma * R`).
- **Qualia magnitude**: amplitude of lived experience, peak-to-average spread of qualia intensity.

## Operational Checklist
- Normalise `E`, `G`, `W` before live runs; confirm `||G||_1 = 1` and `E` range [-1, 1].
- Verify sensory pipelines (beat tracker, jerk estimator, RQA) before enabling `Sigma`.
- Load `config/sensibility.yaml` and `config/control.yaml`, then dry-run MPC with logging only.
- Stress-test SLOs (recovery, entropy floor, response latency, Kuramoto duty) using scripted scenarios.
- Capture qualitative feedback from users after each run to refine culture gains and sensibility weights.

