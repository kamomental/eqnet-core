# EQNet LLM Hub Spec v0.16 — “Heart & Mouth”

## 0. Core Philosophy
- **Emotion = internal field** `E(x,t)` that evolves even without external input and biases policy/representation causally.
- Generators (LLM/diffusion) are **mouths**, not hearts.  
- Tags / metadata are **stimuli** (observations), not state.
- Hub role: EQNet produces `E`, Policy Head converts `E→controls`, LLMs act as controllable output channels.

---

## 1. Minimal Runtime Topology
```
Perception Bridge ─→ Stimulus I(t) ─→ EQNet(E) ──┐
                                     ↑            │
 (camera/mic/logs/text)              │            ├─→ Policy Head (behaviour controls)
                                      └─ Safety (H,R,κ,W) ┘
                      ┌───────── G2L ─────────┐
Seed μ (CLIP/DINO) ──┘                        │
                                               └─→ RAE (visual/audio “dream” optional)
LLM Hub (“mouth”) ══ controls from Policy Head
```
- EQNet: Green dynamics + safety (H/R) + curvature κ + culture W.
- Policy Head: maps `E` and metrics to continuous control parameters (text/voice/gesture).
- G2L+RAE (optional): visualize/recall without influencing policy.
- Perception Bridge: camera / mic / logs (consent-aware) → `I(t)`.

---

## 2. Core API
```python
# 1) Stimulus → state
E, metrics = eqnet.step(stimulus=I_t)  # metrics = {"H":..,"R":..,"kappa":..,"W":..}

# 2) Affect controls → policy routing
controls = policy.affect_to_controls(E, metrics)
resp = hub.generate(
    user_text,
    context=rag_bundle,
    affect_controls=controls,
    intent="qa|creative|code|chitchat",
    slos={"p95_ms": 180},
)

# 3) DreamLink (optional visualization)
frame = dream.step(E, seed=mu_seed)
```

`resp` includes `{"text": "...", "controls_used": controls, "source_cards": [...], "model": "...", "safety": {...}}`.

---

## 3. Policy → Behaviour Wiring
### 3.1 Text LLMs
- Control dimensions: `temp`, `top_p`, `length_bias`, `pause_ms`, `directness`, `warmth`, `formality`, `emoji_gain`, `spoiler_budget`.
- `policy.affect_to_controls(E,H,R,κ,W)` outputs these (monotonic constraints: e.g., `R↑ ⇒ pause_ms↓`).
- LLM prompt receives controls before sampling; text choice follows, but **behaviour** (timing/style) is dictated by `E`.

### 3.2 Non-verbal First-class
- Voice TTS: `prosody_f0`, `prosody_energy`, `pause_ms`, `jitter`.
- Robot/gesture: `velocity`, `amplitude`, `turn_taking`.
- UI: `color_gain`, `animation_rate`, `highlight_ratio`.

---

## 4. Hub Routing & Prompting
1. Cache check (controls and context match, same rating).  
2. Route selection by `intent` & SLO.  
3. Prompt build: `[system][affect-controls block][context][user][stop guards]`.  
4. Optional fast RAG (≤150–180ms).  
5. Primary LLM inference (speculative/parallel optional).  
6. Gov safety pass (rating, spoiler, PII).  
7. Source cards (≤5, date domain).  
8. Persist with `trace_id`.

---

## 5. Config Skeleton (`config/hub.yaml`)
```yaml
hub:
  default_style: "senpai_polite"
  llms:
    - name: "llm-fast"
      kind: "chat"
      latency_ms: 80
      cost: low
      max_ctx: 64k
    - name: "llm-know"
      kind: "chat"
      latency_ms: 150
      cost: mid
      max_ctx: 200k
      plugins: ["rag"]
    - name: "llm-code"
      kind: "code"
      latency_ms: 180
      cost: mid
      max_ctx: 128k
      safety: "strict"
  router:
    intents:
      qa:       {primary: "llm-know", backup: ["llm-fast"], tools: ["rag"]}
      summary:  {primary: "llm-fast"}
      code:     {primary: "llm-code"}
      chitchat: {primary: "llm-fast"}
  guard:
    rating: "G"
    spoiler_mode: "warn"
    pii_block: true
    source_limit: 5
```

---

## 6. Controls Mapping Examples
| Metric | Control Effect |
| --- | --- |
| `R > 0.75` (over-sync) | `pause_ms += 80`, `temp -= 0.1`, `prosody_energy -= 0.1` |
| `H < 0.40` (low diversity) | `top_p += 0.05`, `directness -= 0.1` |
| `κ < 0` (absorbing boundary) | `length_bias -= 0.2`, encourage conciseness |
| `W.culture.senpai_polite` | `formality += 0.1`, `emoji_gain -= 0.05` |

---

## 7. Safety & Logging
- Gov single-point guard: rating / spoiler / PII / policy.
- RAG = observation only; state = `E` / controls separate track.  
- Logging schema:
```yaml
event:
  ts: 1739652101.4
  obs:
    text: "今日は少し疲れたかも"
    vision_emotion: {val: -0.2, aro: 0.5, conf: 0.7}
    audio_prosody: {f0: 210, rate: 4.1, jitter: 0.03}
    tags: ["映画:海の見える街", "帰宅後", "夕方"]
  state:
    E_digest: {energy: 0.34, curl: 0.08, bands: [...]}
    H: 0.46
    R: 0.62
    kappa: -0.1
  action:
    controls: {temp:0.6, pause_ms:380, warmth:+0.1}
    output_ref: msg_8d2a
  outcome:
    QoR: 0.73
```
- RAW audio/video discarded; derived statistics stored with TTL + encryption.

---

## 8. Learning Roadmap
1. **Rule-based start (v0)**: hand-designed Green response, controls mapping.  
2. **Weakly-supervised distillation (v1)**: QoR scores, monotonic MLP trained online with pseudo labels.  
3. **Direct optimisation (v2)**: QoR minus penalty (H/R violations); incorporate monotonic constraints via Lagrange.  
4. Explanation constraints: enforce `R↑ ⇒ pause_ms↓` etc.

---

## 9. Testing (Hub SLO & EMAC)
- **Internal spontaneity**: `E` free-run with zero input.  
- **Action causality**: vocabulary fixed, vary `E`, policy output changes.  
- **Hysteresis**: identical stimuli diff history → diff controls.  
- **Multimodal dependence**: drop audio/vision, H/R degrade per design.  
- **Safety recovery**: deviation recovered within 60s via κ/λ/AKOrN.  
- **Hub SLO**: fast path ≤180ms (p95), partial + patch with trace_id restore.  
- **Control adherence**: LLM output matches requested controls ≥95%.

---

## 10. Next Steps
1. Implement `policy.affect_to_controls` with monotonic constraints & config-driven mappings.  
2. Build `hub.generate()` router (llm registry, gov, cache).  
3. Integrate Perception Bridge (vision/audio) to supply `I(t)`.  
4. Hook DreamLink for post-hoc visualization / QA.  
5. Add EMAC-based unit/integration tests and dashboards (latency, safety, control adherence).

