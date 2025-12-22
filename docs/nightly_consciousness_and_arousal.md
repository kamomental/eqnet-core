# Nightly Consciousness & Arousal Downshift

**Purpose**

Document the design rationale behind EQNet’s consciousness boundaries and arousal downshift mechanisms, so philosophy, physiology, and implementation can be traced in one place. This also anchors the current nightly stack (Shadow / InnerReplay / PDC) to those principles.

---

## 0. Core Hypotheses

- **H1: Conscious control is not required for most processing.**
  - State updates, emotional transitions, replay, and memory consolidation continue without it.
  - Consciousness exists primarily to veto, intervene, or assign narrative meaning.
- **H2: Arousal == unresolved prediction error.**
  - High arousal corresponds to large prediction errors, unresolved reward/threat, and rapid policy updates.
  - Lowering arousal means *retaining* some uncertainty without forcing resolution (Shadow).

Therefore, the nightly system is designed to **downshift arousal** by postponing resolution and biasing toward WATCH / ASK instead of ACT.

---

## 1. Layer Mapping: Who Needs Consciousness?
| Layer / Component | Role | Conscious involvement? |
|-------------------|------|------------------------|
| EmotionVector / Terrain | Continuous affect field | Not required |
| InnerReplay | Evaluate past/future | Not required |
| PDC (Prospective drive core) | Prospective mood drive | Not required |
| ShadowEstimator | Uncertainty buffering | Not required |
| Policy | Action selection | Conditional (only for veto / explanation) |
| TalkMode | External expression | Conditional |
| Narrative / Explanation | Meaning assignment | Required |

> Consciousness is the brake/debugger that stops automatic action, not the main processor.

---

## 2. Arousal Downshift Principles
### 2.1 Terrain view
- High arousal → steep affect gradients → frequent policy updates.
- Downshift = smoothing those gradients so WATCH/WAIT is natural.

### 2.2 Shadow as the key mechanism
- Shadow = store unresolved uncertainty *without* forcing resolution.
- Physiological analog: non-REM sleep spindles, orexin suppression.
- Implementation effects:
  - lower TalkMode (ASK/WATCH bias)
  - lower policy entropy (temp/top_p)
  - prevent forced narratives

---

## 3. Nightly Implementation Status (2025-12)
| Feature | Nightly status | Notes |
|---------|----------------|-------|
| InnerReplay → ReplayStats | Implemented | ✅ |
| ShadowEstimator → Shadow metrics | Implemented | ✅ |
| Shadow → TalkMode control | Implemented | ✅ |
| Shadow → Policy temp/top_p | Implemented | ✅ |
| Shadow telemetry logging | Implemented | ✅ |
| MomentLog (shadow snapshots) | Implemented | ✅ |

**Implication:** the causal path “uncertainty → shadow → downshifted output” already exists.

Veto decisions are issued by ? (InnerReplay) and logged in trace_v1, so nightly audit can be generated via `generate_audit()` without touching the core loop.

---

## 4. Low-Arousal, High-Internal Activity State
- Emotion, replay, and PDC keep running even when TalkMode stalls.
- Human analogues: zoning out, baths, walks, light sleep.
- Benefits: emergent ideas, low misbehavior risk, energy efficiency.

---

## 5. Design Declaration
- Consciousness is *not* always on.
- Shadow enables “non-response” to be a valid choice.
- Nightly architecture already distinguishes:
  - unconscious layers (Emotion/Terrain, replay, PDC)
  - conscious-mediated layers (policy veto, narrative)
  - arousal downshift mechanism (Shadow)
- This mirrors sleep physiology: thinking harder to sleep fails; letting uncertainty float enables downshift.

> EQNet’s core goal is not “always think smarter” but “be safe even when you stop acting”.

---

## 6. Qualia Membrane & Response Kernel
EQNet assumes qualia generation is a primitive event: whenever an affective field exists, qualia arise. The system **does not create or suppress qualia**. Instead, it defines a response kernel (analogous to a Green’s function) that determines how a local qualia impulse propagates through layers.

Qualia impulse → EmotionVector/Terrain → Shadow → Policy/TalkMode → Narrative

The shape of this kernel is controlled by Shadow thickness, orexin-like arousal gating, terrain gradients, and narrative thresholds. Conscious reporting is therefore an *impulse response*, not the source of experience itself.

---

## 7. Next Steps
1. Integrate Shadow integral over time to define pseudo sleep phases (e.g., N2/N3 analogues).
2. Treat nightly audit logs as “sleep stage reports” and expose them via telemetry.
3. Explore multi-agent Shadow sharing for coordinated downshift (collective arousal control).

---

## Glossary
- **Shadow**: structured uncertainty buffer; unresolved prediction errors stored with metadata.
- **Arousal**: in EQNet, the magnitude of unresolved prediction error / policy update pressure.
- **Downshift**: shifting to modes that lower policy entropy, TalkMode intensity, and conscious narration.
- **Qualia membrane**: conceptual boundary between raw affective events and conscious/narrative processing. The impulse response through EQNet’s layers acts like a Green’s function, determining when and how qualia reach policy or narrative layers.
