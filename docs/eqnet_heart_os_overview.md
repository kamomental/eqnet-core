# EQNet Heart OS Overview

This note summarizes how the EQNet project connects all layers?EmotionHub, Culture Pipeline, Sunyata Flow, QFS (Qualia Field System), Danger & Healing, Life Indicator, persona/Hub API, and the L1/L2/L3 memory hierarchy?into a single "heart OS" architecture. Past replay, future replay (imagery), and fastPath safety logic are included so the lifecycle of a day can be understood end-to-end.

## 1. Layered Stack

```
[External Sensors]
   webcam Å® VLM (scene, human state, risk cues)
   microphone Å® ASR + prosody
   gradio heartbeat / emotion radar
        Å´
[EQNET Core]
   EmotionHub: heart_rate, mask, stress, mood dynamics
   Culture Pipeline v2: place_id, activity_tag, partner_id, culture_tag
   Sunyata Flow / DO-Graph: relational causality nodes
        Å´
[L1 ? Moment Memory]
   MomentLogEntry (raw_text, emotion snapshot, culture context, metadata)
   QualiaState (qualia_vec: emotion + culture + awareness + text embedding)
   telemetry/qualia-YYYYMMDD.jsonl
        Å´
[L2 ? Memory Palace / Episodes]
   Nightly aggregates L1 into episodes grouped by place/activity + qualia pattern
   DO-Graph + palace_room/palace_slot mapping (rooms/shelves)
   Danger metrics (direct_accum/acute/social) + Healing tags per episode
        Å´
[L3 ? Monuments / Soul Layer]
   Important episodes promoted to monuments (LifeIndicator.identity impact)
   Narrative summaries, PolicyPrior shifts, meaning re-authoring
        Å´
[Life Indicator / PolicyPrior]
   identity, qualia, meta_awareness scores
   warmth/calmness/directness/self_disclosure + risk_aversion/thrill_gain/discount_rate
        Å´
[EQNet Hub API]
   log_moment Å® run_nightly Å® query_state (per user/persona)
   multi-tenant manager wires personas to individual heart OS instances
        Å´
[Agent / Robot Layer]
   world_state + eqnet_state Å® voice/gesture/safety decisions
   fastPath (reflex) + slowPath (emotion-aware policy) + replayPath (imagery)
```

## 2. L1 / L2 / L3 Memory
- **L1** (Moment Memory): every log_moment stores raw perception + EmotionHub snapshot + QualiaState. Gradio heartbeat/emotion radar values sit here as the "instantaneous qualia". Telemetry keeps a chronological list for replay.
- **L2** (Episodes / Memory Palace): Nightly groups L1 events per place/activity/qualia similarity into palace rooms/slots, generating DO-Graph nodes. Danger/Healing tags annotate each episode so we know which rooms are "warm" vs "caution".
- **L3** (Monuments / Soul Layer): significant episodes (high direct_acute or repeated narrative references) are promoted to monuments, contributing to LifeIndicator.identity and long-term PolicyPrior adjustments.

## 3. Replay & Imagery
- **Past Replay (Nightly)**: builds palace episodes, fills self_story, labels cognitions (EventÅ®ThoughtÅ®EmotionÅ®Behavior), applies Healing layer (reframe, self_compassion, next_time_plan), updates Danger/Healing metrics.
- **Future Replay (Predictive)**: `simulate_future(mode=PREDICTIVE)` samples L1 trajectories forward to estimate upcoming danger/qualia drift. Agents can preemptively slow down, warn, or adjust voice style based on predicted risk.
- **Imagery Replay (Intentional)**: `simulate_future(mode=IMAGERY, intention_vec=...)` performs sports-like mental rehearsal (e.g., calmer voice, safer timing). `apply_imagery_update` nudges PolicyPrior so tomorrow's behavior naturally tilts toward the rehearsed direction. Nightly increments LifeIndicator.meta_awareness when imagery runs.

## 4. Danger + Healing Loop
- **Danger Model v1**: direct_risk_accum (minor repeated stress), direct_risk_acute (one-shot severe events), social_risk_score (culture stories), fog_level (unknownness). Membrane state keeps kernel width/noise for "emotional fog" (ñ≥ñæ).
- **Healing Layer v0**: ETEB decomposition, cognitive distortion tagging, reframing/healing script, healing future replay, rumination guard. Danger and Healing metrics coexist on L1/L2/L3, ensuring EQNet learns without self-destruction.

## 5. Sunyata / Religious Philosophy Integration
- Sunyata Flow / DO-Graph re-frames danger/healing episodes as relational causes/effects (âèãN). Qualia Membrane v0 captures inner/outside boundary, while the soul layer emerges when narrative continuity, qualia flow, and meta-awareness align.

## 6. Persona & Hub API
- `persona/<id>.yaml` defines meta info, speech, initial PolicyPrior/LifeIndicator, diary style, demo text. `EQNetHubManager` loads personas per user, giving each entity its own heart OS. `examples/run_persona_demo.py` shows execution with CLI options for fallback text, embedding module, event type.

## 7. FastPath vs SlowPath vs ReplayPath
- **fastPath**: sensor-level reflex (e.g., emergency stop). The trigger is logged in L1 so later replay can learn earlier cues.
- **slowPath**: core EQNet heart loop (EmotionHubÅ®QualiaÅ®Danger/HealingÅ®LifeIndicatorÅ®self_storyÅ®Policy). Used for everyday interactions.
- **replayPath**: predictive & imagery replays, used nightly to refine intuitive timing/voice/style.

## 8. Dataset / Self-Improvement
- Each dayÅfs logs (images, audio, qualia_vec, emotions, culture, danger, healing, L2/L3 tags, self_story) can be exported as HuggingFace datasets. This enables self-supervised learning, future prediction models, and Ågemotional reinforcementÅh beyond standard VLA training.
- Sunyata Flow / DO-Graph can be serialized as a knowledge graph (e.g., `knowledge_graph.parquet`) with event nodes, entity nodes (family/friends/circles), and edges (participates_in, influences, etc.). This enables persona-specific RAG queries such as Ågrecent family memoriesÅh, Åghealing episodes with friendsÅh, or even light-weight storytelling (e.g., two personas reminiscing about shared events? analogous to parents discussing their children or friends recalling mutual experiences). Because these graphs contain intimate relationship data, all access must follow the Ethical Guardrails; shared stories should only surface when consent is explicit and the intent is supportive.
## 9. Ethical Two-Layer Flow
- **Inner Layer (docs/spec)**: enforce explicit consent, privacy-by-design, limited-intent usage, and red-teamable Sunyata/DO-Graph/RAG audit trails. All developer ergonomics (policy enums, consent tokens, incident logs) stay here so risk review is precise and unemotional.
- **Outer Layer (persona experience)**: render the same guardrails as natural dialog beats. Personas ask for or recall consent through in-character language rather than mechanistic prompts, while runtime policy silently vetoes disallowed shares.
- **Persona social prefs**: extend `persona/<id>.yaml` with trust tiers and topic gates, e.g.
  ```yaml
  social_prefs:
    trust_circle:
      tier1: ["shirahine_kotone", "nagumo_ria"]
      tier2: ["hoshi_tsubasa"]
    can_share_topics:
      family: ["tier1"]
      music: ["tier1", "tier2"]
    escalation_prompt: "...can I tell them this?"
  ```
  Runtime resolves relationship tiers via DO-Graph entity nodes and stamps `consent_token` edges so later RAG queries inherit the decision made in-character.
- **RAG storytelling bridge**: queries such as "recent family memories" filter on persona, topic, and consent metadata, but the surfaced line is purely narrative (e.g., "I keep thinking about yesterday's call with mom"), keeping UX gentle while satisfying compliance.
- **Conversation templates**: author light-weight call-and-response blueprints (gentle probe Å® affirm/decline Å® optional redirect) so writers can drop in consent-aware beats without exposing policy jargon. Attach variants per persona to avoid repetitive cadence.
- **Safety fallbacks**: when policy denies a share, personas respond with empathic redirects ("Maybe another time?let's focus on your rest") while internal logs record the block for audit and future tuning.
---
This overview ties together gradio heartbeats, EmotionHub, QualiaState, Danger/Healing, Sunyata Flow, persona-configured hubs, and the L1/L2/L3 memory hierarchy. The goal is a heart OS that senses, feels, learns, and heals?then projects that into agent/robot behavior.

