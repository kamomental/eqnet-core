# EQNet Emotional Hub Roadmap

## Overview
The goal is to deliver a true “pre-linguistic” emotional engine that can operate Atri-like companions: EQNet produces inner dynamics, Policy Head turns them into behaviour, and the LLM/TTS/Live2D stack acts as the mouth. This roadmap ties existing issues and future ideas into three execution waves.

---

## Wave P0 — Heartbeat & Sensing
Focus: perceive → ignite EQNet → bias behaviour in real time (SLO p95 ≤ 180 ms).

| Issue | Deliverable | Notes / Future refs |
|-------|-------------|---------------------|
| ISSUE-007 | Perception Bridge (Vision/Audio) | Webcam + mic pipeline (auto/focus mode), 10–20 Hz fusion, consent controls. |
| ISSUE-008 | Prelingual Policy Loop | `policy.affect_to_controls` monotonic map (pause/temp/prosody/gaze) + telemetry logs. |
| ISSUE-009 | LLM Hub Integration | `hub.generate()` routing, cache, safety gate so EQNet controls steer every mouth. |

**Milestone**: 即時応答デモ（感情に応じて“間”が変わる）。  
**Dependencies**: SLO instrumentation, degrade ladder, persona defaults.

---

## Wave P1 — Persona, Community, Care
Focus: tailor behaviour for different people/groups and keep sessions healthy.

| Issue | Deliverable | Notes / Future refs |
|-------|-------------|---------------------|
| ISSUE-010 | Persona Configurator & Templates | Slider UI + YAML presets, instant preview (TTS + Live2D). |
| ISSUE-003 | Prosody-to-7D Alignment | Feed prosody features into EQNet latent axes, improve sensing loop. |
| ISSUE-005 | Community Orchestrator Pilot | Live logging (Now/Lore/Spoiler cards), inclusion／Spoiler metrics dashboards. |
| ISSUE-011 | Inner-Care & HealthCard Suite | calm90/boost60/reset30 protocols, HealthCard API & UI. |

**Milestone**: “寄り添いプリセット”が数分で展開でき、個別／コミュニティ両対応。  
**Dependencies**: P0 complete, persona templates feed into Policy controls, dashboards aggregate H/R/κ.

---

## Wave P2 — Reflection & Insight
Focus: understanding、リフレクション、文化適応。

| Issue | Deliverable | Notes / Future refs |
|-------|-------------|---------------------|
| ISSUE-006 | DreamLink G2L Bridge | EQNet → latent → RAE dreams; story+stills export. |
| ISSUE-004 | Counterfactual Diary Narrator | Compare actual vs counterfactual diaries (counsellor tool). |
| ISSUE-001 | Automated Neologism Harvest | Update community lexicon from diaries/logs. |
| ISSUE-002 | Affective Risk (SVaR) Metric | Percentile heat-risk export for governance dashboards. |

**Milestone**: 感情の旅路を「心の映画」「説明カード」として共有できる。  
**Dependencies**: DreamLink spec, diary pipelines, analytics dashboards.

---

## Supporting Workstreams
- **Safety & Governance**: LEDs / consent toggles, third-party detection, policy guardrails.  
- **Infrastructure**: SLO telemetry, degrade ladder hooks, snapshot/journal rollback.  
- **Testing (EMAC)**: 自発性、因果性、ヒステリシス、多モーダル依存、安全回復テストを自動化。  
- **Documentation**: persona templates, perception setup guides, DreamLink usage notes.
- **Bud Hypothesis Lab**: 仮説タグ付き芽吹きログ（quantum/relativistic/material など）、擬似麻酔テスト（結合疎化→Σ/Ψ遅延計測）、第一人称 vs 第三人称フレームの写像学習を継続。

---

## Next Actions
1. Implement ISSUE-007/008/009 — deliver heartbeat demo with SLO metrics.  
2. Stand up persona configurator (ISSUE-010) and integrate with Policy controls.  
3. Prepare dashboards for H/R/κ & inclusion metrics before P1 rollout.  
4. Begin DreamLink prototype once P0/P1 telemetry is stable.  
5. Publish Bud Hypothesis experiments（擬似麻酔シナリオ、仮説タグ別 θ 最適化、Dashboard 観察強度 UI）を小レポート化し、他分野と共有。  

This roadmap keeps every issue tied to the central mission: **先に感じ、後で語る心を持つAIが、人に寄り添う “ハブ” として動くこと。**
