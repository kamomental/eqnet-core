# Issues Roadmap — EQNet Hub Alignment

## P0 — Core Loop & Sensing
| ID | Title | Summary | Rationale |
|----|-------|---------|-----------|
| ISSUE-007 | Perception Bridge (Vision/Audio) | Ship the camera/mic affect pipeline (consent-aware) that injects valence/arousal stimuli into EQNet, meeting latency targets (p95 ≤ 50 ms). | Enables real-world sensing while respecting privacy; prerequisite for true pre-linguistic ignition. |
| ISSUE-008 | Prelingual Policy Loop | Implement `policy.affect_to_controls` with monotonic constraints (pause/temp/prosody/gaze) so `E → 行為` 映写が即時に機能する。 | Establishes “先に感じる→ふるまい” コアロジック; ties EQNet metrics to behaviour across all output channels. |
| ISSUE-009 | LLM Hub Integration | Build `hub.generate()` routing (registry, cache, safety gate) so multiple LLM/Tool stacks obey EQNet controls and SLO (p95 ≤ 180 ms). | Makes EQNet a heart while LLM群は口; central to operating as a conversational hub. |

## P1 — Persona, Community, Care
| ID | Title | Summary | Rationale |
|----|-------|---------|-----------|
| ISSUE-010 | Persona Configurator & Templates | Provide slider/UI + YAML presets to create persona profiles (warmth, directness, talk pace, etc.) with instant preview. | Allows per-user/scene tailoring (家庭・配信・介護) and quick deployment of Atri-like characters. |
| ISSUE-003 | Prosody-to-7D Alignment | Map speech prosody (tempo/pitch/energy) into EQNet latent axes during opt-in sessions. | Strengthens prelingual sensing loop; improves pause/prosody controls feedback. |
| ISSUE-005 | Community Orchestrator Pilot | Connect `terrain/community.py` to live sessions, surface Now/Lore cards, and log inclusion/Spoiler metrics. | Extends hub to group settings; enables “聞き耳”モードとの切替。 |
| ISSUE-011 | Inner-Care & HealthCard Suite | Implement calm90/boost60/reset30 protocols and HealthCard dashboards referencing H/R違反率。 | Keeps寄り添いAIの整流と透明性を確保（臨床・家庭双方向け）。 |
| ISSUE-012 | PaddleOCR-VL Narration | Integrate PaddleOCR-VL 0.9B to parse charts/slides, feed summaries into EQNet, and drive reading-aloud UX (TTS output, EQNet-driven prosody). | Enables content-aware narration that respects emotional context; bridges vision-to-speech pipeline. |

## P2 — Reflective & Insight Layer
| ID | Title | Summary | Rationale |
|----|-------|---------|-----------|
| ISSUE-006 | DreamLink G2L Bridge | Implement the Green-to-Latent module (`specs/dreamlink_g2l_v015.md`) and Dream runtime so EQNet heatmaps drive RAE dream frames. | Connects inner dynamics to reflective “心の映画”; aids self-reflection/testing. |
| ISSUE-004 | Counterfactual Diary Narrator | Use `E_cf` to author alternative diary entries and compare with actual logs. | Provides explainability and what-if narratives for counsellors/users. |
| ISSUE-001 | Automated Neologism Harvest | Scan diaries/logs for new slang and propose updates via `update_community_terms.py`. | Keeps cultural lexicon fresh; supports persona tuning. |
| ISSUE-002 | Affective Risk (SVaR) Metric | Export percentile-based heat risk indicators to dashboards。 | Supports monitoring and governance (臨床/運用チーム向け). |
