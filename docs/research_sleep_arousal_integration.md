# Sleep & Arousal Research Integration (EQNet Core)

**Purpose:** Translate key physiological and computational findings on sleep/arousal into explicit control variables for EQNet core.

---

## 1. Goal
- Bridge neuroscientific evidence (orexin, NREM spindles, glymphatic clearance, parasomnias) to EQNet’s Shadow / InnerReplay / PDC stack.
- Provide concrete latent variables and nightly passes that map to those mechanisms.

---

## 2. Research Overview
| Theme | Findings | References |
|-------|----------|------------|
| **Orexin (Hypocretin) gating** | Oversees arousal, reward, metabolic drive; effectively gates policy complexity. | [Pharm Rev 2017](https://pharmrev.aspetjournals.org/content/69/2/312), [bioRxiv 2024](https://www.biorxiv.org/content/10.1101/2024.02.27.582228v1) |
| **N2 spindles & coupling** | SO–sleep spindle–ripple sequences consolidate procedural memory; coupling strength predicts recall. | [ScienceDirect 2020](https://www.sciencedirect.com/science/article/abs/pii/S0010952520301206), [Nature 2018](https://www.nature.com/articles/s41467-018-06647-0) |
| **Glymphatic clearance** | Sleep boosts interstitial fluid clearance of metabolites ("brain washing"). | [Physiology Journals 2022](https://journals.physiology.org/doi/full/10.1152/physiol.00023.2021), [PMC 2018](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5867279/) |
| **NREM parasomnias** | Complex behaviors (e.g. sleepwalking) emerge from state dissociation; partial wake systems allow action without narrative. | [PubMed 2021](https://pubmed.ncbi.nlm.nih.gov/33568303/), [Nature 2014](https://www.nature.com/articles/nrn3835) |

---

## 3. Proposed EQNet Latents & Passes
### 3.1 Arousal Gate (orexin_like)
- **Inputs:** shadow_uncertainty, replay prediction errors, pdc_story (prospective drive).
- **Output:** scalar 0..1 controlling policy complexity (temperature/top_p, candidate count) & TalkMode aggression.
- **Usage:** integrate into `_apply_shadow_controls()` to bias toward WATCH/ASK when arousal is high.

### 3.2 Consolidation Pass (spindle_like_score)
- Identify repeated action patterns (successful ASK→TALK, veto successes, etc.).
- Score "spindle-like" coupling (consistency + decreasing error).
- Record into a procedural memory lane (e.g., MemoryMosaic with a `skill_signature`).

### 3.3 Clearance Pass (cognitive_debris)
- Treat prolonged high-uncertainty shadow entries as “debris”.
- Nightly pass removes low-impact entries and keeps statistical summaries.
- Set next-day TalkMode/TEMP bias based on unresolved debris (promotes WATCH when over-loaded).

### 3.4 State Dissociation Monitor
- Compute `state_dissociation = f(high arousal, high uncertainty, forced TALK)`.
- When high → auto ASK/WATCH + log “near-parasomnia” event for audits.

---

## 4. Implementation Path
1. **Runtime:** add `orexin_like` latent → feed into shadow controls and talk modes.
2. **Nightly:** add consolidation & clearance passes (hooks into existing nightly runner).
3. **Telemetry:** log orexin-like values, spindle-like scores, cognitive_debris trends, state-dissociation warnings.
4. **Validation:** ablation tests (orexin_on/off, clearance strong/weak) and metrics (next-day uncertainty recovery, misbehavior rate).

---

## 5. Notes
- Existing Shadow → Policy/TalkMode pipeline already downshifts arousal; orexin_like makes it physiologically grounded.
- Consolidation vs clearance ensures nightly logs can be interpreted as pseudo sleep stages (N2 for skills, N3 for cleanup).
- State dissociation detection prevents “sleepwalking” style outputs (actions without narratives).

