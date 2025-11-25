# EQNet Mask Layer & Persona Design

## 1. Goals
- Preserve the existing DNA -> PDC -> Replay backbone.
- Keep inner (Phi/Psi snapshots + PDC state) immutable and loggable.
- Allow outer expressions to change per persona/mask without corrupting inner mood.
- Record inner/outer mismatches (	ension) so masking/acting stays transparent.

## 2. Runtime insertion point
`
Perception -> Phi/Psi -> PDC -> Replay
          ->
      inner_spec  (snapshots + m_t, m_past_hat, m_future_hat, E_story, T, jerk)
          ->
      MaskLayer.build_prompt(inner_spec, persona_cfg, dialog_context)
          ->
      LLM -> outer_text / controls
          ->
      EQFrame logging -> TTS / gesture / prosody
`
No changes to Phi/Psi/PDC/Replay. MaskLayer only shapes the LLM prompt and outer presentation.

## 3. EQFrame schema (one per turn)
| Field | Description |
|-------|-------------|
| phi_snapshot, psi_snapshot | Raw inner field vectors |
| pdc_mood, m_past_hat, m_future_hat | Prospective state |
| E_story, T, jerk_p95 | PDC scalars for guard/tension |
| persona_id, masking_strength, masking_motives, persona_source | Persona metadata |
| outer_text, outer_controls | Final response + behaviour knobs |
| mood_align, 	ension_score | Cosine similarity between affect embedding of outer_text and pdc_mood |
| inner_hash, frame_hash, prev_hash | Optional integrity guards |

This gives a "heart frame" that can be audited after the fact.

## 4. MaskLayer responsibilities
1. Receive inner_spec + persona config + dialog context.
2. Produce an LLM system prompt that says "inner mood is X, masking_strength=k, motives=[...]".
3. Optionally inject seed phrases (tsun/deredere, business, caretaker).
4. Return the final LLM prompt + any post-processing hooks (e.g. line rewrite rules).
MaskLayer never touches Phi/Psi/PDC state; it only controls expression.

## 5. Tension computation
`
e_outer = affect_embed(outer_text)    # valence/arousal/affinity vector
m_inner = pdc_mood
mood_align = cos(e_outer, m_inner)
tension = 1.0 - mood_align
`
High tension = strong masking, low tension = inner ? outer. Log both values and expose aggregates in dev dashboards.

## 6. Persona parameter space
| Parameter | Effect |
|-----------|--------|
| masking_strength (0-1) | Fraction of inner mood suppressed before reaching outer layer |
| masking_style (inhibit, 
everse, delay, ...) | Which display rule to apply |
| leak_coefficient | Probability that inner slips through even when masked |
| delay_time | Number of turns before mask relaxes |
| inverse_reaction_prob | Chance of responding with the opposite polarity (tsun) |
| distance_bias | Preferred interpersonal distance (keigo vs tameguchi) |

## 7. Time-function design
### 7.1 Mood dynamics (inner)
Use first-order response:
`m_{t+1} = m_t + (Δt/τ) * (target - m_t)`

Separate τ for rising/falling to model "easy to excite, slow to cool".

### 7.2 Vocabulary gating (outer)
Replace step thresholds with sigmoid gates:
`P_neutral(m) = σ(-k (m - θ))`

θ = midpoint where neutral speech declines, k = steepness.

### 7.3 Parameter tuning workflow
1. **Hand tune** τ/θ/k using human intuition (e.g. half-way mood shift in ~10 turns).
2. **Collect feedback** ("too cold", "just right") and run CMA-ES/Bayesian search on logs.
3. **Optional user preference adaptation**: Use session length/return rate as implicit signals to nudge τ/k per user.

## 8. User-facing controls
- Level 1: Presets only (default, gentle, business, tsun).
- Level 2: 3–4 meaning-labelled sliders ("mood agility", "candour", "intensity", "distance").  
  Each slider maps to τ, masking_strength, T, distance_bias internally.
- Level 3 (dev): direct access to τ, θ, k, masking params with live graphs.

## 9. Transparency & APIs
- Dev/audit API: `GET /eqnet/frame/{id}` returns inner/spec/persona/outer/tension.
- User-facing meta-messages: translate tension spikes into gentle self-disclosures  
  (e.g. 「さっきはちょっと強がりすぎていたかもしれないね」).
- Persona changes must carry `persona_source` (`user` / `app` / `system`) for traceability.


## 10. Integration checklist
- [ ] Implement inner_spec builder in EmotionalHubRuntime.
- [ ] Add MaskLayer with persona presets, inject into LLM prompt path.
- [ ] Compute 	ension per response and store EQFrame.
- [ ] Expose Prospective + Persona state in Gradio dev tab.
- [ ] Add user presets + labelled sliders, keep raw params hidden.
- [ ] Wire feedback buttons ("too cold" / "great") to log for later tuning.

## 11. Trait → Regulation → Expression Layers
- **Trait layer** (psychological & cognitive factors) captures emotional stability, attentional bias, neuromodulator levels, etc.
- **Regulation layer** (behaviour regulation parameters) translates traits into control values such as τ, T_base, masking_baseline, replay_spread, guard_threshold, meta_rate.
- **Expression mapping layer** routes each regulation parameter to the concrete modules (Φ/Ψ updates, PDC gains, Replay weights, MaskLayer prompts, Guard thresholds, LLM style hints).
- This separation keeps personality definitions interpretable (traits), while code reads only regulation parameters, and Runtime documents which module each mapping influences.

## 12. Cognitive/Psychological Parameter Mapping

### 11.1 Cognitive Style
- **Attentional bias**（外界志向 vs 内界志向、ポジティブ vs ネガティブ）  
  → どの入力が Φ/Ψ/PDC の更新や Replay の重みに強く影響するかを調整。
- **Processing speed (τ)**  
  → `m_{t+1} = m_t + (Δt/τ)(target - m_t)` における時間定数。
- **Cognitive load capacity**  
  → jerk / E_story のガード閾値を変化させる（負荷に弱い／強い）。

### 11.2 Personality Factors (Big Five inspired)
- **Emotional stability**  
  → Φ の揺れ幅（variance）、tensionガードの入りやすさ。
- **Extraversion**  
  → 温度 T のベースライン、発話テンポ。
- **Agreeableness**  
  → mask の“柔らかさ”傾向（soft expressions の比率）。
- **Conscientiousness**  
  → persona（役割）の維持強度。
- **Openness**  
  → 未来リプレイの広がり（future replay spread）。

### 11.3 Neuromodulators
- **Dopamine**  
  → novelty 重み γ、探索半径（exploration radius）。
- **Serotonin**  
  → masking_strength のベース値（高いほど抑制・安定）。
- **Noradrenaline**  
  → T と jerk の反応性（緊張・覚醒）。
- **Oxytocin**  
  → Love軸ゲイン、リーク係数（気持ちが漏れやすくなる）。

### 11.4 Self-model parameters
- **Reflectivity（内省性）**  
  → tension 上昇時のメタ発話（「ちょっと強がったかも」）の出やすさ。
- **Narrativity（物語性）**  
  → Replay / 未来想起が現在 mood に影響する度合い。
- **Self–other distance（自己−他者距離）**  
  → 語彙セット（敬語 vs タメ口）、距離感の出し方。
- **Role awareness（役割意識）**  
  → 恋人／アシスタント／ケアテイカーなどの persona 自動切替傾向。

### 11.5 Mapping summary

| Parameter             | Primary hook                                |
|----------------------|----------------------------------------------|
| Emotional stability  | Φ variance、tension guard                    |
| Extraversion         | Temperature baseline、話し方のリズム         |
| Openness             | Future replay spread                         |
| Dopamine level       | novelty weight γ                             |
| Serotonin level      | masking_strength baseline                    |
| Oxytocin level       | Love-axis gain、leak coefficient             |
| Reflectivity         | Meta-expression rate                         |
| Self–other distance  | Speech style（敬語／タメ口）                 |
