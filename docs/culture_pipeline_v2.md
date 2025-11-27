Culture Pipeline v2

This note captures the in-flight redesign that moves the culture system away from the legacy affective_log pipeline. It maps the runtime → logging → Nightly → diary/translator flow and records the two-layer field model (climate + monuments) plus the upcoming work required to expose it everywhere.

1. Runtime logging contract

Every MomentLog entry carries culture context.

CultureContext dataclass (or TypedDict) with culture_tag, place_id, partner_id, and optional object_id, object_role, activity_tag.

Default fallbacks: culture_tag="default", place_id="unknown_place", partner_id="solo". Missing tags never block climate updates, but monument promotion can optionally require at least one concrete anchor.

Hub decides the tag. EmotionHub.runtime._current_culture_tag() first consults recipient_profile.culture_tag, then heuristics; it is stored on each MomentLog entry.

Moment writers feed the field model. Whenever a MomentLog entry is emitted:

Call update_climate_from_event(event, context) to update EMA states.

Call promote_to_monument_if_needed(event, transcript_text, recurrence_count) so high-salience events can enter the memory palace.

Diary + translator inputs keep the context. Whatever invokes compute_culture_state later on should pass the same CultureContext that the runtime stored, so tone adjustments stay consistent across async jobs.

2. CultureClimate ("the weather")

Implementation: defaultdict(str -> ClimateState) for culture/place/partner buckets (extendable to activity/time-band).

ClimateState tracks the EMA of valence, arousal, intimacy, politeness, rho, plus n and last_ts for decay.

Update rule: time-decayed EMA (half-life ≈ 72h, tunable) per dimension, called on every MomentLog event.

Purpose: provide the slow-moving baseline that reflects recent mood per tag.

Storage: in-memory for now, but wrap behind a small storage backend so SQLite/JSONL/Postgres swaps are trivial later.

3. MemoryMonuments ("the landmarks")
Structure
@dataclass
class MemoryMonument:
    id: str
    kind: MonumentKind  # social / personal / fiction
    place_id: Optional[str]
    partner_id: Optional[str]  # may be None or "fic:anime:character"
    culture_tag: Optional[str]
    object_id: Optional[str]   # anime:xxx, craft:yyy, etc.
    object_role: Optional[str]
    valence: float; arousal: float; intimacy: float; politeness: float; rho: float
    salience: float            # 0..1
    created_ts: float
    replay_count: int = 0


MonumentKind covers:

SOCIAL: real partner present (current behavior).

PERSONAL: partnerless but emotionally intense crafting/growth/solitary scenes.

FICTION: fictive partner or scene (partner_id="fic:work:character").

Promotion logic

Score = weighted mix of |valence| + arousal, text signals（例: 「すごく嬉しい」「胸が熱い」「心に残る」など）、recurrence count、optional explicit user pin.

infer_monument_kind(event, text) uses partner presence + object tags + language cues to pick SOCIAL vs PERSONAL vs FICTION.

Duplicate detection: match on (place_id, partner_id, culture_tag, object_id, kind); bump replay_count and push salience → 1.0 when revisited.

Rescan hook: rescan_events_for_monuments(events) allows future threshold changes without losing history.

4. Runtime composition (culture → behavior)

Base climate: fetch climate states for culture_tag, place_id, partner_id and average (_avg3).

Monument spike: query monuments by matching place/partner/culture/object/kind, weight by salience×replay. Aggregate into a spike_state.

Blend: final = (1 - w) * base + w * spike, where w ∈ [0, 0.9] grows with monument salience (no hit → w=0).

Behavior mod: map CultureState to tone/politeness/empathy/joke ratio so diaries, translators, and EmotionTranslator prompts gain the culture bias automatically.

5. Downstream consumers

Nightly culture_stats & culture_history: Already refactored to read MomentLog first. They now capture counts/means per culture tag and feed culture_stats.json + culture_history.jsonl.

Diary bullets: When assembling diary_payload, load daily_culture, pick top tags (by samples/|valence|/rho), and add one-line bullets such as
「family の文脈では落ち着いた気分が続いていた」
「work の文脈では集中力が高かった」 など。

EmotionTranslator tone: Pass BehaviorMod knobs into LLM style instructions (e.g., higher politeness on work, softer tone on family).

Recipient profile: recipient_profile.culture_tag seeds _current_culture_tag; switching dictionaries brings existing cohorts along.

6. Migration + ops checklist

Deprecate culture_logger. Mark as legacy, keep for archival reads only. Nightly already prioritizes moment_log_path.

Old logs: Either leave “pre-cutover = v1” or run a one-shot converter (affective_log.jsonl -> MomentLog-lite) before backfilling culture_history.

Backfill monuments: run the rescan hook on archived MomentLogs so historical highlights populate the palace.

Diary/translator rollout:

Step 1: include culture_summary in daily payloads.

Step 2: add bullet heuristics + translator tone tweaks.

Step 3: tighten EmotionTranslator prompts with simple rules (e.g., work → polite, family → intimate).

Monument hygiene: document how to pin/unpin via user phrases（例: 「これは大事に残したい」）, and ensure there is a maintenance job to trim low-salience monuments if storage needs it.

7. Open questions / next steps

Storage backend selection (SQLite vs JSONL) for climates & monuments.

Formal schema for object_id / object_role so anime/craft/music scenes share vocabulary.

Threshold tuning for PERSONAL/FICTION monument promotion and diarized bullet limits (e.g., max 2 tags per day).

Integrating Σ/Ψ gating so CultureState also influences mood gain scheduling.

Use this document as the reference when wiring eqnet/culture_model.py, runtime logging hooks, and diary/translator consumers. It should stay close to the implementation so engineers can ship without re-deriving the design from chat logs.## Status Snapshot

**Implemented**
- scripts/backfill_culture_monuments.py rescans any MomentLog JSONL and emits a JSON dump of the resulting monuments, so pre-cutover history can be migrated with a single command.
- Each MomentLog entry now stores behavior_mod (tone/empathy/directness/joke_ratio); runtime also injects the same payload into the LLM context, which makes downstream validation/analytics trivial.
- DiaryEntry.culture_summary is surfaced in scripts/diary_viewer.py and scripts/export_sqlite.py, so both the textual viewer and SQLite exports carry the one-line culture commentary for dashboards or clients. observer_markdown can render the same lines when state["culture_summary"] is provided.

**Next work**
- Wire a persistent storage backend for climates/monuments (current module is in-memory) once rollout freezes.
- Add lightweight analytics (histograms, alerts) on the new behavior_mod field or feed culture_summary into observer/PR tools automatically.
- Continue schema cleanup for object_id/object_role and finalize the ops story for trimming/merging monuments.
