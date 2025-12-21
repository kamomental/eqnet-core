# -*- coding: utf-8 -*-
"""Composable hub orchestrator with interchangeable inference backends."""

from __future__ import annotations

import hashlib
import json
import logging
import math
import random
import time
import uuid
from pathlib import Path
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
)

import yaml
from terrain import llm as terrain_llm

from .engine_bdh import BDHEngine
from .engine_transformer import TransformerEngine
from .inference import InferenceEngine
from ..mind.credit import apply_credit_updates
from ..mind.green import green_response
from ..mind.mood_gate import mood_controls
from ..mind.replay_unified import UnifiedReplayKernel
from ..mind.self_model import SelfModel
from ..mind.interference_gate import InterferenceGate
from ..community.sync import advance_phases, resonance_metrics
from ..core.intero_bus import InteroBus
from ..core.modulation import apply_coupling_with_limits
from ..text.filler_policy import FillerPolicy
from ..text.filler_inserter import (
    insert_fillers,
    to_placeholder,
    scan_protected_regions,
)
from ..memory.go_sc import GoSCWeights, StreamingPercentiler, compute_go_sc
from ..memory.ttl import MemoryTTLManager
from ..mind.metamemory import MetaMemory
from ..mind.thought_extractor import ThoughtExtractor
from ..mind.thought_assigner import ThoughtAssigner
from ..mind.thought_bus import ThoughtBus
from .autopilot import Autopilot, AutopilotCfg
from .qualia_bridge import infer_features_from_text, measure_language_loss
from ..causality.do_graph import DOgraph
from ..mind.self_nonself import NarrativePosterior, RolePosterior, kld as role_kld
from ..sense.envelope import SenseEnvelope, clamp_features
from ..sense.gaze import GazeSummary, extract_gaze_features
from ..sense.residuals import compute_residual
from ..social.group_attention import GroupAttention
from ..social.interest_tracker import InterestTracker
from ..nlg.disclosure import craft_payload, decide_disclosure
from .mode_hysteresis import ModeHysteresis
from ..core.biofield_schema import validate_bio_cfg
from ..tts.renderer import render_for_tts
from ..culture.norms import deontic_gate
from ..policy.arbiter import choose_action
from ..safety.guard import preflight_guard
from ..persona.selector_bayes import BayesPersonaSelector
from ..social.tom import peer_affect_hint
from ..safety.bayes_gate import BayesSafetyGate
from ..utils.receipt import build_receipt, log_receipt
from ..utils.referent import sanitize_referent_label
from .persona_manager import PersonaManager
from .replay_executor import ReplayExecutor
from .solver_manager import SolverManager
from .forgetting_controller import ForgettingAdvice, ForgettingController
from .receipt_composer import augment_receipt, diff_controls, verify_contributions
from ..metrics.composition import assoc_defect
from ..metrics.naturality import naturality_residual, residual_components
from ..metrics.sensitivity import simple_sensitivity
from .safety_orchestrator import SafetyOrchestrator
from ..timekeeper import TimeKeeper
from ..persona.prefs import apply_prefs_to_cfg, load_prefs
from ..safety.acl import load_acl
try:
    from ..feedback.event_bus import record_event as record_feedback_event
    from ..feedback.processor import apply_feedback as apply_feedback_event
except Exception:
    def record_feedback_event(*args, **kwargs):
        return None
    def apply_feedback_event(*args, **kwargs):
        return None
from ..ops.task_profiles import TASK_PROFILES
from ..ops.task_fastpath import summarize_task_fastpath
from ..utils.fastpath_config import load_fastpath_defaults, load_fastpath_overrides
from devlife.mind.replay_memory import ReplayMemory, ReplayTrace



class _LMStudioCallable:
    def __call__(self, prompt: str) -> str:
        system_prompt = "You are a helpful assistant. Keep responses concise."
        try:
            return terrain_llm.chat_text(system_prompt, prompt, temperature=0.6, top_p=0.9) or ""
        except Exception:
            return ""

class Hub:
    """High-level coordinator connecting planning, inference, and logging."""

    logger = logging.getLogger(__name__)

    def __init__(self, cfg: Dict[str, Any]) -> None:
        prefs_path = cfg.get("prefs_path", "config/prefs/user_default.yaml")
        prefs = load_prefs(prefs_path)
        merged_cfg = apply_prefs_to_cfg(prefs, cfg)
        self.cfg = merged_cfg
        cfg = self.cfg
        hub_cfg = self.cfg.get("hub", {})
        self.mode = str(hub_cfg.get("mode", "Supportive"))
        baseline_heartiness = float(hub_cfg.get("heartiness", 0.2))
        self.engine = self._select_engine(cfg)
        self.timekeeper = TimeKeeper(
            base_rate=cfg.get("time", {}).get("base_rate", 1.0)
        )
        self.replay_cfg = cfg.get("replay", {})
        phase_cfg = self.replay_cfg.get("phase", {}) or {}
        thresholds_cfg = phase_cfg.get("direction_thresholds", {}) or {}
        self._replay_direction_thresholds = {
            "reverse": float(thresholds_cfg.get("reverse", 0.6)),
            "forward": float(thresholds_cfg.get("forward", 0.4)),
        }
        self.value_weights = cfg.get(
            "value_weights",
            {
                "extrinsic": 0.40,
                "novelty": 0.08,
                "social": 0.18,
                "coherence": 0.12,
                "homeostasis": 0.15,
                "qualia_fit": 0.07,
                "norm_penalty": 0.55,
            },
        )
        self.replay_kernel = UnifiedReplayKernel(
            c_ucb=self.replay_cfg.get("c_ucb", 1.2),
        )
        self.replay_memory = ReplayMemory()
        self.go_sc_cfg = cfg.get("go_sc", {}) or {}
        score_cfg = self.go_sc_cfg.get("score", {}) or {}
        weights_cfg = score_cfg.get("weights") or self.go_sc_cfg.get("weights") or {}
        percentile_window = int(
            score_cfg.get(
                "percentile_window", self.go_sc_cfg.get("percentile_window", 2048)
            )
        )
        if weights_cfg:
            self._go_sc_weights = GoSCWeights(
                weights=dict(weights_cfg), percentile_window=percentile_window
            )
            self._go_sc_percentiler = StreamingPercentiler(
                self._go_sc_weights.percentile_window
            )
        else:
            self._go_sc_weights = None
            self._go_sc_percentiler = None
        fastpath_cfg = load_fastpath_defaults()
        runtime_fastpath = cfg.get("fastpath") or {}
        if isinstance(runtime_fastpath, dict):
            fastpath_cfg.update(runtime_fastpath)
        override_fastpath = load_fastpath_overrides()
        if override_fastpath:
            fastpath_cfg.update(override_fastpath)
        self._fastpath_cfg = fastpath_cfg
        self._fastpath_enabled = bool(fastpath_cfg.get("enable", True))
        self._fastpath_mode = str(fastpath_cfg.get("enforce_actions", "record_only")).lower()
        self._fastpath_ab_fraction = float(fastpath_cfg.get("ab_test_fraction", 0.15))
        self._fastpath_ab_fraction = min(1.0, max(0.0, self._fastpath_ab_fraction))
        self._fastpath_ab_scale = float(fastpath_cfg.get("ab_test_ttl_scale", 1.15))
        self._fastpath_rng = random.Random(fastpath_cfg.get("ab_test_seed", 1337))
        requested_profiles = fastpath_cfg.get("profiles", ["cleanup"])
        self._fastpath_profiles = [
            name for name in requested_profiles if name in TASK_PROFILES
        ]
        self.self_model = SelfModel()
        self._last_coherence = self.self_model.coherence()
        self.metrics_cfg = dict(cfg.get("metrics", {}))
        self._refresh_metrics_params()
        self.solver_manager = SolverManager(cfg.get("solver", {}))
        self.forgetting = ForgettingController(cfg.get("forgetting", {}))
        self.memory_ttl = MemoryTTLManager(cfg.get("memory", {}), self.timekeeper)
        self.metamemory_cfg = cfg.get("metamemory", {}) or {}
        self.metamemory = MetaMemory(self.metamemory_cfg, self.replay_memory)
        self.mode_hysteresis = ModeHysteresis(
            self.cfg.get("modes", {}).get("read_only", {})
        )
        self._hysteresis_mode = (
            "read_only" if self.mode.lower() == "read_only" else "supportive"
        )
        self.thought_cfg = dict(cfg.get("thought_bus", {}) or {})
        self.thought_extractor = ThoughtExtractor(
            dim=int(self.thought_cfg.get("dim", 128)),
            ttl_tau_default=float(self.thought_cfg.get("ttl_tau_default", 2.0)),
        )
        self.thought_assigner = ThoughtAssigner(
            sim_th=float(self.thought_cfg.get("sim_threshold", 0.75))
        )
        self.thought_bus = ThoughtBus(self.thought_cfg)
        self.peer_acl = load_acl(cfg.get("peer_acl_path", "config/peers/acl.yaml"))
        autopilot_override = {}
        override_path = Path("config/overrides/autopilot.yaml")
        if override_path.exists():
            try:
                autopilot_override = (
                    yaml.safe_load(override_path.read_text(encoding="utf-8")) or {}
                )
            except Exception:
                autopilot_override = {}
        heartiness_start = float(
            autopilot_override.get("autopilot", {}).get(
                "heartiness_start", baseline_heartiness
            )
        )
        self.heartiness = heartiness_start
        self.autopilot = Autopilot(
            AutopilotCfg(),
            safety_gate=None,
            forgetting=self.forgetting,
            solver_mgr=self.solver_manager,
            thought_bus=self.thought_bus,
            heartiness_start=heartiness_start,
        )
        self.sunyata_cfg = (
            self._load_yaml_config(Path("config/sunyata.yaml")) or {}
        ).get("sunyata", {})
        do_cfg = self.sunyata_cfg.get("do_graph", {})
        self.do_graph = DOgraph(
            tau_ema=float(do_cfg.get("tau_ema", 6.0)),
            contrib_cap=float(do_cfg.get("contrib_max_per_turn", 0.25)),
            normalize_topk=bool(do_cfg.get("normalize_topk", False)),
        )
        do_weights = do_cfg.get("weights")
        if isinstance(do_weights, dict):
            self.do_graph.load_weights(do_weights)
        self._do_profiles = {
            str(k).lower(): v for k, v in (do_cfg.get("profiles") or {}).items()
        }
        self._do_profile_active: Optional[str] = None

        sense_cfg = cfg.get("sense", {}) or {}
        self.sense_enabled = bool(sense_cfg.get("enabled", True))
        self._candor_enabled = bool(sense_cfg.get("candor", {}).get("enable", True))
        self._gaze_cfg = self._load_yaml_config(Path("config/sense/gaze.yaml"))
        self._interest_cfg = self._load_yaml_config(Path("config/social/interest.yaml"))
        interest_cfg = (
            (self._interest_cfg or {}).get("interest") if self._interest_cfg else None
        )
        self._interest_tracker = InterestTracker(interest_cfg) if interest_cfg else None
        group_cfg = (
            (self._interest_cfg or {}).get("group") if self._interest_cfg else None
        )
        if group_cfg:
            weights = group_cfg.get("weights") or {"self": 1.0}
            self._group_attention = GroupAttention(
                weights,
                group_cfg.get("theta", 0.5),
                group_cfg.get("dwell_tau", 1.2),
            )
        else:
            self._group_attention = None
        self._gaze_adapter_cfg = (
            (self._gaze_cfg or {}).get("adapter", {}) if self._gaze_cfg else {}
        )
        referent_cfg = (
            (self.sunyata_cfg or {}).get("referent", {}) if self.sunyata_cfg else {}
        )
        self._referent_hard_domains = set(
            str(domain).lower() for domain in (referent_cfg.get("hard_domains") or [])
        )
        self.interference_gate = InterferenceGate(
            cfg.get("interference", {}),
            timekeeper=self.timekeeper,
        )
        self._sense_share_cfg = self._load_yaml_config(
            Path("config/sense/shareability.yaml")
        )
        self._disclosure_templates = self._load_yaml_config(
            Path("config/nlg/disclosures.yaml")
        )
        self._metaphor_templates = self._load_yaml_config(
            Path("config/nlg/metaphors.yaml")
        )
        self._last_autop_metrics: Dict[str, float] = {}
        self._last_memory_gc_tau = (
            self.timekeeper.tau_now() if hasattr(self.timekeeper, "tau_now") else 0.0
        )
        filler_cfg = cfg.get("filler", {}) or {}
        self.filler_enabled = bool(filler_cfg.get("enabled", False))
        self.filler_bank: Dict[str, Any] = {}
        self.filler_policy: Optional[FillerPolicy] = None
        self._filler_last_tau: Dict[str, float] = {}
        if self.filler_enabled:
            bank_path = filler_cfg.get("bank_path", "config/fillers/filler_bank.yaml")
            bank_file = Path(bank_path)
            if not bank_file.is_absolute():
                bank_file = Path.cwd() / bank_file
            try:
                bank_data = yaml.safe_load(bank_file.read_text(encoding="utf-8")) or {}
            except Exception:
                self.logger.exception("Failed to load filler bank from %s", bank_file)
                bank_data = {}
            if filler_cfg.get("hard_off_domains"):
                bank_data["hard_off_domains"] = filler_cfg["hard_off_domains"]
            if filler_cfg.get("allow_modes"):
                bank_data["allow_modes"] = filler_cfg["allow_modes"]
            self.filler_bank = bank_data
            try:
                self.filler_policy = FillerPolicy(bank_data)
            except Exception:
                self.logger.exception(
                    "Failed to initialise filler policy, disabling fillers"
                )
                self.filler_enabled = False
                self.filler_policy = None
        else:
            self.filler_policy = None
        bayes_cfg = cfg.get("bayes", {})
        persona_modes_cfg = (
            cfg.get("persona_modes") or cfg.get("persona", {}).get("modes") or {}
        )
        if not persona_modes_cfg:
            default_mode = cfg.get("persona", {}).get("mode")
            if default_mode:
                persona_modes_cfg = {default_mode: {}}
        persona_modes = list(persona_modes_cfg.keys())
        if not persona_modes:
            persona_modes = ["caregiver", "playful", "researcher"]
        roles_cfg = (
            (self.sunyata_cfg or {}).get("roles", {}) if self.sunyata_cfg else {}
        )
        self.self_roles = RolePosterior(
            persona_modes,
            temperature=float(roles_cfg.get("temperature", 1.0)),
            halflife_tau=float(roles_cfg.get("halflife_tau", 12.0)),
        )
        self.self_narr = NarrativePosterior()
        uniform = 1.0 / len(self.self_roles.roles)
        self._last_roles_post: Dict[str, float] = {
            role: uniform for role in self.self_roles.roles
        }
        self._clinging_cooldown_until: float = 0.0
        self._clinging_consecutive: int = 0
        bayes_enabled = bool(bayes_cfg.get("enabled", True))
        self._persona_selector = (
            BayesPersonaSelector(
                persona_modes or ["default"],
                half_life_tau=float(bayes_cfg.get("half_life_tau", 6.0)),
            )
            if bayes_enabled and persona_modes
            else None
        )
        self._safety_gate = (
            BayesSafetyGate(
                z=float(bayes_cfg.get("credible_z", 1.64)),
                read_only_th=float(bayes_cfg.get("read_only_th", 0.2)),
                block_th=float(bayes_cfg.get("block_th", 0.4)),
            )
            if bayes_enabled
            else None
        )
        self.persona_manager = PersonaManager(
            persona_cfg=cfg.get("persona"),
            heartiness=self.heartiness,
            bayes_selector=self._persona_selector,
        )
        self.safety_orchestrator = SafetyOrchestrator(
            hard_constraints=self.cfg.get("safety", {}).get("hard_constraints", []),
            bayes_gate=self._safety_gate,
        )
        raw_bio_cfg = cfg.get("biofield", {})
        bio_cfg = self._load_biofield_config(raw_bio_cfg)
        if isinstance(bio_cfg.get("metrics"), Mapping):
            self.metrics_cfg.update(bio_cfg["metrics"])
            self._refresh_metrics_params()
        self.biofield_cfg = bio_cfg
        self.biofield_digest: Optional[str] = None
        if bio_cfg:
            warnings = validate_bio_cfg(bio_cfg)
            if warnings:
                self.logger.warning("biofield configuration warnings: %s", warnings)
            try:
                self.biofield_digest = hashlib.sha256(
                    json.dumps(bio_cfg, sort_keys=True, default=str).encode("utf-8")
                ).hexdigest()[:12]
            except Exception:
                self.biofield_digest = None
        self._previous_controls: Dict[str, float] = {}
        self.intero_bus: Optional[InteroBus] = None
        self.biofield_state_path: Optional[Path] = None
        self.coupling_matrix: Dict[str, Dict[str, float]] = {}
        self.control_limits, self.step_limits, self.jerk_limits = self._parse_limits(
            bio_cfg.get("limits")
        )
        if bio_cfg.get("enabled", True):
            nodes = bio_cfg.get("nodes", ["sensory", "hippo", "pfc", "policy"])
            adjacency = self._parse_adjacency(bio_cfg)
            decay_cfg = bio_cfg.get("decay", {})
            alpha = float(decay_cfg.get("alpha", bio_cfg.get("alpha", 0.3)))
            beta = float(decay_cfg.get("beta_inject", bio_cfg.get("beta", 0.2)))
            self.intero_bus = InteroBus(nodes, adjacency, alpha=alpha, beta_inject=beta)
            state_path = bio_cfg.get("state_path", "state/biofield.json")
            self.biofield_state_path = Path(state_path)
            try:
                state = self.intero_bus.load(self.biofield_state_path)
            except Exception:
                self.logger.exception(
                    "Failed to load biofield state from %s", self.biofield_state_path
                )
                state = None
            if isinstance(state, dict):
                stored_digest = state.get("config_digest")
                if (
                    self.biofield_digest
                    and stored_digest
                    and stored_digest != self.biofield_digest
                ):
                    self.logger.warning(
                        "Biofield state digest mismatch (state=%s, config=%s); resetting field",
                        stored_digest,
                        self.biofield_digest,
                    )
                    self.intero_bus.tonic.clear()
                    for node in self.intero_bus.field:
                        self.intero_bus.field[node] = 0.0
                else:
                    self._apply_offline_decay(state, decay_cfg)
        else:
            self.biofield_state_path = None
        self.coupling_matrix = self._parse_coupling(
            bio_cfg.get("coupling") or self.cfg.get("coupling")
        )
        self._nightly_snapshot: Dict[str, Any] = {}
        self._bootstrap_from_nightly()

        self.replay_executor = ReplayExecutor(
            mode=self.mode,
            heartiness=self.heartiness,
            replay_cfg=self.replay_cfg,
            replay_kernel=self.replay_kernel,
            replay_memory=self.replay_memory,
            self_model=self.self_model,
            timekeeper=self.timekeeper,
            value_weights=self.value_weights,
            logs_cfg=self.cfg.get("logs", {}),
            step_limits=self.step_limits,
        )

    def _select_engine(self, cfg: Dict[str, Any]) -> InferenceEngine:
        hub_cfg = cfg.get("hub", {})
        backend = str(hub_cfg.get("inference", "transformer")).lower()
        if backend == "bdh":
            bdh_cfg = cfg.get("inference", {}).get("bdh", {})
            return BDHEngine(bdh_cfg)
        model = cfg.get("transformer_model")
        tokenizer = cfg.get("transformer_tokenizer")
        if not model:
            model = _LMStudioCallable()
        return TransformerEngine(model, tokenizer)

    def _bootstrap_from_nightly(self) -> None:
        logs_cfg = self.cfg.get("logs", {})
        summary_path = Path(
            logs_cfg.get("nightly_summary_path", "logs/nightly_summary.json")
        )
        if not summary_path.exists():
            return
        try:
            snapshot = json.loads(summary_path.read_text(encoding="utf-8"))
        except Exception:
            return
        self._nightly_snapshot = snapshot
        weights = snapshot.get("value_weights")
        if isinstance(weights, dict):
            for key, value in weights.items():
                if key in self.value_weights:
                    self.value_weights[key] = float(value)
        coherence = snapshot.get("narrative_coherence")
        if coherence is not None:
            try:
                self.self_model.set_baseline(float(coherence))
            except Exception:
                pass
        self.cfg.setdefault("value_weights", {})
        self.cfg["value_weights"].update(self.value_weights)
        if snapshot.get("decay_applied"):
            self.timekeeper.mark("nightly_sync")

    def _refresh_metrics_params(self) -> None:
        self.assoc_threshold = float(
            self.metrics_cfg.get("assoc_defect_threshold", 0.15)
        )
        self.naturality_weights = self.metrics_cfg.get(
            "naturality_weights", {"U": 1.0, "coh": 0.5, "mis": 0.7}
        )
        self.sensitivity_eps = self.metrics_cfg.get(
            "sensitivity_eps", {"temp_mul": 0.02, "directness": 0.01, "pause_ms": 20.0}
        )
        self.jerk_rate_max = float(self.metrics_cfg.get("jerk_rate_max", 0.1))
        warn = self.metrics_cfg.get("naturality_warn_threshold")
        self.naturality_warn_threshold = float(warn) if warn is not None else None

    def _load_biofield_config(self, cfg: Mapping[str, Any]) -> Dict[str, Any]:
        if not cfg:
            return {}
        config_path = cfg.get("config_path")
        if not config_path:
            return dict(cfg)
        path = Path(config_path)
        if not path.is_absolute():
            root = self.cfg.get("config_root")
            if root:
                path = Path(root) / path
            else:
                path = Path.cwd() / path
        try:
            loaded = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        except Exception:
            self.logger.exception("Failed to load biofield config from %s", path)
            return dict(cfg)
        merged = self._deep_merge_dicts(
            loaded, {k: v for k, v in dict(cfg).items() if k != "config_path"}
        )
        merged["config_path"] = str(path)
        return merged

    @staticmethod
    def _deep_merge_dicts(
        base: Mapping[str, Any], overlay: Mapping[str, Any]
    ) -> Dict[str, Any]:
        result = dict(base)
        for key, value in overlay.items():
            if (
                key in result
                and isinstance(result[key], dict)
                and isinstance(value, dict)
            ):
                result[key] = Hub._deep_merge_dicts(result[key], value)
            else:
                result[key] = value
        return result

    def _parse_adjacency(
        self, bio_cfg: Mapping[str, Any]
    ) -> Dict[Tuple[str, str], float]:
        adjacency: Dict[Tuple[str, str], float] = {}
        adj_cfg = bio_cfg.get("adjacency")
        if isinstance(adj_cfg, Mapping):
            for key, weight in adj_cfg.items():
                if isinstance(key, str) and "->" in key:
                    src, dst = key.split("->", 1)
                elif isinstance(key, (list, tuple)) and len(key) == 2:
                    src, dst = key
                else:
                    continue
                adjacency[(str(src).strip(), str(dst).strip())] = float(weight)
        edges = bio_cfg.get("edges", [])
        for edge in edges:
            if isinstance(edge, dict):
                src = edge.get("from")
                dst = edge.get("to")
                weight = edge.get("weight", 0.0)
            elif isinstance(edge, (list, tuple)) and len(edge) == 3:
                src, dst, weight = edge
            else:
                continue
            if not (src and dst):
                continue
            adjacency[(str(src), str(dst))] = float(weight)
        if not adjacency:
            adjacency = {
                ("hippo", "pfc"): 0.05,
                ("pfc", "policy"): 0.03,
                ("hippo", "sensory"): 0.02,
            }
        return adjacency

    def _parse_coupling(
        self, coupling_cfg: Optional[Mapping[str, Any]]
    ) -> Dict[str, Dict[str, float]]:
        default = {
            "inflammation_global": {
                "pause_ms": 120.0,
                "directness": -0.04,
                "steps_bias": -0.6,
            },
            "energy": {"temp_mul": -0.06},
            "uncertainty": {"temp_mul": 0.08, "directness": -0.02},
        }
        if not coupling_cfg:
            return default
        matrix: Dict[str, Dict[str, float]] = {}
        for src, mapping in coupling_cfg.items():
            if not isinstance(mapping, Mapping):
                continue
            stage: Dict[str, float] = {}
            for dst, spec in mapping.items():
                if isinstance(spec, Mapping):
                    gain = spec.get("gain")
                    if isinstance(gain, (int, float)):
                        stage[dst] = float(gain)
                else:
                    try:
                        stage[dst] = float(spec)
                    except (TypeError, ValueError):
                        continue
            if stage:
                matrix[src] = stage
        return matrix or default

    def _parse_limits(
        self, limit_cfg: Optional[Mapping[str, Any]]
    ) -> Tuple[Dict[str, Tuple[float, float]], Tuple[int, int], Dict[str, float]]:
        control_limits: Dict[str, Tuple[float, float]] = {
            "pause_ms": (0.0, 400.0),
            "temp_mul": (0.6, 1.2),
            "directness": (-0.2, 0.2),
            "top_p_mul": (0.5, 1.5),
        }
        step_limits: Tuple[int, int] = (1, 6)
        jerk_limits: Dict[str, float] = {
            "pause_ms": 40.0,
            "directness": 0.03,
            "temp_mul": 0.06,
        }
        if isinstance(limit_cfg, Mapping):
            for key, bounds in limit_cfg.items():
                if key == "jerk" and isinstance(bounds, Mapping):
                    for ctrl, delta in bounds.items():
                        try:
                            jerk_limits[ctrl] = abs(float(delta))
                        except (TypeError, ValueError):
                            continue
                elif (
                    key == "steps"
                    and isinstance(bounds, (list, tuple))
                    and len(bounds) == 2
                ):
                    lo, hi = int(bounds[0]), int(bounds[1])
                    if lo > hi:
                        lo, hi = hi, lo
                    step_limits = (max(1, lo), max(1, hi))
                elif isinstance(bounds, (list, tuple)) and len(bounds) == 2:
                    lo, hi = float(bounds[0]), float(bounds[1])
                    if lo > hi:
                        lo, hi = hi, lo
                    control_limits[key] = (lo, hi)
        return control_limits, step_limits, jerk_limits

    def _apply_offline_decay(
        self, state: Mapping[str, Any], decay_cfg: Mapping[str, Any]
    ) -> None:
        if self.intero_bus is None:
            return
        ts_wall = state.get("ts_wall")
        if ts_wall is None:
            return
        try:
            elapsed = max(0.0, time.time() - float(ts_wall))
        except Exception:
            return
        tau_per_sec = float(decay_cfg.get("offline_tau_per_sec", 0.5))
        tau_cap = float(decay_cfg.get("offline_tau_cap", 3600.0))
        step_dt = float(decay_cfg.get("offline_step_dt", 1.0))
        max_steps = int(decay_cfg.get("offline_max_steps", 7200))
        tau_equiv = min(tau_cap, max(0.0, tau_per_sec) * elapsed)
        if tau_equiv <= 0.0 or step_dt <= 1e-6:
            return
        steps = int(min(max_steps, tau_equiv // step_dt))
        remainder = tau_equiv - steps * step_dt
        for _ in range(steps):
            self.solver_manager.step_biofield(self.intero_bus, step_dt)
        if remainder > 1e-6:
            self.solver_manager.step_biofield(self.intero_bus, remainder)

    def _apply_forgetting_advice(self, advice: ForgettingAdvice) -> None:
        if advice is None:
            return
        try:
            saint = getattr(self.engine, "saint_eryngium", None)
            if saint is not None:
                lstm = getattr(saint, "lstm", None)
                if lstm is not None and hasattr(lstm, "set_forget_bias_delta"):
                    lstm.set_forget_bias_delta(
                        float(advice.lstm.get("forget_bias_delta", 0.0))
                    )
                ssm = getattr(saint, "ssm", None)
                if ssm is not None:
                    if hasattr(ssm, "set_forgetting_params"):
                        ssm.set_forgetting_params(advice.ssm)
                    elif hasattr(ssm, "set_stabilizer"):
                        stabilize = advice.ssm.get("stabilize_tau")
                        if stabilize is not None:
                            try:
                                ssm.set_stabilizer(float(stabilize))
                            except Exception:
                                pass
            if self.intero_bus is not None and hasattr(
                self.intero_bus, "set_decay_params"
            ):
                self.intero_bus.set_decay_params(advice.intero)
            if advice.replay and hasattr(self.replay_executor, "set_forgetting_params"):
                self.replay_executor.set_forgetting_params(advice.replay)
            if advice.persona and hasattr(
                self.persona_manager, "set_forgetting_params"
            ):
                self.persona_manager.set_forgetting_params(advice.persona)
        except Exception:
            self.logger.exception("Failed to apply forgetting advice")

    def _maybe_run_memory_gc(self, tau_now: float) -> Optional[Dict[str, float]]:
        if self.memory_ttl is None:
            return None
        interval = float(self.cfg.get("memory", {}).get("gc_interval_tau", 6.0))
        if interval <= 0.0:
            return None
        if (tau_now - self._last_memory_gc_tau) < interval:
            return None
        events = self.replay_memory.load_all()
        if not events:
            self._last_memory_gc_tau = tau_now
            return {"dropped": 0, "kept": 0, "interval_tau": interval}
        kept, stats = self.memory_ttl.gc(events)
        if kept or stats.get("dropped", 0) > 0:
            try:
                self.replay_memory.rewrite(kept)
            except Exception:
                self.logger.exception("Failed to rewrite replay memory during GC")
        stats = {
            key: float(value) if isinstance(value, (int, float)) else value
            for key, value in stats.items()
        }
        stats["interval_tau"] = interval
        stats["timestamp_tau"] = tau_now
        self._last_memory_gc_tau = tau_now
        return stats

    def _prepare_thought_trace(
        self,
        candidates: List[Dict[str, Any]],
        replay_details: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        trace: Dict[str, Any] = {}
        vectorised: List[Dict[str, Any]] = []
        for cand in (candidates or [])[:5]:
            if not isinstance(cand, dict):
                continue
            summary = cand.get("summary", {}) or {}
            numeric_summary = {}
            for key, value in summary.items():
                if isinstance(value, (int, float)):
                    numeric_summary[str(key)] = float(value)
            vectorised.append(
                {
                    "U": float(cand.get("U", 0.0)),
                    "coherence": float(cand.get("coherence", 0.0)),
                    "summary": numeric_summary,
                    "metadata": cand.get("metadata", {}),
                }
            )
        if vectorised:
            trace["candidates"] = vectorised
            utilities = [entry["U"] for entry in vectorised]
            mean_u = sum(utilities) / len(utilities)
            variance = sum((u - mean_u) ** 2 for u in utilities) / max(
                1, len(utilities)
            )
            spread = max(utilities) - min(utilities)
            entropy = min(1.0, math.sqrt(max(variance, 0.0)))
            trace["entropy"] = entropy
            trace["uncertainty_entropy"] = entropy
            trace["uncertainty_map"] = {"pc1": [mean_u, variance, spread]}
        elif replay_details:
            trace["candidates"] = []
            if "uncertainty" in replay_details:
                try:
                    trace["uncertainty_entropy"] = float(
                        replay_details.get("uncertainty")
                    )
                except Exception:
                    trace["uncertainty_entropy"] = 0.0
        return trace

    def _sanitize_peer_packets(
        self,
        packets: Iterable[Dict[str, Any]],
        *,
        default_origin: str,
        dim_max: int = 256,
        ttl_lo: float = 0.2,
        ttl_hi: float = 12.0,
        allowed_tags: Optional[Iterable[str]] = None,
        norm_max: float = 8.0,
    ) -> List[Dict[str, Any]]:
        allowed_tags_set = set(allowed_tags or ("ephemeral",))
        sanitized: List[Dict[str, Any]] = []
        for packet in packets:
            if not isinstance(packet, Mapping):
                continue
            vec = packet.get("vec")
            if not isinstance(vec, list):
                continue
            try:
                vec_clean = [float(x) for x in vec[:dim_max] if math.isfinite(float(x))]
            except Exception:
                continue
            if not vec_clean:
                continue
            norm = math.sqrt(sum(x * x for x in vec_clean))
            if norm > norm_max:
                scale = norm_max / (norm + 1e-9)
                vec_clean = [x * scale for x in vec_clean]
            kind = str(packet.get("kind", ""))
            if kind not in {"hypothesis", "constraint", "uncertainty"}:
                continue
            tags_raw = packet.get("tags")
            if isinstance(tags_raw, list):
                tags = [str(tag) for tag in tags_raw if str(tag) in allowed_tags_set]
            elif tags_raw is None:
                tags = []
            else:
                tag = str(tags_raw)
                tags = [tag] if tag in allowed_tags_set else []
            ttl_tau = float(
                packet.get("ttl_tau", self.thought_cfg.get("ttl_tau_default", 2.0))
            )
            ttl_tau = max(ttl_lo, min(ttl_hi, ttl_tau))
            sanitized.append(
                {
                    "id": str(packet.get("id", "")),
                    "origin": str(packet.get("origin", default_origin)),
                    "kind": kind,
                    "vec": vec_clean,
                    "entropy": float(packet.get("entropy", 0.3)),
                    "ttl_tau": ttl_tau,
                    "tags": tags,
                    "created_tau": float(packet.get("created_tau", 0.0)),
                }
            )
        return sanitized

    def _apply_jerk_limits(
        self, controls: Mapping[str, float | int]
    ) -> Dict[str, float | int]:
        if not self.jerk_limits:
            return dict(controls)
        smoothed: Dict[str, float | int] = dict(controls)
        for key, max_step in self.jerk_limits.items():
            if key not in smoothed:
                continue
            prev = self._previous_controls.get(key)
            if prev is None:
                continue
            max_delta = abs(float(max_step))
            new_val = float(smoothed[key])
            lower = prev - max_delta
            upper = prev + max_delta
            smoothed[key] = max(lower, min(upper, new_val))
        for key, (lo, hi) in self.control_limits.items():
            if key in smoothed:
                smoothed[key] = max(lo, min(hi, float(smoothed[key])))
        if "pause_ms" in smoothed:
            smoothed["pause_ms"] = int(round(float(smoothed["pause_ms"])))
        return smoothed

    def _update_intero_signals(
        self, ctx_time: Dict[str, Any], d_tau: float
    ) -> Dict[str, float]:
        if self.intero_bus is None:
            return {}
        intero = ctx_time.get("intero", {}) or {}
        energy = intero.get("energy")
        if energy is not None:
            self.intero_bus.publish("energy", float(energy))
        if ctx_time.get("misfire", False):
            self.intero_bus.publish("inflammation", 0.5, node="hippo")
        d_u = float(ctx_time.get("dU_est", 0.0))
        if abs(d_u) > 0.3:
            node = "pfc" if d_u >= 0 else "policy"
            self.intero_bus.publish("inflammation", min(1.0, abs(d_u)), node=node)
        self.solver_manager.step_biofield(self.intero_bus, d_tau)
        signals = self.intero_bus.effective()
        if energy is not None:
            signals.setdefault("energy", float(energy))
        else:
            signals.setdefault("energy", 0.5)
        signals.setdefault("uncertainty", float(ctx_time.get("uncertainty", 0.3)))
        return signals

    def plan(self, ctx: Dict[str, Any]) -> Dict[str, Any]:
        prompt = ctx.get("prompt", "")
        base_controls = {
            "temp_mul": 1.0,
            "top_p_mul": 1.0,
            "pause_ms": 0,
            "directness": 0.0,
        }
        mood = dict(ctx.get("mood", {"v": 0.0, "a": 0.5, "u": 0.3}))
        style = ctx.get("style", "chat_support")
        peers = ctx.get("peers", [])
        if peers and self.mode.lower() in {"reflective", "living"}:
            r_metrics = resonance_metrics(peers)
            r_val = r_metrics.get("r")
            if not (r_val is not None and r_val > 0.8):
                hint = peer_affect_hint(peers[0])
                mood = _apply_mood_hint(mood, hint)
        controls = mood_controls(base_controls, mood, self.heartiness, style=style)
        gaze_mod = ctx.get("gaze_modulation")
        if isinstance(gaze_mod, Mapping):
            controls["directness"] += float(gaze_mod.get("directness_add", 0.0))
            controls["pause_ms"] += int(gaze_mod.get("pause_ms_add", 0))
            if "ask_bias" in gaze_mod and "gaze_ask_bias" not in ctx:
                ctx["gaze_ask_bias"] = gaze_mod.get("ask_bias")
        control_stages: Dict[str, Dict[str, float]] = {"base": dict(controls)}

        tau_rate = float(ctx.get("time", {}).get("tau_rate", 1.0))
        if self.mode.lower() in {"reflective", "living"}:
            qualia = ctx.get("qualia", {})
            green_cfg = self.cfg.get("green", {})
            culture_res = float(green_cfg.get("culture_resonance", 0.3))
            green_resp = green_response(
                qualia,
                culture_resonance=culture_res,
                culture_kernel=green_cfg.get("axis_gain"),
                memory_trace=ctx.get("green_memory"),
                affect_state=mood,
                tau_rate=tau_rate,
            )
            controls["directness"] += green_resp.controls.get("directness_add", 0.0)
            controls["pause_ms"] += int(green_resp.controls.get("pause_ms_add", 0))
            if "warmth_add" in green_resp.controls:
                controls["warmth_bias"] = (
                    controls.get("warmth_bias", 0.0) + green_resp.controls["warmth_add"]
                )
            if "exploration_bias" in green_resp.controls:
                controls["exploration_bias"] = (
                    controls.get("exploration_bias", 0.0)
                    + green_resp.controls["exploration_bias"]
                )
            mood = _apply_green_delta(mood, green_resp.delta_mood)
        else:
            green_resp = None

        controls, persona_payload, chosen_mode = self.persona_manager.prepare(
            ctx, controls
        )
        control_stages["persona"] = dict(controls)

        plan = {
            "prompt": prompt,
            "controls": controls,
            "mood": mood,
            "style": style,
        }
        norms_cfg = self.cfg.get("norms", {"politeness": 0.6, "humility": 0.3})
        plan = deontic_gate(plan, norms_cfg)
        plan["norms_cfg"] = norms_cfg
        control_stages["norms"] = dict(plan["controls"])
        if green_resp is not None:
            plan["green"] = {
                "delta": green_resp.delta_mood,
                "controls": green_resp.controls,
                "qualia_vector": green_resp.qualia_vector,
            }
            control_stages["green"] = dict(plan["controls"])
        else:
            control_stages["green"] = dict(plan["controls"])
        if persona_payload:
            plan["persona"] = persona_payload
        if chosen_mode:
            plan["chosen_mode"] = chosen_mode
        plan["_control_stages"] = control_stages
        return plan

    def run(self, ctx: Dict[str, Any]) -> Dict[str, Any]:
        time_info = self.timekeeper.tick(ctx.get("mood"), ctx.get("intero", {}))
        ctx_time = dict(ctx)
        ctx_time["time"] = time_info
        sense_envelope: Optional[SenseEnvelope] = None
        gaze_summary: Optional[GazeSummary] = None
        if self.sense_enabled:
            sense_envelope, gaze_summary = self._build_sense_envelope(ctx_time)
            if sense_envelope is not None:
                ctx_time.setdefault("sense_envelope", sense_envelope.to_dict())
        else:
            ctx_time.pop("gaze_modulation", None)
            ctx_time.pop("gaze_focus", None)
            ctx_time.pop("referent_hint", None)
        ctx_time["mode_default"] = self.mode
        tau_now = float(time_info.get("tau", self.timekeeper.tau_now()))
        memory_gc_stats = self._maybe_run_memory_gc(tau_now)
        d_tau = float(time_info.get("d_tau", 0.0))
        raw_peers = ctx.get("peers", [])
        if raw_peers:
            coupling = float(self.replay_cfg.get("sync_coupling", 0.15))
            ctx_time["peers"] = advance_phases(
                raw_peers, d_tau=d_tau, coupling=coupling
            )
        else:
            ctx_time["peers"] = raw_peers
        domain_current = str(ctx_time.get("domain", "")).lower()
        self._apply_do_profile(domain_current)

        gaze_context: Optional[Dict[str, Any]] = None
        if gaze_summary is not None:
            gaze_context = self._update_gaze_context(gaze_summary, time_info, ctx_time)
        else:
            ctx_time.pop("gaze_modulation", None)
            ctx_time.pop("gaze_focus", None)
            ctx_time.pop("referent_hint", None)
        sunyata_metrics: Dict[str, Any] = {}
        roles_post: Dict[str, float] = {}
        narrative_coherence = 1.0

        autop_pre = {
            "mode": self.mode,
            "msv_k_min": None,
            "tb_gain_cap": None,
            "reason": "disabled",
        }
        if hasattr(self, "autopilot") and self.autopilot is not None:
            try:
                autop_pre = self.autopilot.pre_plan(
                    self._last_autop_metrics, dict(ctx_time)
                )
            except Exception:
                autop_pre = {
                    "mode": self.mode,
                    "msv_k_min": None,
                    "tb_gain_cap": None,
                    "reason": "error",
                }
        self.mode = str(autop_pre.get("mode", self.mode))
        msv_value = autop_pre.get("msv_k_min")
        if msv_value is not None and hasattr(self, "replay_executor"):
            try:
                self.replay_executor.set_min_steps(msv_value)
            except Exception:
                pass
        tb_cap = autop_pre.get("tb_gain_cap")
        if tb_cap is not None and hasattr(self.thought_bus, "set_gain_cap"):
            try:
                self.thought_bus.set_gain_cap(tb_cap)
            except Exception:
                pass
        ctx_time["mode"] = self.mode

        plan = self.plan(ctx_time)
        if gaze_context:
            plan.setdefault("gaze_focus", gaze_context)
            if "referent_hint" in ctx_time:
                plan.setdefault("referent_hint", ctx_time["referent_hint"])
        referent_hint = ctx_time.get("referent_hint")
        domain = str(ctx_time.get("domain", "")).lower()
        if (
            referent_hint
            and domain not in self._referent_hard_domains
            and isinstance(plan.get("prompt"), str)
        ):
            confidence = float(referent_hint.get("confidence", 0.0))
            label = referent_hint.get("label") or referent_hint.get("object_id")
            if label and confidence >= 0.6:
                prefix = f"【参照】相手はいま「{label}」を見ています（指さし無しの“これ”参照）。\n"
                plan["prompt"] = prefix + plan["prompt"]
        session_id = str(
            ctx_time.get("session_id")
            or ctx_time.get("conversation_id")
            or ctx_time.get("episode_id")
            or ctx_time.get("user_id")
            or "default"
        )

        metamemory_snapshot: Optional[Dict[str, Any]] = None
        metamemory_tot_flag = 0.0
        hysteresis_event: Optional[Dict[str, Any]] = None
        meta_ctx: Dict[str, Any] = {}
        if self.metamemory is not None and bool(
            self.metamemory_cfg.get("enabled", True)
        ):
            cue_payload: Optional[Mapping[str, Any] | str] = ctx.get("metamemory_cue")
            if cue_payload is None:
                meta_req = ctx.get("metamemory", {})
                if isinstance(meta_req, Mapping):
                    cue_payload = meta_req.get("cue")  # type: ignore[assignment]
            if isinstance(cue_payload, str):
                cue: Optional[Dict[str, Any]] = {"text": cue_payload}
            elif isinstance(cue_payload, Mapping):
                cue = dict(cue_payload)
            else:
                prompt_text = ctx.get("prompt") or ctx.get("last_user") or ""
                cue = (
                    {"text": prompt_text}
                    if isinstance(prompt_text, str) and prompt_text
                    else None
                )
            meta_out = None
            if cue and cue.get("text"):
                try:
                    meta_out = self.metamemory.estimate(cue, tau_now)
                except Exception:
                    meta_out = None
            if meta_out is not None:
                metamemory_snapshot = {
                    "F": float(meta_out.F),
                    "R": float(meta_out.R),
                    "FOK": float(meta_out.FOK),
                    "TOT": bool(meta_out.TOT),
                    "clue": meta_out.clue or {},
                }
                meta_ctx = {
                    "F": float(meta_out.F),
                    "R": float(meta_out.R),
                    "FOK": float(meta_out.FOK),
                    "clue": meta_out.clue or {},
                    "tot_active": bool(meta_out.TOT),
                }
                if meta_out.TOT:
                    tot_cfg = self.metamemory_cfg.get("tot_mode", {}) or {}
                    meta_ctx["horizon_cap"] = int(tot_cfg.get("horizon_max", 2))
                    meta_ctx["force_read_only"] = bool(tot_cfg.get("read_only", True))
                    meta_ctx["cooldown_tau"] = float(
                        self.metamemory_cfg.get("cooldown_tau", 0.9)
                    )
                    metamemory_tot_flag = 1.0
                else:
                    metamemory_tot_flag = 0.0
                ctx_time["metamemory"] = dict(meta_ctx)
                if meta_ctx.get("clue"):
                    ctx_time["metamemory_clue"] = meta_ctx["clue"]
                plan["metamemory"] = dict(meta_ctx)
            else:
                ctx_time["metamemory"] = {"tot_active": False}
                plan["metamemory"] = {"tot_active": False}
        else:
            ctx_time["metamemory"] = {"tot_active": False}
            plan["metamemory"] = {"tot_active": False}

        control_stages = dict(plan.pop("_control_stages", {}))
        control_stages.setdefault("base", dict(plan["controls"]))
        bio_signals = (
            self._update_intero_signals(ctx_time, d_tau)
            if self.intero_bus is not None
            else {}
        )
        if not bio_signals:
            intero_default = ctx_time.get("intero", {}) or {}
            if "energy" in intero_default:
                bio_signals["energy"] = float(intero_default["energy"])
        bio_signals.setdefault("energy", 0.5)
        bio_signals.setdefault("uncertainty", float(ctx_time.get("uncertainty", 0.3)))
        bio_signals["metamemory_tot"] = float(metamemory_tot_flag)
        plan["bio_signals"] = dict(bio_signals)
        plan_controls, contributions = apply_coupling_with_limits(
            plan["controls"],
            bio_signals,
            self.coupling_matrix,
            limits=self.control_limits,
            d_tau=d_tau,
        )
        control_stages["biofield"] = dict(plan_controls)
        plan["controls"] = self._apply_jerk_limits(plan_controls)
        control_stages["jerk"] = dict(plan["controls"])
        control_deltas = diff_controls(control_stages)
        if not verify_contributions(control_stages, control_deltas):
            self.logger.warning("Control deltas failed verification")
        bio_snapshot = self.intero_bus.snapshot() if self.intero_bus is not None else {}
        jerk_rates: Dict[str, float] = {}
        if d_tau > 1e-6:
            for key in self.jerk_limits:
                if key not in plan["controls"]:
                    continue
                prev_val = self._previous_controls.get(key)
                if prev_val is None:
                    continue
                new_val = float(plan["controls"].get(key, 0.0))
                jerk_rates[key] = abs(new_val - prev_val) / d_tau
        plan["biofield"] = {
            "snapshot": bio_snapshot,
            "signals": {k: round(float(v), 4) for k, v in bio_signals.items()},
            "contributions": {
                k: round(float(v), 4) for k, v in contributions.items() if abs(v) > 1e-9
            },
            "controls_delta": control_deltas,
        }
        if jerk_rates:
            plan["biofield"]["jerk_rate"] = {
                k: round(v, 4) for k, v in jerk_rates.items()
            }
            max_jerk = max(jerk_rates.values())
            if max_jerk > self.jerk_rate_max:
                self.logger.warning(
                    "Jerk rate %.3f exceeds limit %.3f", max_jerk, self.jerk_rate_max
                )
        else:
            plan["biofield"]["jerk_rate"] = {}
        base_stage = control_stages.get("base", {})
        persona_stage = control_stages.get("persona", base_stage)
        norms_stage = control_stages.get("norms", persona_stage)
        green_stage = control_stages.get("green", norms_stage)
        assoc_value, assoc_diff = assoc_defect(
            base_stage, persona_stage, norms_stage, green_stage
        )
        if assoc_value > self.assoc_threshold:
            self.logger.warning(
                "Associativity defect %.3f exceeds threshold %.3f",
                assoc_value,
                self.assoc_threshold,
            )
        composition_info = {
            "assoc_defect": round(assoc_value, 4),
            "assoc_diff": {k: round(v, 4) for k, v in assoc_diff.items()},
            "threshold": self.assoc_threshold,
        }
        control_total = control_deltas.get("total", {})
        sensitivity_map = simple_sensitivity(control_total, self.sensitivity_eps)

        solver_residual = plan.get("bio_signals", {}).get("naturality_residual", 0.0)
        self.solver_manager.adjust(composition_info, {"residual": solver_residual})
        solver_snapshot = self.solver_manager.snapshot()

        if self.biofield_digest:
            plan["biofield"]["config_digest"] = self.biofield_digest

        tau_rate = float(time_info.get("tau_rate", 1.0))
        norms = plan.get("norms_cfg", self.cfg.get("norms", {}))
        safety = self.safety_orchestrator.evaluate(ctx_time, d_tau)
        base_read_only = bool(safety.get("domain_read_only")) or (
            safety.get("bayes_decision") in {"READ_ONLY", "BLOCK"}
        )
        safety["read_only"] = bool(safety.get("read_only", False) or base_read_only)
        meta_state = ctx_time.get("metamemory", {}) or {}
        if meta_state.get("tot_active") and meta_state.get("force_read_only", True):
            prev_state = self._hysteresis_mode
            if prev_state != "read_only":
                hysteresis_event = {
                    "from": prev_state,
                    "to": "read_only",
                    "reason": "metamemory_tot",
                    "enter": float(self.mode_hysteresis.enter),
                    "exit": float(self.mode_hysteresis.exit),
                }
            self._hysteresis_mode = "read_only"
            safety["read_only"] = True
            safety["metamemory_forced"] = True
        else:
            risk_upper = safety.get("risk_upper")
            risk_p = float(risk_upper) if risk_upper is not None else 0.0
            prev_state = self._hysteresis_mode
            decision = self.mode_hysteresis.decide(prev_state, risk_p)
            self._hysteresis_mode = decision.new_mode
            if decision.new_mode != prev_state:
                hysteresis_event = {
                    "from": prev_state,
                    "to": decision.new_mode,
                    "reason": decision.reason,
                    "enter": float(decision.thresholds.get("enter", 0.0)),
                    "exit": float(decision.thresholds.get("exit", 0.0)),
                }
            safety["read_only"] = bool(base_read_only)
            if decision.new_mode == "read_only":
                safety["read_only"] = True
        safety["mode_state"] = self._hysteresis_mode
        if hysteresis_event is not None:
            safety["hysteresis_event"] = hysteresis_event
        risk_upper_value = safety.get("risk_upper")
        risk_value = float(risk_upper_value) if risk_upper_value is not None else 0.0
        prev_coherence = self._last_coherence
        replay_result = self.replay_executor.run(
            ctx_time,
            plan,
            norms,
            safety,
            tau_rate,
        )
        replay_info = replay_result["info"]
        replay_details = replay_result["details"]
        best_choice = replay_result["best"]
        candidates_dump = replay_result["candidates"]
        d_tau_step_used = replay_result["d_tau_step"]

        if replay_details:
            best_summary = best_choice.get("summary") if best_choice else None
            if isinstance(best_summary, dict):
                ctx_time["value_summary"] = dict(best_summary)

        new_coherence = self.self_model.coherence()
        self._last_coherence = new_coherence

        result = self.engine.generate(plan["prompt"], plan["controls"], ctx_time)
        original_text = result.get("text", "") or ""
        protected_spans, protected_stats = scan_protected_regions(original_text)
        filler_summary = {
            "enabled": self.filler_enabled,
            "injected": 0,
            "items": [],
            "bank_version": self.filler_bank.get("version"),
            "rate_per_100chars": 0.0,
            "cooldown_ok": True,
            "damped_by_residual": 1.0,
            "probability": 0.0,
            "protected_spans": protected_stats,
            "added_chars": 0,
            "session_id": session_id,
            "max_added_chars": (
                self.filler_policy.base_max_added if self.filler_policy else 0
            ),
            "breaks_ms": [],
        }
        language_loss_info: Dict[str, Any] = {
            "loss": 0.0,
            "reconstructed": {},
            "covered": [],
            "missing": [],
        }
        sense_residual: Dict[str, Any] = {"delta": 0.0, "top": []}
        disclosure_decision: Dict[str, Any] = {"level": "none", "targets": []}
        disclosure_payload: Dict[str, Any] = {
            "disclosure": "",
            "bridges": [],
            "asks": [],
        }
        if (
            self.sense_enabled
            and sense_envelope is not None
            and sense_envelope.features
            and original_text
        ):
            language_loss_info = measure_language_loss(
                sense_envelope,
                original_text,
                self._metaphor_templates,
            )
            shareability = self._sense_share_cfg.get("shareability", {})
            weights = self._sense_share_cfg.get("weights", {})
            sense_residual = compute_residual(
                sense_envelope.features,
                language_loss_info.get("reconstructed", {}),
                shareability,
                weights,
            )
            disclosure_decision = decide_disclosure(
                float(sense_residual.get("delta", 0.0)),
                sense_residual.get("top", []),
                self._sense_share_cfg.get("thresholds", {}),
            )
            forced_level = ctx_time.get("candor_force_level")
            if isinstance(forced_level, str) and forced_level in {
                "warn",
                "must",
                "ask",
            }:
                priority = {"none": 0, "warn": 1, "ask": 2, "must": 3}
                current = disclosure_decision.get("level", "none")
                if priority.get(forced_level, 0) > priority.get(current, 0):
                    disclosure_decision["level"] = forced_level
            disclosure_payload = craft_payload(
                disclosure_decision["level"],
                disclosure_decision.get("targets", []),
                locale=str(ctx.get("lang") or ctx_time.get("lang") or "ja"),
                persona=str(
                    ctx.get("candor_persona")
                    or plan.get("chosen_mode")
                    or ctx.get("persona")
                    or "default"
                ),
                templates=self._disclosure_templates,
                metaphors=self._metaphor_templates,
            )
            if disclosure_payload.get("asks"):
                disclosure_payload["asks"] = disclosure_payload["asks"][:1]
            injected_text = self._inject_disclosure(original_text, disclosure_payload)
            if injected_text != original_text:
                original_text = injected_text
                result["text"] = injected_text
        peers = ctx_time.get("peers", [])
        community_metrics = resonance_metrics(peers)
        raw_value = (
            ctx_time.get("value_summary") or ctx_time.get("value") or {"total": 0.0}
        )
        if (
            isinstance(raw_value, dict)
            and "total" not in raw_value
            and "score" in raw_value
        ):
            raw_value = {"total": float(raw_value.get("score", 0.0)), **raw_value}
        value_state = (
            dict(raw_value)
            if isinstance(raw_value, dict)
            else {"total": float(raw_value)}
        )
        if best_choice and isinstance(value_state, dict):
            value_state.setdefault("meta", {})
            value_state["meta"]["coherence"] = float(
                best_choice.get("coherence", self.self_model.coherence())
            )
        value_committee = ctx_time.get("value_committee")
        if value_state:
            plan.setdefault("value_summary", value_state)
        if value_committee:
            plan.setdefault("value_committee", value_committee)

        u_top = max(
            (float(item.get("U", 0.0)) for item in candidates_dump), default=0.0
        )
        misfire_flag = bool(ctx_time.get("misfire", False))
        delta_coh_positive = new_coherence >= (prev_coherence - 1e-6)
        persona_success = (not misfire_flag) and delta_coh_positive and (u_top >= 0.0)
        self.persona_manager.update(
            plan.get("chosen_mode"), success=persona_success, d_tau=d_tau
        )
        self.safety_orchestrator.update(safety, misfire=misfire_flag, d_tau=d_tau)

        feedback_events = (
            ctx_time.get("feedback") or ctx_time.get("feedback_events") or []
        )
        feedback_results: List[Dict[str, Any]] = []
        if isinstance(feedback_events, (list, tuple)):
            for event in feedback_events:
                if not isinstance(event, dict):
                    continue
                event_payload = dict(event)
                if "mode" not in event_payload and plan.get("chosen_mode"):
                    event_payload["mode"] = plan.get("chosen_mode")
                try:
                    result = apply_feedback_event(
                        event_payload,
                        persona_manager=self.persona_manager,
                        safety_orchestrator=self.safety_orchestrator,
                        safety_ctx=safety,
                        default_mode=plan.get("chosen_mode"),
                    )
                    feedback_results.append(result)
                    log_payload = dict(event_payload)
                    log_payload["result"] = result
                    record_feedback_event(log_payload)
                except Exception:
                    continue

        thought_bus_report: Optional[Dict[str, Any]] = None
        thought_packets_payload: List[Dict[str, Any]] = []
        thought_bus_tx_logs: List[Dict[str, Any]] = []

        advice = self.forgetting.advise(
            tau_rate=float(time_info.get("tau_rate", 1.0)),
            inflammation=float(
                plan.get("bio_signals", {}).get("inflammation_global", 0.0)
            ),
            uncertainty=float(ctx_time.get("uncertainty", 0.0)),
            novelty=float(ctx_time.get("novelty", 0.0)),
            cfl=float(self.solver_manager.modules.get("biofield", {}).get("cfl", 0.0)),
        )
        self._apply_forgetting_advice(advice)
        forgetting_snapshot = {
            "lstm": advice.lstm,
            "ssm": advice.ssm,
            "intero": advice.intero,
            "replay": advice.replay,
            "persona": advice.persona,
        }

        pred_metrics = {
            "U_top": float(best_choice.get("U", 0.0)) if best_choice else 0.0,
            "coh_pred": (
                float(best_choice.get("coherence", 0.0)) if best_choice else 0.0
            ),
        }
        obs_metrics = {
            "U_real": float(value_state.get("total", 0.0)),
            "coh_delta": float(new_coherence - prev_coherence),
            "misfire": float(misfire_flag),
        }
        naturality_value = naturality_residual(
            pred_metrics, obs_metrics, self.naturality_weights
        )
        naturality_components = residual_components(
            pred_metrics, obs_metrics, self.naturality_weights
        )
        if self.naturality_warn_threshold is not None and naturality_value > float(
            self.naturality_warn_threshold
        ):
            self.logger.warning(
                "Naturality residual %.3f exceeds threshold %.3f",
                naturality_value,
                self.naturality_warn_threshold,
            )
        plan.setdefault("bio_signals", {})["naturality_residual"] = naturality_value

        autop_mid = {"heartiness": self.heartiness}
        autop_post = {"tb_gain_cap": autop_pre.get("tb_gain_cap")}
        jerk_dict = (
            plan.get("biofield", {}).get("jerk_rate", {})
            if isinstance(plan.get("biofield"), dict)
            else {}
        )
        jerk_rate = 0.0
        if isinstance(jerk_dict, dict) and jerk_dict:
            try:
                jerk_rate = max(float(v) for v in jerk_dict.values())
            except Exception:
                jerk_rate = 0.0
        latency_ratio = float(
            ctx_time.get("performance", {}).get(
                "latency_p95_ratio", ctx.get("latency_p95_ratio", 1.0)
            )
        )
        autop_metrics_current = {
            "assoc_defect": composition_info.get("assoc_defect", 0.0),
            "naturality_residual": naturality_value,
            "r": float(community_metrics.get("r") or 0.0),
            "jerk_rate": jerk_rate,
            "latency_p95_ratio": latency_ratio,
        }
        if hasattr(self, "autopilot") and self.autopilot is not None:
            try:
                autop_mid = self.autopilot.mid_adjust(autop_metrics_current)
                self.heartiness = float(autop_mid.get("heartiness", self.heartiness))
            except Exception:
                autop_mid = {"heartiness": self.heartiness}
            try:
                autop_post = self.autopilot.post_turn(autop_metrics_current)
            except Exception:
                autop_post = {"tb_gain_cap": autop_pre.get("tb_gain_cap")}
        self._last_autop_metrics = autop_metrics_current
        roles_post, narrative_coherence = self._update_self_posteriors(
            plan.get("chosen_mode"),
            d_tau,
            ctx_time,
            gaze_context,
        )
        sunyata_metrics = self._update_sunyata_metrics(
            d_tau,
            ctx_time,
            plan,
            autop_metrics_current,
            sense_residual,
            roles_post,
            narrative_coherence,
            gaze_context,
        )

        if self.thought_bus.enabled():
            tau_now_current = float(time_info.get("tau", self.timekeeper.tau_now()))
            agent_id = str(
                ctx_time.get("agent_id")
                or ctx_time.get("session_id")
                or ctx_time.get("conversation_id")
                or ctx_time.get("user_id")
                or "self"
            )
            policy_cfg = {
                "ttl_tau": float(self.thought_cfg.get("ttl_tau_default", 2.0)),
                "ttl_tau_uncertainty": float(
                    self.thought_cfg.get(
                        "ttl_tau_uncertainty",
                        self.thought_cfg.get("ttl_tau_default", 2.0),
                    )
                ),
                "tags": list(self.thought_cfg.get("tags_default", [])),
            }
            urk_trace_payload = self._prepare_thought_trace(
                candidates_dump, replay_details
            )
            se_stats_payload = (
                result.get("trace", {}) if isinstance(result, dict) else {}
            )
            packets_objs = self.thought_extractor.extract(
                agent_id=agent_id,
                tau_now=tau_now_current,
                urk_trace=urk_trace_payload,
                se_stats=se_stats_payload if isinstance(se_stats_payload, dict) else {},
                policy=policy_cfg,
            )
            thought_packets_payload = [packet.to_dict() for packet in packets_objs]
            packets_by_agent: Dict[str, List[Dict[str, Any]]] = {
                agent_id: list(thought_packets_payload)
            }
            peer_ids: List[str] = []
            peers_current = ctx_time.get("peers", []) or []

            for idx, peer in enumerate(peers_current):
                peer_id = str(peer.get("id") or f"peer_{idx}")
                peer_ids.append(peer_id)
                peer_packets_raw = peer.get("thoughts")
                if isinstance(peer_packets_raw, list):
                    sanitized_peer_packets = self._sanitize_peer_packets(
                        peer_packets_raw, default_origin=peer_id
                    )
                    if sanitized_peer_packets:
                        packets_by_agent[peer_id] = sanitized_peer_packets

            packets_by_agent, share_graph = self.thought_assigner.assign(
                packets_by_agent
            )
            local_packets_annotated = packets_by_agent.get(
                agent_id, thought_packets_payload
            )
            thought_packets_payload = local_packets_annotated
            entropy_vals = [
                float(packet.get("entropy", 0.0)) for packet in thought_packets_payload
            ]
            avg_entropy = sum(entropy_vals) / len(entropy_vals) if entropy_vals else 0.0
            junk_prob_context = float(
                ctx_time.get(
                    "junk_prob",
                    plan.get("bio_signals", {}).get(
                        "junk_prob", ctx_time.get("hygiene", {}).get("junk_prob", 0.0)
                    ),
                )
            )
            ctx_time["junk_prob_context"] = junk_prob_context
            gate = self.thought_bus.policy_gate(
                mode=self.mode,
                risk_p=risk_value,
                read_only=bool(safety.get("read_only", False)),
                tau_rate=float(time_info.get("tau_rate", 1.0)),
                inflammation=float(
                    plan.get("bio_signals", {}).get("inflammation_global", 0.0)
                ),
                synchrony=community_metrics.get("r"),
                assoc_defect=composition_info.get("assoc_defect", 0.0),
                naturality_residual=naturality_value,
                avg_entropy=avg_entropy,
                junk_prob=junk_prob_context,
                tx_count_last=self.thought_bus.last_tx_count(),
            )
            tx_logs_objs = self.thought_bus.deliver(
                me=agent_id,
                peers=peer_ids,
                packets=local_packets_annotated,
                gate=gate,
                tau_now=tau_now_current,
                cooldown_tau=float(self.thought_cfg.get("cooldown_tau", 0.8)),
                acl=self.peer_acl,
            )
            thought_bus_tx_logs = [log.to_dict() for log in tx_logs_objs]
            tb_rejects = getattr(self.thought_bus, "last_rejects", lambda: [])()
            ctx_time["thought_packets"] = thought_packets_payload
            plan["thought_bus"] = {
                "packets": [
                    {k: v for k, v in packet.items() if k != "vec"}
                    for packet in thought_packets_payload
                ],
                "reason": gate.get("reason"),
            }
            redacted_packets = [
                {k: v for k, v in packet.items() if k != "vec"}
                for packet in thought_packets_payload
            ]
            thought_bus_report = {
                "allow": bool(gate.get("allow")),
                "gain": float(gate.get("gain", 0.0)),
                "reason": gate.get("reason"),
                "mode": gate.get("mode", self.mode),
                "packets": redacted_packets,
                "avg_entropy": round(float(avg_entropy), 4),
                "junk_prob": round(float(junk_prob_context), 4),
                "tx": thought_bus_tx_logs,
                "rejects": tb_rejects,
                "share_graph": {
                    "digest": share_graph.digest,
                    "edges": [
                        [a, b, round(val, 3)]
                        for (a, b), val in sorted(share_graph.adjacency.items())
                    ],
                },
            }

        last_tau = self._filler_last_tau.get(session_id)
        if self.filler_enabled and self.filler_policy is not None and original_text:
            signals_for_filler = dict(plan.get("bio_signals", {}))
            signals_for_filler["naturality_residual"] = naturality_value
            mode_name = (
                plan.get("chosen_mode") or ctx_time.get("mode") or self.mode.lower()
            )
            decision = self.filler_policy.decide(
                mode=str(mode_name),
                mood=plan.get("mood", {}),
                signals=signals_for_filler,
                tau_info=time_info,
                norms=plan.get("norms_cfg", {}),
                domain=str(ctx_time.get("domain", "dialogue")),
                heartiness=self.heartiness,
                last_tau=last_tau,
            )
            self._filler_last_tau[session_id] = decision.new_last_tau
            filler_summary["cooldown_ok"] = decision.cooldown_ok
            filler_summary["damped_by_residual"] = decision.damp_factor
            filler_summary["probability"] = decision.probability
            filler_summary["max_added_chars"] = decision.max_added_chars
            if decision.enabled and decision.entries:
                filler_entries = [
                    (entry["kind"], entry["position"], entry["phrase"])
                    for entry in decision.entries
                    if entry.get("phrase")
                ]
                if filler_entries:
                    breaks_ms = [
                        int(entry.get("ssml_break", 0)) for entry in decision.entries
                    ]
                    new_text = insert_fillers(original_text, filler_entries)
                    added_chars = max(len(new_text) - len(original_text), 0)
                    if added_chars <= decision.max_added_chars:
                        result["text"] = new_text
                        filler_summary["injected"] = len(filler_entries)
                        filler_summary["items"] = [
                            {
                                "kind": entry["kind"],
                                "position": entry["position"],
                                "phrase": entry["phrase"],
                                "ssml_break": entry.get("ssml_break"),
                            }
                            for entry in decision.entries
                        ]
                        filler_summary["added_chars"] = added_chars
                        filler_summary["breaks_ms"] = breaks_ms
                    else:
                        self.logger.debug(
                            "Filler skipped due to char limit (added=%s, limit=%s)",
                            added_chars,
                            decision.max_added_chars,
                        )
                tts_cfg = self.cfg.get("tts", {})
        if ctx.get("tts", False):
            text_for_tts = result.get("text", "")
            breaks_ms = filler_summary.get("breaks_ms") or []
            if breaks_ms:
                text_for_tts = to_placeholder(text_for_tts, breaks_ms)
            rendered = render_for_tts(
                text_for_tts,
                backend=tts_cfg.get("backend", "stylebert"),
                unit_ms=int(tts_cfg.get("unit_ms", 90)),
                max_commas=int(tts_cfg.get("max_commas", 4)),
                wrap_ssml=bool(tts_cfg.get("wrap_ssml", True)),
            )
            result["tts_text"] = rendered
            filler_summary.setdefault("tts", {})
            filler_summary["tts"].update(
                {
                    "backend": tts_cfg.get("backend", "stylebert"),
                    "unit_ms": int(tts_cfg.get("unit_ms", 90)),
                    "max_commas": int(tts_cfg.get("max_commas", 4)),
                    "placeholders": len(breaks_ms),
                }
            )

        char_count_final = max(len(result.get("text", "")), 1)
        filler_summary["rate_per_100chars"] = round(
            filler_summary["injected"] * 100.0 / char_count_final, 3
        )
        plan["filler"] = filler_summary

        if replay_details and best_choice:
            eta = float(self.replay_cfg.get("eligibility_eta", 0.8))
            lr = float(self.replay_cfg.get("eligibility_lr", 0.05))
            rollout = [
                {
                    "predU": float(
                        best_choice.get("summary", {}).get(
                            "total", best_choice.get("U", 0.0)
                        )
                    ),
                    "targetU": float(
                        value_state.get("total", best_choice.get("U", 0.0))
                    ),
                }
            ]
            d_tau_for_credit = float(
                d_tau_step_used
                if d_tau_step_used is not None
                else self.replay_cfg.get("d_tau_step_defaults", {}).get(
                    self.replay_cfg.get("adapter", "dialogue"), 0.7
                )
            )
            eligibility = apply_credit_updates(
                rollout,
                read_only_policy=best_choice.get("read_only", False),
                d_tau_step=d_tau_for_credit,
                eta=eta,
                lr=lr,
            )
            replay_details["eligibility"] = eligibility

        organism_state = dict(ctx_time.get("organism", {"drives": {}}))

        receipt = build_receipt(
            mode=self.mode,
            style=plan["style"],
            heartiness=self.heartiness,
            controls_before={"temp": 1.0, "top_p": 1.0, "pause": 0, "direct": 0.0},
            controls_after=plan["controls"],
            mood=plan["mood"],
            replay=replay_info,
            green={
                "weight": (self.heartiness**2) * 0.4,
                "qualia": ctx_time.get("qualia"),
                "response": plan.get("green", {}),
            },
            norms=plan.get("norms", {}),
            community=community_metrics,
            value=value_state,
            organism=organism_state,
            engine_trace=result.get("trace", {}),
        )
        receipt["time"] = time_info
        if "persona" in plan:
            receipt["persona"] = plan["persona"]
        interference_metrics = {
            "masked": 0,
            "skipped": 0,
            "sum_similarity": 0.0,
            "mask_windows": [],
        }
        if replay_details:
            replay_unified = dict(replay_details)
            receipt["replay_unified"] = replay_unified
            reverse_ratio = replay_details.get("reverse_ratio")
            replay_unified["reverse_ratio"] = reverse_ratio
            summary_block = replay_unified.setdefault("summary", {})
            summary_block["direction"] = self._resolve_replay_direction(reverse_ratio)
            try:
                trace_meta = {
                    "domain": ctx_time.get("domain", ""),
                    "norm_risk": ctx_time.get("norm_risk", 0.0),
                    "novelty": ctx_time.get("novelty", 0.0),
                    "policy_update": replay_details.get("policy_update", True),
                    "reverse_ratio": replay_details.get("reverse_ratio"),
                    "junk_prob": float(
                        ctx_time.get("junk_prob")
                        or plan.get("bio_signals", {}).get("junk_prob", 0.0)
                    ),
                    "safety_block": bool(
                        safety.get("read_only")
                        or safety.get("bayes_decision") in {"READ_ONLY", "BLOCK"}
                    ),
                }
                qualia_block = receipt.setdefault("qualia", {})
                qualia_gate = receipt.setdefault("qualia_gate", {})
                def _opc_vec(src):
                    return [float(src.get(k) or 0.0) for k in ("uncertainty", "pressure", "novelty", "norm_risk", "dU_est")]
                u_t = float(qualia_block.get("u_t") or ctx_time.get("u_t") or ctx_time.get("uncertainty") or 0.0)
                cur_vec = _opc_vec(ctx_time)
                prev_vec = getattr(self, "_opc_prev_vec", None) or cur_vec
                self._opc_prev_vec = cur_vec
                def _opc_cosine_dist(a, b):
                    da = math.sqrt(sum(x*x for x in a)) + 1e-12
                    db = math.sqrt(sum(x*x for x in b)) + 1e-12
                    dot = sum(x*y for x, y in zip(a, b))
                    return float(1.0 - dot / (da * db))
                m_t = _opc_cosine_dist(prev_vec, cur_vec)
                load_t = float(max(0.0, float(ctx_time.get("pressure") or 0.0)))
                gate_enabled = bool(ctx_time.get("qualia_gate_enabled", False))
                k_u = 2.0
                k_l = 2.0
                u0 = 0.5
                l0 = 0.5
                theta = float((self.cfg.get("qualia_gate") or {}).get("theta", -1.45))
                setattr(self, "_opc_theta", theta)
                logit = k_u * (u0 - u_t) + k_l * (l0 - load_t) - theta
                p_gate = 1.0 / (1.0 + math.exp(-logit))
                if not gate_enabled:
                    allow = True
                    suppress = False
                else:
                    allow = bool(p_gate >= 0.5)
                    suppress = not allow
                qualia_block["u_t"] = u_t
                qualia_block["m_t"] = float(m_t)
                qualia_block["load"] = load_t
                qualia_block["a_t"] = int(1 if allow else 0)
                qualia_block["unconscious_success"] = int(1 if suppress else 0)
                qualia_gate["logit"] = float(logit)
                qualia_gate["p_t"] = float(p_gate)
                qualia_gate["theta"] = float(theta)
                qualia_gate["k_u"] = float(k_u)
                qualia_gate["k_l"] = float(k_l)
                qualia_gate["u0"] = float(u0)
                qualia_gate["l0"] = float(l0)
                qualia_gate["suppress_narrative"] = bool(suppress)
                if receipt:
                    trace_meta["receipt"] = receipt
                trace = ReplayTrace(
                    trace_id=str(uuid.uuid4()),
                    episode_id=str(ctx_time.get("episode_id", "")),
                    timestamp=time.time(),
                    source="internal",
                    horizon=int(replay_details.get("steps", 0)),
                    uncertainty=float(ctx_time.get("uncertainty", 0.0)),
                    mood=plan["mood"],
                    value=value_state,
                    controls=plan["controls"],
                    imagined={
                        "best_action": replay_details.get("best_action"),
                        "utility": replay_details.get("utility"),
                        "candidates": replay_details.get("candidates", []),
                    },
                    meta=trace_meta,
                )
                gate_result = (
                    self.interference_gate.evaluate(trace)
                    if self.interference_gate
                    else {"action": "pass"}
                )
                action = gate_result.get("action", "pass")
                if action == "skip":
                    interference_metrics["skipped"] += 1
                else:
                    if action == "mask":
                        info = {
                            "action": "mask",
                            "similarity": float(gate_result.get("similarity", 0.0)),
                            "mask_until_tau": gate_result.get("mask_until_tau"),
                            "ttl_override_tau": gate_result.get("ttl_override_tau"),
                            "tau": float(
                                getattr(self.timekeeper, "tau_now", lambda: 0.0)()
                            ),
                        }
                        trace.meta["interference"] = info
                        interference_metrics["masked"] += 1
                        interference_metrics["sum_similarity"] += float(
                            gate_result.get("similarity", 0.0)
                        )
                        mask_until = gate_result.get("mask_until_tau")
                        if mask_until is not None:
                            current_tau = float(
                                getattr(self.timekeeper, "tau_now", lambda: 0.0)()
                            )
                            interference_metrics["mask_windows"].append(
                                float(mask_until) - current_tau
                            )
                        trace.meta["ttl_override_tau"] = gate_result.get(
                            "ttl_override_tau"
                        )
                        trace.meta["mask_until_tau"] = gate_result.get("mask_until_tau")
                    self._update_go_sc_annotations(trace, receipt)
                    self._attach_fastpath_receipt(trace, receipt)
                    self.replay_memory.store(trace)
            except Exception:
                pass
        if value_committee:
            receipt["value_committee"] = value_committee
        persona_report = {
            "chosen": plan.get("chosen_mode"),
            "success": persona_success,
            "metrics": self.persona_manager.metrics(),
        }
        safety_report = {
            "category": safety["act_category"],
            "decision": safety["bayes_decision"]
            or ("READ_ONLY" if safety["read_only"] else "ALLOW"),
            "read_only": safety["read_only"],
            "posteriors": self.safety_orchestrator.metrics(),
        }
        receipt = augment_receipt(
            receipt,
            coherence=new_coherence,
            persona=persona_report,
            safety=safety_report,
            value_weights=self.value_weights,
            biofield=plan.get("biofield"),
            control_deltas=plan.get("biofield", {}).get("controls_delta"),
        )
        receipt["autopilot"] = {
            "pre": {
                "mode": autop_pre.get("mode", self.mode),
                "msv_k_min": autop_pre.get("msv_k_min"),
                "tb_gain_cap": autop_pre.get("tb_gain_cap"),
                "reason": autop_pre.get("reason"),
            },
            "mid": {"heartiness": autop_mid.get("heartiness", self.heartiness)},
            "post": {"tb_gain_cap": autop_post.get("tb_gain_cap")},
        }
        if feedback_results:
            receipt["feedback"] = feedback_results
        receipt["solver"] = solver_snapshot
        receipt["forgetting"] = forgetting_snapshot
        receipt["filler"] = plan.get("filler", filler_summary)
        receipt["composition"] = composition_info
        receipt["naturality"] = {
            "residual": round(naturality_value, 4),
            "components": {k: round(v, 4) for k, v in naturality_components.items()},
            "pred": {k: round(float(v), 4) for k, v in pred_metrics.items()},
            "obs": {k: round(float(v), 4) for k, v in obs_metrics.items()},
        }
        if gaze_context:
            receipt["gaze"] = gaze_context
        if self.sense_enabled and sense_envelope is not None:
            receipt["sense"] = {
                "envelope": sense_envelope.to_dict(),
                "language_loss": {
                    "loss": round(float(language_loss_info.get("loss", 0.0)), 4),
                    "covered": list(language_loss_info.get("covered", [])),
                    "missing": list(language_loss_info.get("missing", [])),
                },
                "residual": {
                    "delta": round(float(sense_residual.get("delta", 0.0)), 4),
                    "top": sense_residual.get("top", []),
                },
            }
            receipt["disclosure"] = {
                "level": disclosure_decision.get("level", "none"),
                "targets": disclosure_decision.get("targets", []),
                "bridges": disclosure_payload.get("bridges", []),
                "ask": disclosure_payload.get("asks", []),
            }
        if roles_post:
            top_roles = sorted(roles_post.items(), key=lambda item: -item[1])[:3]
            receipt["self"] = {
                "roles": [(role, round(prob, 4)) for role, prob in top_roles],
                "coherence": round(narrative_coherence, 4),
            }
        if sunyata_metrics:
            merged = dict(receipt.get("sunyata", {}))
            merged.update(
                {
                    "do_topk": sunyata_metrics.get(
                        "do_topk", merged.get("do_topk", [])
                    ),
                    "rigidity_kld": round(
                        float(sunyata_metrics.get("rigidity_kld", 0.0)), 4
                    ),
                    "isolation_index": round(
                        float(sunyata_metrics.get("isolation_index", 0.0)), 4
                    ),
                    "clinging_triggered": bool(
                        sunyata_metrics.get("clinging_triggered", False)
                    ),
                }
            )
            receipt["sunyata"] = merged
        if "s_index" in sunyata_metrics:
            receipt["s_index"] = dict(sunyata_metrics["s_index"])
        receipt["intuition_map"] = {
            "sensitivity": {k: round(v, 4) for k, v in sensitivity_map.items()},
        }
        if interference_metrics["masked"] or interference_metrics["skipped"]:
            gate_info = {
                "masked": interference_metrics["masked"],
                "skipped": interference_metrics["skipped"],
            }
            if interference_metrics["masked"]:
                gate_info["avg_similarity"] = round(
                    interference_metrics["sum_similarity"]
                    / max(1, interference_metrics["masked"]),
                    4,
                )
            if interference_metrics["mask_windows"]:
                gate_info["avg_cooldown_tau"] = round(
                    sum(interference_metrics["mask_windows"])
                    / len(interference_metrics["mask_windows"]),
                    4,
                )
            gate_info["masked_count"] = gate_info["masked"]
            gate_info["skipped_count"] = gate_info["skipped"]
            if "avg_similarity" in gate_info:
                gate_info["avg_sim"] = gate_info["avg_similarity"]
            if "avg_cooldown_tau" in gate_info:
                gate_info["window_tau"] = gate_info["avg_cooldown_tau"]
            receipt["interference_gate"] = gate_info

        if thought_bus_report is not None:
            receipt["thought_bus"] = thought_bus_report
        if metamemory_snapshot is not None:
            receipt["metamemory"] = metamemory_snapshot
        if memory_gc_stats is not None:
            receipt["memory_gc"] = memory_gc_stats
        if hysteresis_event is not None:
            receipt["mode_transition"] = hysteresis_event
        action = choose_action(
            float(receipt.get("value", {}).get("total", 0.0)),
            receipt.get("organism", {}).get("drives", {}),
        )
        gate = preflight_guard(action, ctx_time)
        if not gate.get("ok", True):
            action = "ask_consent"
        receipt["action"] = action
        receipt["safety"] = gate
        log_receipt(receipt)
        if self.intero_bus is not None and self.biofield_state_path is not None:
            try:
                state = self.intero_bus.snapshot()
                state["ts_wall"] = time.time()
                state["config_digest"] = self.biofield_digest
                self.intero_bus.save(self.biofield_state_path, state)
            except Exception:
                self.logger.exception(
                    "Failed to persist biofield state to %s", self.biofield_state_path
                )
        if isinstance(plan.get("controls"), Mapping):
            self._previous_controls = {
                key: float(value)
                for key, value in plan["controls"].items()
                if isinstance(value, (int, float))
            }
        response = {
            "text": result.get("text", ""),
            "tts_text": result.get("tts_text"),
            "receipt": receipt,
        }
        if thought_packets_payload:
            response["thought_packets"] = thought_packets_payload
        if thought_bus_tx_logs:
            response["thought_bus_tx"] = thought_bus_tx_logs
        return response

    def _load_yaml_config(self, path: Path) -> Dict[str, Any]:
        try:
            if not path.exists():
                return {}
            return yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        except Exception:
            self.logger.debug("Failed to load %s", path, exc_info=True)
            return {}

    def _attach_fastpath_receipt(
        self, trace: ReplayTrace, receipt: Dict[str, Any]
    ) -> None:
        if not self._fastpath_enabled or not self._fastpath_profiles:
            return
        projector_map = self._projector_map_for_trace(trace)
        time_index = self._extract_time_index(trace)
        if not projector_map or not time_index:
            return
        fast_section: Dict[str, Dict[str, Any]] = {}
        for profile_name in self._fastpath_profiles:
            profile = TASK_PROFILES.get(profile_name)
            if profile is None:
                continue
            try:
                summary = summarize_task_fastpath(
                    profile, time_index, projector_map, include_full_labels=False
                )
            except Exception:
                continue
            fast_entry = {
                "final_ok": bool(summary.get("final_ok")),
                "fast": {
                    k: self._sanitize_fastpath_value(v)
                    for k, v in (summary.get("fast") or {}).items()
                },
                "needs_full": list(summary.get("needs_full") or []),
                "predicates": dict(summary.get("predicates") or {}),
            }
            fast_section[profile_name] = fast_entry
            if not fast_entry["predicates"].get("fast_rescue"):
                continue
            receipt.setdefault("go_sc", {}).setdefault("resolution_reason", "pending")
            hint_reason = "fast_rescue_candidate"
            if self._fastpath_mode == "soft_hint":
                decisions = receipt.setdefault("decisions", [])
                decisions.append({"reason": "fast_rescue_hint", "profile": profile_name})
            elif self._fastpath_mode == "ab_test" and self._fastpath_ab_fraction > 0.0:
                if self._fastpath_rng.random() < self._fastpath_ab_fraction:
                    hint_reason = "fast_rescue_ab_test"
                    go_sc_info = receipt.setdefault("go_sc", {})
                    prev_scale = float(go_sc_info.get("ttl_scale", trace.meta.get("ttl_scale", 1.0)))
                    new_scale = max(prev_scale, self._fastpath_ab_scale)
                    go_sc_info["ttl_scale"] = new_scale
                    trace.meta["ttl_scale"] = new_scale
                    fast_entry["ab_test_applied"] = True
            if receipt["go_sc"].get("resolution_reason") == "pending":
                receipt["go_sc"]["resolution_reason"] = hint_reason
            if hint_reason != "fast_rescue_candidate":
                trace.meta["resolution_reason"] = hint_reason
        if not fast_section:
            return
        receipt["fastpath"] = fast_section
        trace.meta.setdefault("receipt", {})
        trace.meta["receipt"]["fastpath"] = fast_section
        if "go_sc" in receipt:
            trace.meta["receipt"]["go_sc"] = dict(receipt["go_sc"])
        if time_index and not trace.meta.get("time_index"):
            trace.meta["time_index"] = list(time_index)

    def _sanitize_fastpath_value(self, value: Any) -> Any:
        if isinstance(value, set):
            return sorted(value)
        if isinstance(value, list):
            return [self._sanitize_fastpath_value(item) for item in value]
        if isinstance(value, tuple):
            return [self._sanitize_fastpath_value(item) for item in value]
        if isinstance(value, dict):
            return {
                key: self._sanitize_fastpath_value(val) for key, val in value.items()
            }
        return value

    def _extract_time_index(self, trace: ReplayTrace) -> List[float]:
        times: List[float] = []
        raw_index = trace.meta.get("time_index")
        if isinstance(raw_index, Sequence):
            for entry in raw_index:
                try:
                    times.append(float(entry))
                except (TypeError, ValueError):
                    continue
        streams = trace.meta.get("fastpath_streams")
        if isinstance(streams, Mapping):
            for payload in streams.values():
                if isinstance(payload, Mapping):
                    for key in payload.keys():
                        try:
                            times.append(float(key))
                        except (TypeError, ValueError):
                            continue
        return sorted({t for t in times if isinstance(t, (int, float))})

    def _projector_map_for_trace(
        self, trace: ReplayTrace
    ) -> Dict[str, Callable[[float], Any]]:
        streams = trace.meta.get("fastpath_streams")
        projectors: Dict[str, Callable[[float], Any]] = {}
        if isinstance(streams, Mapping):
            for name, payload in streams.items():
                projector = self._build_projector_from_payload(payload)
                if projector is not None:
                    projectors[str(name)] = projector
        return projectors

    def _build_projector_from_payload(
        self, payload: Any
    ) -> Optional[Callable[[float], Any]]:
        if callable(payload):
            return payload
        data: Dict[float, Any] = {}
        if isinstance(payload, Mapping):
            for key, value in payload.items():
                try:
                    data[float(key)] = value
                except (TypeError, ValueError):
                    continue
        elif isinstance(payload, (list, tuple)):
            for entry in payload:
                if isinstance(entry, (list, tuple)) and len(entry) == 2:
                    t, val = entry
                    try:
                        data[float(t)] = val
                    except (TypeError, ValueError):
                        continue
        if not data:
            return None
        return lambda t, table=data: table.get(t)

    def _resolve_replay_direction(self, reverse_ratio: Optional[float]) -> str:
        if reverse_ratio is None:
            return "mixed"
        reverse_threshold = self._replay_direction_thresholds.get("reverse", 0.6)
        forward_threshold = self._replay_direction_thresholds.get("forward", 0.4)
        if reverse_ratio >= reverse_threshold:
            return "reverse"
        if reverse_ratio <= forward_threshold:
            return "forward"
        return "mixed"

    def _update_go_sc_annotations(
        self, trace: ReplayTrace, receipt: Dict[str, Any]
    ) -> None:
        if not self._go_sc_weights or not self._go_sc_percentiler:
            return
        event_view = {
            "uncertainty": trace.uncertainty,
            "value": trace.value,
            "meta": trace.meta,
        }
        try:
            score, percentile = compute_go_sc(
                self._go_sc_weights, event_view, self._go_sc_percentiler
            )
        except Exception:
            return
        trace.meta["go_score"] = score
        trace.meta["go_percentile"] = percentile
        trace.meta.setdefault("ttl_scale", 1.0)
        trace.meta.setdefault("resolution_reason", "pending")
        receipt.setdefault("go_sc", {}).update(
            {
                "score": score,
                "percentile": percentile,
                "ttl_scale": trace.meta.get("ttl_scale", 1.0),
                "resolution_reason": trace.meta.get("resolution_reason", "pending"),
            }
        )

    def _apply_do_profile(self, domain: str) -> None:
        profiles = getattr(self, "_do_profiles", None)
        if not profiles:
            return
        normalized = (domain or "").lower()
        target = None
        if normalized and normalized in profiles:
            target = normalized
        elif "default" in profiles:
            target = "default"
        if target == self._do_profile_active:
            return
        weights = None
        if target:
            weights = profiles.get(target)
        if not weights:
            weights = (self.sunyata_cfg.get("do_graph") or {}).get("weights")
        if isinstance(weights, dict):
            self.do_graph.load_weights(weights)
            self._do_profile_active = target

    def _build_sense_envelope(
        self,
        ctx: Mapping[str, Any],
    ) -> Tuple[Optional[SenseEnvelope], Optional[GazeSummary]]:
        payload = ctx.get("sense_envelope")
        if isinstance(payload, Mapping):
            features = clamp_features(payload.get("features", {}) or {})
            if not features:
                return None, None
            envelope = SenseEnvelope(
                id=str(payload.get("id") or uuid.uuid4().hex),
                modality=str(payload.get("modality") or "vision"),
                features=features,
                confidence=float(payload.get("confidence", 0.7)),
                source=str(payload.get("source") or "external"),
                t_tau=float(payload.get("t_tau", self.timekeeper.tau_now())),
                tags=list(payload.get("tags", [])),
            )
            return envelope, None

        features_raw = ctx.get("sense_features", {})
        features = (
            clamp_features(features_raw) if isinstance(features_raw, Mapping) else {}
        )
        tags = list(ctx.get("sense_tags") or [])
        confidence = float(ctx.get("sense_confidence", 0.7))
        modality = str(ctx.get("sense_modality") or "vision")
        summary: Optional[GazeSummary] = None

        gaze_payload = ctx.get("gaze") or ctx.get("gaze_lle")
        if gaze_payload and self._gaze_cfg is not None:
            gaze_result = extract_gaze_features(
                gaze_payload,
                self._gaze_cfg,
                t_tau=self.timekeeper.tau_now(),
            )
            if gaze_result:
                gaze_envelope: SenseEnvelope = gaze_result["envelope"]
                summary = gaze_result["summary"]
                features.update(gaze_envelope.features)
                tags.extend(gaze_envelope.tags)
                confidence = max(confidence, gaze_envelope.confidence)
                modality = gaze_envelope.modality

        if not features:
            prompt_text = (
                ctx.get("observation")
                or ctx.get("prompt")
                or ctx.get("last_user")
                or ctx.get("input_text")
                or ""
            )
            if isinstance(prompt_text, str) and prompt_text:
                features = infer_features_from_text(prompt_text)
        if not features:
            return None, summary

        envelope = SenseEnvelope(
            id=uuid.uuid4().hex,
            modality=modality,
            features=features,
            confidence=confidence,
            source=str(ctx.get("sense_source") or "external"),
            t_tau=float(ctx.get("sense_tau") or self.timekeeper.tau_now()),
            tags=list(dict.fromkeys(tags)),
        )
        return envelope, summary

    def _update_gaze_context(
        self,
        summary: GazeSummary,
        time_info: Mapping[str, Any],
        ctx_time: Dict[str, Any],
    ) -> Dict[str, Any]:
        d_tau = float(time_info.get("d_tau", 0.0))
        interest_report_dict: Optional[Dict[str, object]] = None
        snapshot: Dict[str, float] = {}
        if self._interest_tracker is not None:
            interest_report = self._interest_tracker.update(summary, d_tau)
            if interest_report is not None:
                interest_report_dict = interest_report.to_dict()
            snapshot = self._interest_tracker.snapshot()
        group_report = None
        if self._group_attention is not None and snapshot:
            group_report = self._group_attention.update({"self": snapshot}, d_tau)
        modulation = self._compute_gaze_modulation(summary, interest_report_dict)
        if modulation:
            ctx_time["gaze_modulation"] = modulation
        if (
            interest_report_dict
            and interest_report_dict.get("phase") == "focusing"
            and interest_report_dict.get("object_id")
        ):
            hint_label = sanitize_referent_label(summary.label or summary.target_id)
            if hint_label:
                ctx_time["referent_hint"] = {
                    "object_id": interest_report_dict["object_id"],
                    "label": hint_label,
                    "confidence": float(
                        max(
                            summary.confidence,
                            float(interest_report_dict.get("interest", 0.0)),
                        )
                    ),
                }
        context = {
            "summary": summary.to_dict(),
            "interest": interest_report_dict,
            "group": group_report,
            "modulation": modulation,
        }
        ctx_time["gaze_focus"] = context
        return context

    def _compute_gaze_modulation(
        self,
        summary: GazeSummary,
        interest_report: Optional[Dict[str, object]],
    ) -> Dict[str, object]:
        if not self._gaze_adapter_cfg:
            return {}
        pause_scale = int(
            float(self._gaze_adapter_cfg.get("pause_ms_scale", 0.0))
            * float(summary.gaze_on_me)
        )
        directness_delta = -float(
            self._gaze_adapter_cfg.get("directness_scale", 0.0)
        ) * float(summary.gaze_on_me)
        if interest_report and interest_report.get("phase") == "focusing":
            pause_scale += int(self._gaze_adapter_cfg.get("focus_bonus_ms", 0))
        modulation: Dict[str, object] = {}
        if pause_scale:
            modulation["pause_ms_add"] = pause_scale
        if abs(directness_delta) > 1e-6:
            modulation["directness_add"] = directness_delta
        if (
            interest_report
            and interest_report.get("phase") == "orienting"
            and summary.target_id
            and int(self._gaze_adapter_cfg.get("max_questions", 0)) > 0
        ):
            modulation["ask_bias"] = 1
        return modulation

    @staticmethod
    def _inject_disclosure(text: str, payload: Mapping[str, Any]) -> str:
        if not text:
            text = ""
        additions: List[str] = []
        disclosure = str(payload.get("disclosure") or "").strip()
        if disclosure:
            additions.append(disclosure)
        for phrase in payload.get("bridges", []) or []:
            if phrase:
                additions.append(str(phrase))
        for ask in payload.get("asks", []) or []:
            if ask:
                additions.append(str(ask))
        if not additions:
            return text
        addition_text = "\n".join(additions).strip()
        if not addition_text:
            return text
        if not text.strip():
            return addition_text
        return f"{text.rstrip()}\n\n{addition_text}"

    def _update_self_posteriors(
        self,
        chosen_mode: Optional[str],
        d_tau: float,
        ctx_time: Mapping[str, Any],
        gaze_context: Optional[Dict[str, Any]],
    ) -> Tuple[Dict[str, float], float]:
        if not hasattr(self, "self_roles"):
            return {}, 1.0
        self.self_roles.decay(d_tau)
        evidence: Dict[str, float] = {}
        role_hint = ctx_time.get("role_evidence")
        if isinstance(role_hint, Mapping):
            for role, value in role_hint.items():
                try:
                    evidence[str(role)] = evidence.get(str(role), 0.0) + float(value)
                except Exception:
                    continue
        if chosen_mode and chosen_mode in self.self_roles.roles:
            evidence[chosen_mode] = evidence.get(chosen_mode, 0.0) + 0.4
        if gaze_context:
            summary = gaze_context.get("summary") or {}
            try:
                gaze_on_me = float(summary.get("gaze_on_me", 0.0))
            except Exception:
                gaze_on_me = 0.0
            if gaze_on_me >= 0.6:
                evidence["caregiver"] = evidence.get("caregiver", 0.0) + 0.2
        self.self_roles.nudge(evidence)
        roles_post = self.self_roles.posterior()
        narrative_events = ctx_time.get("narrative_events")
        if isinstance(narrative_events, list):
            parsed_events: List[Tuple[str, str]] = []
            for event in narrative_events:
                if isinstance(event, Mapping):
                    kind = event.get("type")
                    target = event.get("target", "")
                elif isinstance(event, (list, tuple)) and len(event) >= 1:
                    kind = event[0]
                    target = event[1] if len(event) >= 2 else ""
                else:
                    continue
                parsed_events.append((str(kind), str(target)))
            if parsed_events:
                self.self_narr.update(parsed_events)
        coherence = self.self_narr.coherence()
        return roles_post, coherence

    def _gather_do_signals(
        self,
        plan: Mapping[str, Any],
        autop_metrics: Mapping[str, float],
        sense_residual: Mapping[str, Any],
        narrative_coherence: float,
    ) -> Dict[str, float]:
        plan_mood = plan.get("mood", {})
        mood = plan_mood if isinstance(plan_mood, Mapping) else {}
        bio_signals = plan.get("bio_signals", {}) if isinstance(plan, Mapping) else {}
        if not isinstance(bio_signals, Mapping):
            bio_signals = {}
        inflammation = float(bio_signals.get("inflammation_global", 0.0))
        energy = float(bio_signals.get("energy", 0.5))
        color_act = max(inflammation, abs(energy - 0.5) * 2.0)
        affect_val = float(
            abs(mood.get("v", 0.0)) + abs(mood.get("a", 0.0)) + abs(mood.get("u", 0.0))
        )
        ju_act = min(1.0, affect_val / 3.0)
        so_act = float(sense_residual.get("delta", 0.0)) if sense_residual else 0.0
        jerk_rate = float(autop_metrics.get("jerk_rate", 0.0)) if autop_metrics else 0.0
        gyo_act = max(0.0, min(1.0, jerk_rate))
        shiki_act = max(0.0, min(1.0, narrative_coherence))
        return {
            "色": color_act,
            "受": ju_act,
            "想": so_act,
            "行": gyo_act,
            "識": shiki_act,
        }

    @staticmethod
    def _compute_isolation_index(gaze_context: Optional[Dict[str, Any]]) -> float:
        if not gaze_context:
            return 0.0
        group_report = gaze_context.get("group")
        if not isinstance(group_report, Mapping):
            return 0.0
        values: List[float] = []
        for data in group_report.values():
            if isinstance(data, Mapping):
                try:
                    values.append(float(data.get("group_interest", 0.0)))
                except Exception:
                    continue
        if not values:
            return 0.0
        peak = max(values)
        return float(max(0.0, 1.0 - min(1.0, peak)))

    def _update_sunyata_metrics(
        self,
        d_tau: float,
        ctx_time: Dict[str, Any],
        plan: Mapping[str, Any],
        autop_metrics: Mapping[str, float],
        sense_residual: Mapping[str, Any],
        roles_post: Dict[str, float],
        narrative_coherence: float,
        gaze_context: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        if not self.sunyata_cfg:
            return {}
        ctx_time.pop("candor_force_level", None)
        signals = self._gather_do_signals(
            plan, autop_metrics, sense_residual, narrative_coherence
        )
        self.do_graph.update(signals, d_tau)
        block: Dict[str, Any] = {"do_topk": self.do_graph.topk()}
        rigidity_kld = 0.0
        if roles_post:
            rigidity_kld = role_kld(roles_post, self._last_roles_post)
            self._last_roles_post = dict(roles_post)
        isolation_index = self._compute_isolation_index(gaze_context)
        block["rigidity_kld"] = round(rigidity_kld, 4)
        block["isolation_index"] = round(isolation_index, 4)
        clinging_cfg = self.sunyata_cfg.get("clinging", {})
        actions_cfg = self.sunyata_cfg.get("actions", {})
        rigid_th = float(clinging_cfg.get("rigidity_kld_max", 1.0))
        iso_th = float(clinging_cfg.get("isolation_max", 1.0))
        tau_now = float(ctx_time.get("time", {}).get("tau", self.timekeeper.tau_now()))
        cooldown_tau = float(clinging_cfg.get("cooldown_tau", 0.0))
        consecutive_needed = max(1, int(clinging_cfg.get("consecutive_needed", 1)))
        threshold_hit = bool(rigidity_kld >= rigid_th or isolation_index >= iso_th)
        cooling = cooldown_tau > 0.0 and tau_now < getattr(
            self, "_clinging_cooldown_until", 0.0
        )
        if threshold_hit and not cooling:
            self._clinging_consecutive += 1
        else:
            self._clinging_consecutive = (
                0 if not threshold_hit else self._clinging_consecutive
            )
        triggered = (
            threshold_hit
            and not cooling
            and self._clinging_consecutive >= consecutive_needed
        )
        if triggered:
            self._clinging_consecutive = 0
            if cooldown_tau > 0.0:
                self._clinging_cooldown_until = tau_now + cooldown_tau
            boost = actions_cfg.get("forgetting_boost")
            if boost is not None:
                try:
                    self.forgetting.set_bias(sunyata=float(boost))
                except Exception:
                    pass
            urk_min = actions_cfg.get("urk_msv_min")
            if urk_min is not None and hasattr(self.replay_executor, "set_min_steps"):
                try:
                    self.replay_executor.set_min_steps(int(urk_min))
                except Exception:
                    pass
            candor_level = actions_cfg.get("candor_level")
            if candor_level:
                ctx_time["candor_force_level"] = candor_level
        block["clinging_triggered"] = triggered
        affect_honesty = max(
            0.0, min(1.0, 1.0 - float(sense_residual.get("delta", 0.0)))
        )
        attn_alignment = max(0.0, min(1.0, 1.0 - isolation_index))
        rigid_norm = rigidity_kld / max(rigid_th, 1e-6)
        iso_norm = isolation_index / max(iso_th, 1e-6)
        clinging_risk = max(0.0, min(1.0, (rigid_norm + iso_norm) / 2.0))
        block["s_index"] = {
            "affect_honesty": round(affect_honesty, 4),
            "self_continuity": round(narrative_coherence, 4),
            "clinging_risk": round(clinging_risk, 4),
            "attentional_alignment": round(attn_alignment, 4),
        }
        return block


def _apply_mood_hint(
    mood: Dict[str, float], hint: Dict[str, float]
) -> Dict[str, float]:
    updated = dict(mood)
    if "dv" in hint:
        updated["v"] = float(max(-1.0, min(1.0, updated.get("v", 0.0) + hint["dv"])))
    if "da" in hint:
        updated["a"] = float(max(-1.0, min(1.0, updated.get("a", 0.0) + hint["da"])))
    return updated


def _apply_green_delta(
    mood: Dict[str, float], delta: Dict[str, float]
) -> Dict[str, float]:
    updated = dict(mood)

    def _bounded(val: float) -> float:
        return float(max(-1.0, min(1.0, val)))

    if "v" in delta:
        updated["v"] = _bounded(updated.get("v", 0.0) + delta["v"])
    if "a" in delta:
        updated["a"] = _bounded(updated.get("a", 0.0) + delta["a"])
    if "d" in delta:
        updated["d"] = _bounded(updated.get("d", 0.0) + delta["d"])
    if "n" in delta:
        updated["novelty"] = _bounded(updated.get("novelty", 0.0) + delta["n"])
    if "c" in delta:
        updated["certainty"] = _bounded(updated.get("certainty", 0.0) + delta["c"])
        updated["u"] = _bounded(updated.get("u", 0.0) - delta["c"])
    if "e" in delta:
        updated["effort"] = _bounded(updated.get("effort", 0.0) + delta["e"])
    if "s" in delta:
        updated["social"] = _bounded(updated.get("social", 0.0) + delta["s"])
    return updated


__all__ = ["Hub"]
