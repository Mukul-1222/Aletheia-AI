from __future__ import annotations

import copy
import gc
import hashlib
import json
import math
import os
import random
import re
import time
import unicodedata
import shutil
import tempfile
from dataclasses import asdict
from contextlib import contextmanager, nullcontext
from pathlib import Path

import torch
import torch.nn.functional as F

from .capabilities import Capability, CapabilityRegistry
from .config import AgentConfig, Paths
from .memory import ExperienceMemory, lexical_terms
from .curriculum import Lesson, iter_parquet_lessons, iter_text_files, prepare_language_text, read_text_file, split_lesson_text
from .documents import is_document
from .dialogue import format_dialogue_example, format_turn_prompt, parse_dialogue_example, split_dialogue_response_pairs
from .eval_corpus import build_reference
from .evaluation import best_byte_baseline_bpb, load_reference, corpus_fingerprint
from .evaluation_protocol import summarize_bpb_rows
from .model import AletheiaModel
from .tokenizer import PAD, ByteFallbackTokenizer, LearnedBPETokenizer, load_tokenizer
from .web import Page, WebAccessError, WebExplorer
from .objectives import ObjectiveManager
from .perception import PerceptionSystem
from .voice import VoiceOutput
from .speech import SpeechRecognizer, SpeakerIdentityMemory, StreamingSpeechSession
from .evolution import EvolutionLab, GenerationalEvolution
from .planner import WorldModelPlanner
from .objective_value import ObjectiveOutcomeLearner
from .browser import BrowserEnvironment, BrowserEnvironmentError, BrowserObservation
from .tts import LearnedVoiceSynthesizer
from .outcomes import OpenWorldOutcomeEngine, OpenWorldSignals


def _sha256_file(path: Path) -> str | None:
    """Hash a file without loading the entire artifact into memory."""
    try:
        digest = hashlib.sha256()
        with Path(path).open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()
    except OSError:
        return None


def _sha256_model_state(state_dict: dict) -> str:
    """Content fingerprint for in-memory weights, independent of checkpoint metadata."""
    digest = hashlib.sha256()
    for name, value in sorted(state_dict.items()):
        digest.update(str(name).encode("utf-8", errors="replace"))
        digest.update(b"\0")
        if not isinstance(value, torch.Tensor):
            digest.update(repr(value).encode("utf-8", errors="replace"))
            digest.update(b"\n")
            continue
        tensor = value.detach().to(device="cpu").contiguous()
        digest.update(str(tensor.dtype).encode("ascii", errors="replace"))
        digest.update(repr(tuple(tensor.shape)).encode("ascii"))
        # Reinterpreting as bytes supports bfloat16 as well as ordinary float/integer tensors.
        byte_array = tensor.reshape(-1).view(torch.uint8).numpy()
        # hashlib accepts the NumPy buffer directly; avoiding .tobytes() prevents a second
        # full-size copy of model weights during repeated evaluation gates.
        digest.update(memoryview(byte_array))
    return digest.hexdigest()


def _atomic_write_text(path: Path, text: str) -> None:
    """Write UTF-8 text atomically to avoid truncated continuity/config records."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    os.close(fd)
    temp_path = Path(temp_name)
    try:
        with temp_path.open("w", encoding="utf-8", newline="") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_path, path)
    finally:
        try:
            temp_path.unlink(missing_ok=True)
        except OSError:
            pass



class AletheiaAgent:
    """From-scratch continuous learner. World-specific choices live in configuration."""

    def __init__(self, root: Path, config: AgentConfig):
        self.root = root.resolve()
        self.config = config
        # The fixed reference is part of this project instance. Reject absolute paths,
        # traversal, and symlinks that resolve outside the instance before any module
        # can read it for evaluation or use it to filter training data.
        self._evaluation_path()
        self.paths = Paths(self.root)
        self.paths.data.mkdir(parents=True, exist_ok=True)
        tokenizer_rel = str(getattr(config.runtime, "tokenizer_path", "data/tokenizer.json"))
        tokenizer_path = Path(tokenizer_rel)
        self.tokenizer_path = tokenizer_path if tokenizer_path.is_absolute() else (self.root / tokenizer_path)
        self.tokenizer = load_tokenizer(self.tokenizer_path)
        # The active tokenizer is authoritative for the model vocabulary. A missing
        # tokenizer falls back to exact byte encoding, so old installations remain usable.
        self.config.model.vocab_size = int(self.tokenizer.vocab_size)
        self._tokenizer_fingerprint = hashlib.sha256(
            json.dumps(self.tokenizer.state(), sort_keys=True).encode("utf-8")
        ).hexdigest()
        self.open_world_outcomes = OpenWorldOutcomeEngine()

        requested_device = str(config.runtime.device).lower().strip()
        if requested_device not in {"auto", "cpu", "cuda"}:
            raise ValueError("runtime.device must be one of: auto, cpu, cuda")
        if requested_device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but torch.cuda.is_available() is False")
        self.device = torch.device("cuda" if requested_device == "cuda" or (requested_device == "auto" and torch.cuda.is_available()) else "cpu")

        self.registry = CapabilityRegistry()
        self.memory = ExperienceMemory(
            self.paths.memory, config.runtime.memory_limit, config.runtime.max_frontier,
            kind_limits=getattr(config.runtime, "memory_kind_limits", {}),
            eviction_order=getattr(config.runtime, "memory_eviction_order", []),
        )
        self.objectives = ObjectiveManager(self.paths.data / "objectives.json")
        self.perception = PerceptionSystem(
            self.root,
            self.device,
            embedding_dim=int(getattr(config.runtime, "perception_embedding_dim", config.model.perception_dim)),
            batch_size=int(getattr(config.runtime, "perception_batch_size", 32)),
            buffer_size=int(getattr(config.runtime, "perception_buffer_size", 256)),
            learning_rate=float(getattr(config.runtime, "perception_learning_rate", 2e-4)),
            min_visual_delta=float(getattr(config.runtime, "perception_min_visual_delta", 0.02)),
            identity_threshold=float(getattr(config.runtime, "identity_threshold", 0.92)),
            identity_max_exemplars=int(getattr(config.runtime, "perception_identity_exemplars", 4)),
            identity_reindex_interval=int(getattr(config.runtime, "perception_identity_reindex_interval", 8)),
            object_slots=int(getattr(config.runtime, "vision_object_slots", 6)),
            object_slot_dim=int(getattr(config.runtime, "vision_object_slot_dim", 128)),
            object_iterations=int(getattr(config.runtime, "vision_object_iterations", 3)),
            object_max=int(getattr(config.runtime, "vision_object_max", 6)),
            object_min_mass=float(getattr(config.runtime, "vision_object_min_mass", 0.02)),
            tracker_max_tracks=int(getattr(config.runtime, "vision_tracker_max_tracks", 32)),
            tracker_match_threshold=float(getattr(config.runtime, "vision_tracker_match_threshold", 0.55)),
            tracker_max_age=int(getattr(config.runtime, "vision_tracker_max_age", 6)),
            tracker_ema=float(getattr(config.runtime, "vision_tracker_ema", 0.25)),
            entity_threshold=float(getattr(config.runtime, "vision_entity_threshold", 0.84)),
            entity_ema=float(getattr(config.runtime, "vision_entity_ema", 0.15)),
            entity_max_entities=int(getattr(config.runtime, "vision_entity_max_entities", 5000)),
        )
        self.perception._buffer_persist_interval = max(1, int(getattr(config.runtime, "perception_buffer_persist_interval", 32)))
        self.voice = VoiceOutput()
        self.speech = SpeechRecognizer(
            self.root, self.device,
            learning_rate=float(getattr(config.runtime, "speech_learning_rate", 2e-4)),
            ready_min_steps=int(getattr(config.runtime, "speech_ready_min_steps", 100)),
            ready_max_cer=float(getattr(config.runtime, "speech_ready_max_cer", 0.35)),
            ready_min_eval_examples=int(getattr(config.runtime, "speech_ready_min_eval_examples", 8)),
            holdout_fraction=float(getattr(config.runtime, "speech_holdout_fraction", 0.20)),
            batch_size=int(getattr(config.runtime, "speech_batch_size", 4)),
            eval_max_examples=int(getattr(config.runtime, "speech_eval_max_examples", 256)),
            augmentation=bool(getattr(config.runtime, "speech_augmentation", True)),
            quarantine_corrupt=bool(getattr(config.runtime, "speech_checkpoint_quarantine", True)),
        )
        self.speakers = SpeakerIdentityMemory(
            self.paths.data / "speaker_memory.json",
            threshold=float(getattr(config.runtime, "speaker_threshold", 0.78)),
            max_exemplars=int(getattr(config.runtime, "speaker_max_exemplars", 8)),
        )
        self.tts = LearnedVoiceSynthesizer(
            self.root,
            self.device,
            speaker_encoder=self.perception.audio,
            learning_rate=float(getattr(config.runtime, "tts_learning_rate", 2e-4)),
            holdout_fraction=float(getattr(config.runtime, "tts_holdout_fraction", 0.20)),
            ready_min_steps=int(getattr(config.runtime, "tts_ready_min_steps", 25)),
            ready_max_loss=float(getattr(config.runtime, "tts_ready_max_loss", 1.0)),
            ready_min_eval_examples=int(getattr(config.runtime, "tts_ready_min_eval_examples", 8)),
            sample_rate=int(getattr(config.runtime, "tts_sample_rate", 16000)),
            n_fft=int(getattr(config.runtime, "tts_n_fft", 400)),
            hop_length=int(getattr(config.runtime, "tts_hop_length", 160)),
            n_mels=int(getattr(config.runtime, "tts_n_mels", 80)),
            speaker_dim=int(getattr(config.runtime, "tts_speaker_dim", config.runtime.perception_embedding_dim)),
            text_dim=int(getattr(config.runtime, "tts_text_dim", 128)),
            encoder_hidden=int(getattr(config.runtime, "tts_encoder_hidden", 192)),
            decoder_hidden=int(getattr(config.runtime, "tts_decoder_hidden", 256)),
            attention_dim=int(getattr(config.runtime, "tts_attention_dim", 128)),
            griffin_lim_iterations=int(getattr(config.runtime, "tts_griffin_lim_iterations", 32)),
            max_synthesis_frames=int(getattr(config.runtime, "tts_max_synthesis_frames", 240)),
        )
        self.evolution = EvolutionLab(self.root)
        self.generational_evolution = GenerationalEvolution(self.root)
        self.web = WebExplorer(
            config.web,
            config.runtime.web_timeout,
            config.runtime.web_max_page_chars,
            config.runtime.user_agent,
            config.runtime.web_min_host_interval,
            config.runtime.web_max_retries,
            config.runtime.web_max_redirects,
            config.runtime.web_max_download_bytes,
            config.runtime.web_max_pdf_chars,
            config.runtime.web_plain_text_default_encoding,
            getattr(config.runtime, "web_crawl_exclude_patterns", []),
            float(getattr(config.runtime, "browser_dns_cache_seconds", 30.0)),
        )
        self.browser = None
        if bool(getattr(config.runtime, "browser_enabled", False)):
            self.browser = BrowserEnvironment(
                self.root,
                self.web,
                headless=bool(getattr(config.runtime, "browser_headless", True)),
                executable_path=str(getattr(config.runtime, "browser_executable_path", "")),
                user_data_dir=str(getattr(config.runtime, "browser_user_data_dir", "data/browser_profile")),
                timeout_ms=int(getattr(config.runtime, "browser_timeout_ms", 20000)),
                respect_robots=bool(getattr(config.runtime, "browser_respect_robots", True)),
                screenshot_max_bytes=int(getattr(config.runtime, "browser_screenshot_max_bytes", 8 * 1024 * 1024)),
                persistent_profile=bool(getattr(config.runtime, "browser_persistent_profile", False)),
                allow_domains=list(getattr(config.runtime, "browser_allow_domains", [])),
                allow_form_submit=bool(getattr(config.runtime, "browser_allow_form_submit", False)),
                allow_downloads=bool(getattr(config.runtime, "browser_allow_downloads", False)),
                allow_file_uploads=bool(getattr(config.runtime, "browser_allow_file_uploads", False)),
                allow_authentication=bool(getattr(config.runtime, "browser_allow_authentication", False)),
                dns_cache_seconds=float(getattr(config.runtime, "browser_dns_cache_seconds", 30.0)),
                egress_proxy_url=str(getattr(config.runtime, "browser_egress_proxy_url", "")),
                egress_proxy_policy_confirmed=bool(getattr(config.runtime, "browser_egress_proxy_policy_confirmed", False)),
            )
        self._last_browser_observation: dict = {}
        self._last_browser_action: dict = {}
        self._register_capabilities()
        action_names = self.registry.names()
        self.model = AletheiaModel(config.model, action_names, tokenizer=self.tokenizer).to(self.device)
        self.optimizer = self._make_optimizer(config, self.model)
        self.amp_enabled = bool(config.runtime.mixed_precision and self.device.type == "cuda")
        self.amp_dtype = self._select_amp_dtype(self.device) if self.amp_enabled else None
        scaler_enabled = self.amp_enabled and self.amp_dtype == torch.float16
        try:
            self.scaler = torch.amp.GradScaler("cuda", enabled=scaler_enabled)
        except (AttributeError, TypeError):
            self.scaler = torch.cuda.amp.GradScaler(enabled=scaler_enabled)
        self.training_steps = 0
        self.dialogue_examples_seen = 0
        self.last_dialogue_accepted: bool | None = None
        self.last_dialogue_fixed_gain_delta: float | None = None
        self.last_lesson_accepted: bool | None = None
        self.last_lesson_fixed_gain_delta: float | None = None
        checkpoint_backup = self.paths.model.with_suffix(self.paths.model.suffix + ".bak")
        if self.paths.model.exists():
            try:
                loaded_model = AletheiaModel.load(
                    self.paths.model, self.device, config.model, action_names, self.optimizer, tokenizer=self.tokenizer
                )
                saved_optimizer_state = self.optimizer.state_dict()
                self.model = loaded_model
                self.optimizer = self._make_optimizer(config, self.model)
                try:
                    self.optimizer.load_state_dict(saved_optimizer_state)
                except (ValueError, RuntimeError):
                    pass
            except Exception as primary_error:
                recovered = False
                if checkpoint_backup.exists():
                    try:
                        loaded_model = AletheiaModel.load(
                            checkpoint_backup, self.device, config.model, action_names, self.optimizer, tokenizer=self.tokenizer
                        )
                        saved_optimizer_state = self.optimizer.state_dict()
                        self.model = loaded_model
                        self.optimizer = self._make_optimizer(config, self.model)
                        try:
                            self.optimizer.load_state_dict(saved_optimizer_state)
                        except (ValueError, RuntimeError):
                            pass
                        stamp = time.strftime("%Y%m%d_%H%M%S")
                        corrupt = self.paths.data / f"corrupt_checkpoint_{stamp}.pt"
                        if self.paths.model.exists():
                            shutil.move(str(self.paths.model), str(corrupt))
                        shutil.copy2(checkpoint_backup, self.paths.model)
                        self._checkpoint_recovery_message = (
                            f"Recovered from backup checkpoint after primary load failed: {primary_error!r}"
                        )
                        recovered = True
                    except Exception as backup_error:
                        self._checkpoint_recovery_message = (
                            f"Primary checkpoint failed ({primary_error!r}); backup also failed ({backup_error!r})"
                        )
                if not recovered:
                    stamp = time.strftime("%Y%m%d_%H%M%S")
                    suffix = "incompatible" if isinstance(primary_error, ValueError) else "corrupt"
                    quarantine = self.paths.data / f"{suffix}_checkpoint_{stamp}.pt"
                    if self.paths.model.exists():
                        shutil.move(str(self.paths.model), str(quarantine))
                    self._checkpoint_recovery_message = (
                        f"Preserved unusable checkpoint at {quarantine}: {primary_error!r}"
                    )

        self.planner = WorldModelPlanner(
            self.root,
            self.device,
            self.model,
            action_names,
            horizon=int(getattr(config.runtime, "planning_horizon", 3)),
            beam_width=int(getattr(config.runtime, "planning_beam_width", 6)),
            branch_factor=int(getattr(config.runtime, "planning_branch_factor", 4)),
            goal_weight=float(getattr(config.runtime, "planning_goal_weight", 0.65)),
            alignment_weight=float(getattr(config.runtime, "planning_alignment_weight", 0.20)),
            policy_weight=float(getattr(config.runtime, "planning_policy_weight", 0.15)),
            uncertainty_weight=float(getattr(config.runtime, "planning_uncertainty_weight", 0.08)),
            discount=float(getattr(config.runtime, "planning_discount", 0.90)),
            learning_rate=float(getattr(config.runtime, "planning_learning_rate", 2e-4)),
            hidden=int(getattr(config.runtime, "planning_value_hidden", 256)),
            enabled=bool(getattr(config.runtime, "planning_enabled", True)),
        )
        self.objective_value = ObjectiveOutcomeLearner(
            self.root,
            self.device,
            state_dim=int(config.model.recurrent_dim),
            action_dim=int(config.model.dim),
            hidden=int(getattr(config.runtime, "objective_value_hidden", 192)),
            learning_rate=float(getattr(config.runtime, "objective_value_learning_rate", 2e-4)),
            batch_size=int(getattr(config.runtime, "objective_value_batch_size", 32)),
            buffer_size=int(getattr(config.runtime, "objective_value_buffer_size", 4096)),
            min_examples=int(getattr(config.runtime, "objective_value_min_examples", 32)),
        )
        self._last_plan: dict = {}
        self.state = self.model.initial_state(1, self.device)
        self._chat_turns_since_checkpoint = 0
        self.previous_action: int | None = None
        self.prediction_error = 0.0
        self.policy_baseline = 0.0
        self.cycles = 0
        # total_tokens counts unique lesson bytes/tokens, independent of epoch repetition.
        # processed_tokens measures actual optimizer exposure, including replay and epochs.
        self.total_tokens = 0
        self.processed_tokens = 0
        self._unique_training_fingerprints: set[str] = set()
        self.last_language_loss: float | None = None
        self.last_replay_stats: dict = {"requested": 0, "used": 0, "lesson_count": 0, "lesson_ids": [], "per_epoch": []}
        self.last_teaching_loss: float | None = None
        self.last_lesson_id: int | None = None
        self.last_eval_bpb: float | None = None
        self.eval_baseline_bpb: float | None = None
        self.eval_reference_bytes = 0
        self.last_holdout_bpb: float | None = None
        self.pretrain_shards_seen = 0
        self.pretrain_bytes_seen = 0
        self._last_policy_mask: torch.Tensor | None = None
        self._last_action_state: torch.Tensor | None = None
        self._last_action_state_after: torch.Tensor | None = None
        self._last_action_selection_mode: str | None = None
        self._last_action_index: int | None = None
        self._last_feedback_target: dict | None = None
        self._cognitive_step = 0
        self._last_attention_uncertainty = 0.0
        self._last_cognitive_event_id: int | None = None
        self._last_initiative_time = 0.0
        self._last_visual_embedding: torch.Tensor | None = None
        self._last_recorded_visual_time = 0.0
        self._last_speaker_id: str | None = None
        self._last_speaker_score: float = 0.0
        self._last_online_dialogue: dict = {}
        self._evaluation_baseline_cache: dict | None = None
        self._evaluation_overlap_cache: dict | None = None
        self._training_mode = "continual"
        self._schedule_total_steps = 0
        self._schedule_step_start = 0
        self._schedule_warmup_steps = 0
        self._schedule_decay_steps = 0
        self._schedule_min_lr_ratio = 1.0
        self._schedule_peak_lrs: list[float] = []
        self._ema_shadow: dict[str, torch.Tensor] | None = None
        self._load_perception_state()
        self._load_runtime_state()

    def _evaluation_path(self) -> Path:
        """Resolve the configured fixed evaluation file strictly inside this project."""
        configured = Path(str(self.config.runtime.language_eval_path)).expanduser()
        if configured.is_absolute():
            raise ValueError("runtime.language_eval_path must be a project-relative path")
        resolved = (self.root / configured).resolve()
        try:
            resolved.relative_to(self.root)
        except ValueError as exc:
            raise ValueError("runtime.language_eval_path must remain inside the Aletheia root") from exc
        return resolved

    def _is_fixed_evaluation_source(self, source: str | Path) -> bool:
        """Detect the corpus and its sidecars by source path, not text similarity alone."""
        raw = str(source)
        # Parquet lesson source labels may append a row suffix to the source path.
        raw_path = raw.split("#parquet", 1)[0]
        candidate = Path(raw_path).expanduser()
        if not candidate.is_absolute():
            candidate = self.root / candidate
        try:
            resolved = candidate.resolve()
            evaluation = self._evaluation_path()
        except (OSError, RuntimeError, ValueError):
            # Invalid or unresolvable source identities must not be trusted for training.
            return True
        protected = {
            evaluation,
            evaluation.with_suffix(".sources.json"),
            evaluation.with_suffix(".paragraphs.json"),
        }
        if resolved in protected:
            return True
        evaluation_dir = evaluation.parent
        return evaluation_dir != self.root and evaluation_dir in resolved.parents

    def _register_capabilities(self) -> None:
        handlers = {
            "discover": (self._cap_discover, lambda: bool(self.config.web.bootstrap_discovery)),
            "search": (self._cap_search, lambda: bool(self.config.web.search_providers) and bool(self.memory.query_seed())),
            "follow": (self._cap_follow, lambda: self.memory.next_link() is not None),
            "revisit": (self._cap_revisit, lambda: bool(self.memory.recent(1, kinds=("external",)))),
            "reflect": (self._cap_reflect, lambda: self._language_ready()),
            "consolidate": (self._cap_consolidate, lambda: bool(self.memory.recent(2, kinds=("external",)))),
            "pursue_objective": (self._cap_pursue_objective, lambda: self.objectives.current() is not None),
            "evolve": (self._cap_evolve, lambda: self.objectives.current() is not None or self.memory.lesson_count() > 0),
            "browser_interact": (
                self._cap_browser_interact,
                lambda: self.browser is not None and (
                    self.browser.last_observation is not None
                    or bool(str(getattr(self.config.runtime, "browser_start_url", "")).strip())
                ),
            ),
        }
        for name, settings in self.config.capabilities.items():
            item = handlers.get(name)
            if item is None:
                raise ValueError(f"no implementation exists for configured capability: {name}")
            handler, availability = item
            self.registry.register(
                Capability(
                    name=name,
                    description=settings.description,
                    run=handler,
                    available=availability,
                    external=settings.external,
                    produces_observation=settings.produces_observation,
                    requires_memory=settings.requires_memory,
                    requires_language=settings.requires_language,
                    warmup_priority=settings.warmup_priority,
                    policy_train=settings.policy_train,
                )
            )
        if not len(self.registry):
            raise ValueError("at least one capability must be enabled")

    def _load_runtime_state(self) -> None:
        path = self.paths.data / "continuity.json"
        if not path.exists():
            return
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            if payload.get("capabilities") != self.registry.names():
                return
            saved = payload.get("state")
            if saved:
                state = torch.tensor(saved, dtype=torch.float32, device=self.device)
                if tuple(state.shape) == tuple(self.state.shape):
                    self.state = state
            self.previous_action = payload.get("previous_action")
            self.prediction_error = float(payload.get("prediction_error", 0.0))
            self.policy_baseline = float(payload.get("policy_baseline", 0.0))
            self.cycles = int(payload.get("cycles", 0))
            # A legacy checkpoint stored processed exposure in total_tokens. Do not silently
            # promote that repeated-compute count into the new diversity gate.
            self.total_tokens = int(payload.get("unique_tokens", 0))
            self.processed_tokens = int(payload.get("processed_tokens", payload.get("total_tokens", 0)))
            saved_tokenizer = str(payload.get("tokenizer_fingerprint", ""))
            self._unique_training_fingerprints = {str(x) for x in payload.get("unique_training_fingerprints", []) if x}
            if saved_tokenizer and saved_tokenizer != self._tokenizer_fingerprint and bool(getattr(self.config.runtime, "tokenizer_reset_readiness_on_change", True)):
                self.total_tokens = 0
                self.processed_tokens = 0
                self._unique_training_fingerprints.clear()
                self.last_eval_bpb = None
                self.eval_baseline_bpb = None
            self.training_steps = int(payload.get("training_steps", 0))
            self.dialogue_examples_seen = int(payload.get("dialogue_examples_seen", 0))
            self.last_dialogue_accepted = payload.get("last_dialogue_accepted")
            self.last_dialogue_fixed_gain_delta = payload.get("last_dialogue_fixed_gain_delta")
            self.last_lesson_accepted = payload.get("last_lesson_accepted")
            self.last_lesson_fixed_gain_delta = payload.get("last_lesson_fixed_gain_delta")
            self.last_language_loss = payload.get("last_language_loss")
            self.last_teaching_loss = payload.get("last_teaching_loss")
            self.last_lesson_id = payload.get("last_lesson_id")
            self.last_eval_bpb = payload.get("last_eval_bpb")
            self.eval_baseline_bpb = payload.get("eval_baseline_bpb")
            self._last_feedback_target = payload.get("last_feedback_target")
            self.eval_reference_bytes = int(payload.get("eval_reference_bytes", 0))
            self.last_holdout_bpb = payload.get("last_holdout_bpb")
            self.pretrain_shards_seen = int(payload.get("pretrain_shards_seen", 0))
            self.pretrain_bytes_seen = int(payload.get("pretrain_bytes_seen", 0))
            self._cognitive_step = int(payload.get("cognitive_step", 0))
            self._last_attention_uncertainty = float(payload.get("last_attention_uncertainty", 0.0))
            self._last_cognitive_event_id = payload.get("last_cognitive_event_id")
            self._last_speaker_id = payload.get("last_speaker_id")
            self._last_speaker_score = float(payload.get("last_speaker_score", 0.0))
            self._last_online_dialogue = dict(payload.get("last_online_dialogue", {}))
        except Exception:
            return

    def _save_runtime_state(self) -> None:
        path = self.paths.data / "continuity.json"
        payload = {
            "state": self.state.detach().cpu().tolist(),
            "previous_action": self.previous_action,
            "prediction_error": self.prediction_error,
            "policy_baseline": self.policy_baseline,
            "cycles": self.cycles,
            "unique_tokens": self.total_tokens,
            "processed_tokens": self.processed_tokens,
            "unique_training_fingerprints": sorted(self._unique_training_fingerprints),
            "tokenizer_fingerprint": self._tokenizer_fingerprint,
            "tokenizer_kind": getattr(self.tokenizer, "name", "byte"),
            "tokenizer_vocab_size": int(self.tokenizer.vocab_size),
            "training_steps": self.training_steps,
            "dialogue_examples_seen": self.dialogue_examples_seen,
            "last_dialogue_accepted": self.last_dialogue_accepted,
            "last_dialogue_fixed_gain_delta": self.last_dialogue_fixed_gain_delta,
            "last_lesson_accepted": self.last_lesson_accepted,
            "last_lesson_fixed_gain_delta": self.last_lesson_fixed_gain_delta,
            "last_language_loss": self.last_language_loss,
            "last_teaching_loss": self.last_teaching_loss,
            "last_lesson_id": self.last_lesson_id,
            "last_eval_bpb": self.last_eval_bpb,
            "eval_baseline_bpb": self.eval_baseline_bpb,
            "last_feedback_target": self._last_feedback_target,
            "eval_reference_bytes": self.eval_reference_bytes,
            "last_holdout_bpb": self.last_holdout_bpb,
            "pretrain_shards_seen": self.pretrain_shards_seen,
            "pretrain_bytes_seen": self.pretrain_bytes_seen,
            "cognitive_step": self._cognitive_step,
            "last_attention_uncertainty": self._last_attention_uncertainty,
            "last_cognitive_event_id": self._last_cognitive_event_id,
            "last_speaker_id": self._last_speaker_id,
            "last_speaker_score": self._last_speaker_score,
            "last_online_dialogue": self._last_online_dialogue,
            "capabilities": self.registry.names(),
        }
        _atomic_write_text(path, json.dumps(payload, ensure_ascii=False))

    def _tokenizer_state_fingerprint(self, tokenizer=None) -> str:
        tokenizer = tokenizer or self.tokenizer
        return hashlib.sha256(
            json.dumps(tokenizer.state(), sort_keys=True).encode("utf-8")
        ).hexdigest()

    def _replace_language_tokenizer(self, tokenizer) -> dict:
        """Install a newly learned tokenizer while migrating compatible language weights."""
        if int(tokenizer.vocab_size) <= 259:
            raise ValueError("learned tokenizer must contain more than the 259-token byte fallback vocabulary")
        old_model = self.model
        old_state = old_model.state_dict()
        old_embedding = old_state.get("token_embedding.weight")
        cfg = copy.deepcopy(self.config.model)
        cfg.vocab_size = int(tokenizer.vocab_size)
        new_model = AletheiaModel(cfg, self.registry.names(), tokenizer=tokenizer).to(self.device)
        with torch.no_grad():
            new_state = new_model.state_dict()
            for key, value in old_state.items():
                target = new_state.get(key)
                if target is not None and tuple(target.shape) == tuple(value.shape) and key not in {"token_embedding.weight", "lm_head.weight", "position_embedding.weight"}:
                    target.copy_(value.to(target))
            if old_embedding is not None:
                old_embedding = old_embedding.detach().to(self.device)
                base_rows = min(259, old_embedding.size(0), new_model.token_embedding.weight.size(0))
                new_model.token_embedding.weight[:base_rows].copy_(old_embedding[:base_rows])
                raw_tokens = getattr(tokenizer, "token_bytes", [])
                old_raw_tokens = getattr(old_model.tokenizer, "token_bytes", [])
                old_by_bytes = {}
                for old_id in range(259, min(len(old_raw_tokens), old_embedding.size(0))):
                    raw_old = old_raw_tokens[old_id]
                    if raw_old and raw_old not in old_by_bytes:
                        old_by_bytes[raw_old] = old_id
                for token_id in range(259, new_model.token_embedding.weight.size(0)):
                    raw = raw_tokens[token_id] if token_id < len(raw_tokens) else b""
                    if not raw:
                        continue
                    prior_id = old_by_bytes.get(raw)
                    if prior_id is not None:
                        new_model.token_embedding.weight[token_id].copy_(old_embedding[prior_id])
                    elif old_embedding.size(0) >= 256:
                        ids = torch.tensor(list(raw), dtype=torch.long, device=self.device)
                        new_model.token_embedding.weight[token_id].copy_(old_embedding.index_select(0, ids).mean(dim=0))
                new_model.lm_head.weight = new_model.token_embedding.weight
            old_pos = old_state.get("position_embedding.weight")
            if old_pos is not None:
                rows = min(old_pos.size(0), new_model.position_embedding.weight.size(0))
                new_model.position_embedding.weight[:rows].copy_(old_pos[:rows].to(new_model.position_embedding.weight))
            # Action rows survive because the action interface is unchanged.
        self.tokenizer = tokenizer
        self.tokenizer_path.parent.mkdir(parents=True, exist_ok=True)
        self.tokenizer.save(self.tokenizer_path)
        self.config.model = cfg
        self.model = new_model
        self.optimizer = self._make_optimizer(self.config, self.model)
        self.planner.model = self.model
        self._tokenizer_fingerprint = self._tokenizer_state_fingerprint(tokenizer)
        self.total_tokens = 0
        self.processed_tokens = 0
        self._unique_training_fingerprints.clear()
        self.last_eval_bpb = None
        self.eval_baseline_bpb = None
        self.eval_reference_bytes = 0
        return {
            "kind": tokenizer.name,
            "vocab_size": int(tokenizer.vocab_size),
            "fingerprint": self._tokenizer_fingerprint,
        }

    def train_tokenizer(
        self,
        source: str | Path | None = None,
        *,
        vocab_size: int | None = None,
        min_frequency: int | None = None,
        max_training_bytes: int | None = None,
        max_segments: int | None = None,
    ) -> dict:
        """Learn a byte-fallback BPE tokenizer solely from supplied local text/data."""
        source_value = source or self.root
        source_path = Path(source_value).expanduser().resolve()
        budget = max(1, int(max_training_bytes or getattr(self.config.runtime, "tokenizer_train_max_bytes", 8_000_000)))
        target_vocab = max(260, int(vocab_size or getattr(self.config.runtime, "tokenizer_vocab_size", 1024)))
        min_freq = max(1, int(min_frequency or getattr(self.config.runtime, "tokenizer_min_frequency", 2)))
        segment_limit = max(1, int(max_segments or getattr(self.config.runtime, "tokenizer_max_segments", 75000)))
        texts: list[str] = []
        consumed = 0
        skipped: list[str] = []
        evaluation_path = self._evaluation_path()
        evaluation_root = evaluation_path.parent
        protected_eval_paths = {
            evaluation_path,
            evaluation_path.with_suffix(".sources.json"),
            evaluation_path.with_suffix(".paragraphs.json"),
        }

        def is_evaluation_asset(path: Path) -> bool:
            resolved = path.resolve()
            if resolved in protected_eval_paths:
                return True
            # Do not treat the project root as the excluded directory when someone
            # configures a root-level heldout.txt; that would exclude every lesson.
            return evaluation_root != self.root and (
                resolved == evaluation_root or evaluation_root in resolved.parents
            )

        if source_path.is_file() and is_evaluation_asset(source_path):
            raise ValueError("refusing to train tokenizer directly from fixed evaluation data or its manifest")

        def add_text(value: str) -> None:
            nonlocal consumed
            if consumed >= budget or not value:
                return
            # Protect fixed-heldout text even when it is copied into a differently
            # named local file or already exists in persisted lesson memory.
            safe_value = self._exclude_evaluation_text(str(value))
            if len(safe_value.strip()) < 8:
                return
            remaining = budget - consumed
            raw = safe_value.encode("utf-8", errors="replace")[:remaining]
            if not raw:
                return
            clipped = raw.decode("utf-8", errors="ignore")
            consumed += len(raw)
            if clipped:
                texts.append(clipped)

        candidates: list[Path] = []
        if source_path.is_file():
            candidates = [source_path]
        elif source_path.is_dir():
            candidates = sorted(
                path for path in source_path.rglob("*")
                if path.is_file() and (path.suffix.lower() in {".txt", ".md", ".markdown", ".html", ".htm", ".xhtml", ".json", ".csv", ".pdf", ".docx", ".epub"} or is_document(path))
            )
        else:
            raise FileNotFoundError(source_path)
        for path in candidates:
            if consumed >= budget:
                break
            if is_evaluation_asset(path):
                skipped.append(f"{path}: excluded fixed evaluation asset")
                continue
            try:
                if path.suffix.lower() == ".parquet":
                    remaining = budget - consumed
                    for lesson in iter_parquet_lessons(path, shard_max_chars=min(1_000_000, max(1, remaining)), text_column=self.config.runtime.parquet_text_column, max_source_bytes=remaining):
                        add_text(lesson.text)
                        if consumed >= budget:
                            break
                else:
                    lesson = read_text_file(path, max_chars=max(0, budget - consumed))
                    add_text(lesson.text)
            except Exception as exc:
                skipped.append(f"{path}: {exc}")
        # Existing learned lessons are valuable training material when the source path is the project.
        if source is None or source_path == self.root:
            for row in self.memory.recent(max(0, 256), kinds=("lesson", "external")):
                if consumed >= budget:
                    break
                add_text(str(row.get("text", "")))
        if not texts:
            raise ValueError("no text available for tokenizer training")
        tokenizer, stats = LearnedBPETokenizer.train(
            texts, vocab_size=target_vocab, min_frequency=min_freq,
            max_training_bytes=budget, max_segments=segment_limit,
        )
        installed = self._replace_language_tokenizer(tokenizer)
        result = {**stats, **installed, "source": str(source_path), "skipped": skipped[:16], "training_texts": len(texts)}
        self._record({"event": "tokenizer_training", **result})
        self.save()
        return result

    def _make_optimizer(self, config: AgentConfig, model: AletheiaModel) -> torch.optim.Optimizer:
        predictive_parameters = [
            parameter
            for name, parameter in model.named_parameters()
            if not name.startswith("action_head.")
        ]
        return torch.optim.AdamW(
            [
                {"params": predictive_parameters, "lr": config.runtime.learning_rate},
                {"params": model.action_head.parameters(), "lr": config.runtime.policy_learning_rate},
            ],
            weight_decay=config.runtime.weight_decay,
            betas=(float(config.runtime.adam_beta1), float(config.runtime.adam_beta2)),
        )

    def parameter_count(self) -> int:
        return sum(p.numel() for p in self.model.parameters())

    def _encode_token_chunks(self, token_ids: list[int], *, max_chunks: int = 0, context: int | None = None) -> list[torch.Tensor]:
        payload = max(1, int(context or self.model.cfg.context) - 2)
        body = [int(value) for value in token_ids if int(value) not in (257, 258, PAD)]
        chunks: list[torch.Tensor] = []
        starts = range(0, max(len(body), 1), payload)
        for start in starts:
            piece = body[start:start + payload]
            is_final_document_chunk = start + len(piece) >= len(body)
            ids = [257, *piece]
            if is_final_document_chunk:
                ids.append(258)
            chunks.append(torch.tensor(ids, dtype=torch.long, device=self.device).unsqueeze(0))
            if max_chunks and len(chunks) >= max_chunks:
                break
        if not chunks:
            chunks.append(torch.tensor([257, 258], dtype=torch.long, device=self.device).unsqueeze(0))
        return chunks

    def _encode_byte_chunks(self, raw: bytes, *, max_chunks: int = 0, context: int | None = None) -> list[torch.Tensor]:
        # Backward-compatible name: bytes are passed through the active tokenizer.
        token_ids = self.tokenizer.encode_bytes(raw, bos=False, eos=False)
        return self._encode_token_chunks(token_ids, max_chunks=max_chunks, context=context)

    def _encode_chunks(self, text: str, *, max_chars: int = 0, max_chunks: int = 0, context: int | None = None) -> list[torch.Tensor]:
        if max_chars and max_chars > 0:
            text = text[:max_chars]
        token_ids = self.tokenizer.encode(text, bos=False, eos=False)
        return self._encode_token_chunks(token_ids, max_chunks=max_chunks, context=context)

    def _encode_text(self, text: str) -> torch.Tensor:
        return self._encode_chunks(text, max_chunks=1)[0]

    def _memory_chunk_texts(self, text: str, *, context: int | None = None) -> list[str]:
        token_ids = self.tokenizer.encode(text, bos=False, eos=False)
        payload = max(1, int(context or self.model.cfg.context) - 2)
        pieces = [token_ids[start:start + payload] for start in range(0, max(len(token_ids), 1), payload)]
        return [self.tokenizer.decode(piece) for piece in pieces]

    @staticmethod
    def _select_amp_dtype(device: torch.device):
        """Prefer BF16 on CUDA devices that advertise support; otherwise use FP16."""
        if device.type != "cuda":
            return None
        try:
            if bool(torch.cuda.is_bf16_supported()):
                return torch.bfloat16
        except (AttributeError, RuntimeError):
            pass
        return torch.float16

    def _autocast(self):
        if not self.amp_enabled:
            return nullcontext()
        dtype = self.amp_dtype or torch.float16
        return torch.autocast(device_type="cuda", dtype=dtype)

    def _configure_pretraining(self, total_steps: int) -> None:
        """Configure the optimizer for initial foundation pretraining.

        Pretraining intentionally uses a separate recipe from continual learning:
        higher peak LR, stronger decoupled weight decay, a warmup/decay schedule,
        and gradient accumulation. No lesson acceptance gates are applied to the
        pretraining stream itself.
        """
        runtime = self.config.runtime
        # Foundation pretraining starts with fresh AdamW moments even if a checkpoint
        # was previously used for continual learning.
        self.optimizer = self._make_optimizer(self.config, self.model)
        self._training_mode = "pretrain"
        self._schedule_step_start = self.training_steps
        self._schedule_total_steps = max(1, int(total_steps))
        self._schedule_warmup_steps = max(0, int(runtime.pretrain_warmup_steps))
        if self._schedule_warmup_steps >= self._schedule_total_steps:
            self._schedule_warmup_steps = max(0, self._schedule_total_steps - 1)
        configured_decay = int(runtime.pretrain_decay_steps)
        remaining = max(1, self._schedule_total_steps - self._schedule_warmup_steps)
        self._schedule_decay_steps = max(1, configured_decay if configured_decay > 0 else remaining)
        self._schedule_min_lr_ratio = min(1.0, max(0.0, float(runtime.pretrain_min_lr_ratio)))
        self._schedule_peak_lrs = [
            float(runtime.pretrain_learning_rate),
            float(runtime.policy_learning_rate),
        ]
        for index, group in enumerate(self.optimizer.param_groups):
            group["lr"] = self._schedule_peak_lrs[min(index, len(self._schedule_peak_lrs) - 1)]
            group["weight_decay"] = float(runtime.pretrain_weight_decay)
            group["betas"] = (float(runtime.pretrain_beta1), float(runtime.pretrain_beta2))
        self._init_ema(float(runtime.pretrain_ema_decay))

    def _schedule_multiplier_for_next_step(self) -> float:
        step = max(1, self.training_steps - self._schedule_step_start + 1)
        if self._training_mode != "pretrain":
            return 1.0
        if self._schedule_warmup_steps > 0 and step <= self._schedule_warmup_steps:
            return max(1e-3, step / self._schedule_warmup_steps)
        decay_step = max(0, step - self._schedule_warmup_steps)
        progress = min(1.0, decay_step / max(1, self._schedule_decay_steps))
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return self._schedule_min_lr_ratio + (1.0 - self._schedule_min_lr_ratio) * cosine

    def _prepare_next_optimizer_step(self) -> None:
        if self._training_mode != "pretrain" or not self._schedule_peak_lrs:
            return
        multiplier = self._schedule_multiplier_for_next_step()
        for index, group in enumerate(self.optimizer.param_groups):
            peak = self._schedule_peak_lrs[min(index, len(self._schedule_peak_lrs) - 1)]
            group["lr"] = peak * multiplier

    def _init_ema(self, decay: float) -> None:
        self._ema_decay = min(0.999999, max(0.0, float(decay)))
        self._ema_shadow = {
            name: parameter.detach().clone()
            for name, parameter in self.model.named_parameters()
            if parameter.requires_grad
        }

    def _update_ema(self) -> None:
        if self._ema_shadow is None:
            return
        decay = float(getattr(self, "_ema_decay", 0.999))
        with torch.no_grad():
            for name, parameter in self.model.named_parameters():
                shadow = self._ema_shadow.get(name)
                if shadow is None:
                    continue
                shadow.mul_(decay).add_(parameter.detach(), alpha=1.0 - decay)

    @contextmanager
    def _ema_weights(self, enabled: bool = True):
        if not enabled or self._ema_shadow is None:
            yield
            return
        backup: dict[str, torch.Tensor] = {}
        try:
            with torch.no_grad():
                for name, parameter in self.model.named_parameters():
                    shadow = self._ema_shadow.get(name)
                    if shadow is None:
                        continue
                    backup[name] = parameter.detach().clone()
                    parameter.copy_(shadow)
            yield
        finally:
            with torch.no_grad():
                for name, parameter in self.model.named_parameters():
                    original = backup.get(name)
                    if original is not None:
                        parameter.copy_(original)

    def _apply_ema_to_model(self) -> None:
        if self._ema_shadow is None:
            return
        with torch.no_grad():
            for name, parameter in self.model.named_parameters():
                shadow = self._ema_shadow.get(name)
                if shadow is not None:
                    parameter.copy_(shadow)

    def _optimizer_step_from_accumulated(self, token_count: int) -> float:
        self._prepare_next_optimizer_step()
        if self.amp_enabled and self.amp_dtype == torch.float16:
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), float(self.config.runtime.gradient_clip))
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), float(self.config.runtime.gradient_clip))
            self.optimizer.step()
        self.optimizer.zero_grad(set_to_none=True)
        self.training_steps += 1
        self._update_ema()
        return float(token_count)

    def _backward_into_accumulator(self, loss: torch.Tensor, token_count: int, accumulation_steps: int) -> float:
        if not torch.isfinite(loss):
            return 0.0
        scale = max(1, int(accumulation_steps))
        scaled_loss = loss / scale
        if self.amp_enabled and self.amp_dtype == torch.float16:
            self.scaler.scale(scaled_loss).backward()
        else:
            scaled_loss.backward()
        self.processed_tokens += int(max(0, token_count))
        return float(loss.detach().float().cpu())

    def _optimizer_step(self, loss: torch.Tensor, token_count: int) -> float:
        if not torch.isfinite(loss):
            return 0.0
        self.optimizer.zero_grad(set_to_none=True)
        if self.amp_enabled and self.amp_dtype == torch.float16:
            self._prepare_next_optimizer_step()
            self.scaler.scale(loss).backward()
            has_grad = any(
                parameter.grad is not None
                for group in self.optimizer.param_groups
                for parameter in group["params"]
                if parameter.requires_grad
            )
            if not has_grad:
                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                has_grad = any(
                    parameter.grad is not None
                    for group in self.optimizer.param_groups
                    for parameter in group["params"]
                    if parameter.requires_grad
                )
                if not has_grad:
                    raise RuntimeError(
                        "training loss produced no optimizer gradients; "
                        "check the model forward/autograd path"
                    )
                self.amp_enabled = False
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), float(self.config.runtime.gradient_clip))
                self.optimizer.step()
            else:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), float(self.config.runtime.gradient_clip))
                self.scaler.step(self.optimizer)
                self.scaler.update()
        else:
            self._prepare_next_optimizer_step()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), float(self.config.runtime.gradient_clip))
            self.optimizer.step()
        self.training_steps += 1
        self.processed_tokens += int(max(0, token_count))
        self._update_ema()
        value = float(loss.detach().float().cpu())
        del loss
        return value

    def _register_unique_training_tokens(self, text: str, token_count: int | None = None) -> int:
        """Count newly supplied training tokens once, independent of epochs/replay.

        The lesson fingerprint is based on the exact supplied UTF-8 bytes so repeated
        copies of the same lesson cannot satisfy the diversity gate. The readiness count
        itself uses the active tokenizer, so learned BPE tokens are counted consistently
        with the current model context.
        """
        canonical = self._normalize_overlap_text(text)
        raw = canonical.encode("utf-8", errors="replace")
        digest = hashlib.sha256(raw).hexdigest()
        if digest in self._unique_training_fingerprints:
            return 0
        self._unique_training_fingerprints.add(digest)
        added = self.tokenizer.tokenize_count(canonical) if token_count is None else max(0, int(token_count))
        self.total_tokens += added
        return added

    def evaluate_text_loss(self, text: str, *, holdout: bool = False) -> float:
        chunks = self._encode_chunks(text, max_chars=self.config.runtime.language_eval_max_chars if holdout else 0)
        self.model.eval()
        with torch.inference_mode():
            losses = [self.model.language_loss(chunk) for chunk in chunks]
        finite = [loss for loss in losses if torch.isfinite(loss)]
        return float(torch.stack(finite).mean().detach().cpu()) if finite else 0.0

    def evaluate_text_bpb(self, text: str) -> float:
        """Score text by total negative log-likelihood divided by represented bytes."""
        chunks = self._encode_chunks(text, max_chars=self.config.runtime.language_eval_max_chars)
        self.model.eval()
        numerator = denominator = 0.0
        with torch.inference_mode():
            for chunk in chunks:
                nll_bits, byte_count = self.model.language_bpb_totals(chunk)
                numerator += nll_bits
                denominator += byte_count
        return numerator / denominator if denominator > 0.0 else 8.0

    def evaluate_byte_chunks_totals(self, raw: bytes) -> dict:
        """Return byte-weightable NLL totals for a fixed byte sequence."""
        chunks = self._encode_byte_chunks(raw, max_chunks=0, context=self.model.cfg.context)
        self.model.eval()
        numerator = denominator = 0.0
        with torch.inference_mode():
            for chunk in chunks:
                nll_bits, byte_count = self.model.language_bpb_totals(chunk)
                numerator += nll_bits
                denominator += byte_count
        return {"nll_bits": numerator, "byte_count": int(round(denominator))}

    def evaluate_byte_chunks_bpb(self, raw: bytes) -> float:
        totals = self.evaluate_byte_chunks_totals(raw)
        denominator = int(totals["byte_count"])
        return float(totals["nll_bits"]) / denominator if denominator > 0 else 8.0

    def _fixed_reference_source_metrics(self, reference_raw: bytes, eval_start: int) -> dict:
        """Summarize the fixed-heldout tail by source when a v2 manifest exists.

        The main release-gate BPB remains measured on the exact contiguous held-out
        byte tail. Supplemental source rows are reconstructed from only the bytes
        intersecting that same tail, avoiding a second, larger evaluation pass.
        """
        path = self._evaluation_path().with_suffix(".paragraphs.json")
        if not path.exists():
            return {"status": "unavailable", "reason": "source_manifest_missing"}
        try:
            manifest = json.loads(path.read_text(encoding="utf-8"))
            paragraphs = manifest.get("paragraphs", [])
        except (OSError, ValueError, TypeError):
            return {"status": "unavailable", "reason": "source_manifest_invalid"}
        if manifest.get("schema_version") not in {2, 3} or not isinstance(paragraphs, list):
            return {"status": "unavailable", "reason": "source_manifest_legacy"}

        fragments: dict[str, list[bytes]] = {}
        titles: dict[str, str] = {}
        cursor = 0
        for row in paragraphs:
            if not isinstance(row, dict) or not isinstance(row.get("text"), str):
                return {"status": "unavailable", "reason": "source_manifest_missing_text"}
            encoded = row["text"].encode("utf-8", errors="replace")
            start, end = cursor, cursor + len(encoded)
            left, right = max(start, int(eval_start)), min(end, len(reference_raw))
            source_id = str(row.get("source_id", "")).strip()
            if source_id and right > left:
                fragments.setdefault(source_id, []).append(reference_raw[left:right])
                titles[source_id] = str(row.get("domain", row.get("title", "unknown")))
            cursor = end + 2  # joined corpus uses exactly two LF bytes between paragraphs
        rows = []
        for source_id, pieces in fragments.items():
            source_bytes = b"\n\n".join(pieces)
            if len(source_bytes) < 8:
                continue
            totals = self.evaluate_byte_chunks_totals(source_bytes)
            if totals["byte_count"] > 0:
                rows.append({"source_id": source_id, "domain": titles[source_id], **totals})
        return summarize_bpb_rows(
            rows,
            bootstrap_samples=int(getattr(self.config.runtime, "evaluation_bootstrap_samples", 1000)),
            seed=int(getattr(self.config.runtime, "evaluation_seed", 1337)),
        )

    def evaluate_fixed_reference(self, *, use_ema: bool = False) -> dict:
        try:
            text = load_reference(self.root, self.config.runtime.language_eval_path, 0)
        except FileNotFoundError:
            self.last_eval_bpb = None
            self.eval_baseline_bpb = None
            self.eval_reference_bytes = 0
            return {"available": False, "reason": "evaluation corpus has not been built"}
        raw = text.encode("utf-8", errors="replace")
        reference_bytes = len(raw)
        if reference_bytes < 8:
            self.last_eval_bpb = None
            self.eval_baseline_bpb = None
            self.eval_reference_bytes = reference_bytes
            return {"available": False, "reason": "evaluation corpus is too small to score", "reference_bytes": reference_bytes}
        split = min(len(raw) - 1, max(1, int(len(raw) * 0.8)))
        eval_bytes = raw[split - 1:]
        with self._ema_weights(enabled=bool(use_ema)):
            bpb = self.evaluate_byte_chunks_bpb(eval_bytes)
            source_metrics = self._fixed_reference_source_metrics(raw, split - 1)
        fingerprint = corpus_fingerprint(text)
        if self._evaluation_baseline_cache is not None and self._evaluation_baseline_cache.get("fingerprint") == fingerprint:
            baseline = self._evaluation_baseline_cache["baseline"]
        else:
            baseline = best_byte_baseline_bpb(text, train_fraction=0.8)
            self._evaluation_baseline_cache = {"fingerprint": fingerprint, "baseline": baseline}
        self.last_eval_bpb = bpb
        self.eval_baseline_bpb = float(baseline["bpb"])
        self.eval_reference_bytes = reference_bytes
        excluded_training_paths = []
        manifest_path = self._evaluation_path().with_suffix(".sources.json")
        if manifest_path.exists():
            try:
                manifest_payload = json.loads(manifest_path.read_text(encoding="utf-8"))
                excluded_training_paths = list(manifest_payload.get("excluded_training_paths", []))
            except (OSError, ValueError, TypeError):
                excluded_training_paths = []
        config_payload = asdict(self.config)
        config_fingerprint = hashlib.sha256(
            json.dumps(config_payload, sort_keys=True, ensure_ascii=False, default=str).encode("utf-8")
        ).hexdigest()
        sources_manifest = self._evaluation_path().with_suffix(".sources.json")
        paragraphs_manifest = self._evaluation_path().with_suffix(".paragraphs.json")
        return {
            "available": True,
            "model_bpb": bpb,
            "baseline_bpb": float(baseline["bpb"]),
            "baseline_kind": baseline["kind"],
            "baseline_order": int(baseline.get("order", 1)),
            "baseline_alpha": float(baseline["alpha"]),
            "alternate_baseline": baseline["alternates"],
            "gain": float(baseline["bpb"]) - bpb,
            "reference_bytes": reference_bytes,
            "test_bytes": len(eval_bytes),
            "source_metrics": source_metrics,
            "fingerprint": corpus_fingerprint(text),
            "model_state_sha256": _sha256_model_state(self.model.state_dict()),
            "checkpoint_sha256": _sha256_file(self.paths.model),
            "tokenizer_sha256": str(self._tokenizer_fingerprint),
            "config_sha256": config_fingerprint,
            "evaluation_source_manifest_sha256": _sha256_file(sources_manifest),
            "evaluation_paragraph_manifest_sha256": _sha256_file(paragraphs_manifest),
            "evaluation_split": "fixed-heldout-contiguous-tail",
            "excluded_training_paths": excluded_training_paths,
        }

    def _build_evaluation_corpus(self) -> dict:
        path = self._evaluation_path()
        result = build_reference(
            path,
            target_bytes=int(self.config.runtime.evaluation_corpus_target_bytes),
            exclude_paths=list(self.config.runtime.evaluation_exclude_paths or []),
            overlap_words=int(self.config.runtime.evaluation_overlap_words),
            overlap_match_fraction=float(self.config.runtime.evaluation_overlap_match_fraction),
        )
        self.last_eval_bpb = None
        self.eval_baseline_bpb = None
        return result

    def _training_chunks(self, text: str, row_id: int = 0, *, max_chunks: int | None = None) -> list[torch.Tensor]:
        limit = self.config.runtime.max_training_chunks_per_memory if max_chunks is None else int(max_chunks)
        # The model context is authoritative. Holdout indices are computed over
        # these exact chunks and replay uses the stored indices, so a separate
        # configurable context cannot silently invalidate the split.
        return self._encode_chunks(text, max_chunks=max(0, limit), context=self.model.cfg.context)

    def _split_train_holdout_indices(self, count: int, fraction: float) -> tuple[list[int], list[int]]:
        """Use a contiguous suffix as the autoregressive holdout."""
        if count < 2 or fraction <= 0:
            return list(range(count)), []
        holdout_count = max(1, int(round(count * fraction)))
        holdout_count = min(holdout_count, count - 1)
        split = count - holdout_count
        return list(range(split)), list(range(split, count))

    def _split_train_holdout(self, chunks: list[torch.Tensor], fraction: float) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        train_idx, holdout_idx = self._split_train_holdout_indices(len(chunks), fraction)
        return [chunks[i] for i in train_idx], [chunks[i] for i in holdout_idx]

    def _chunks_for_memory_row(self, row: dict, *, all_chunks: bool = True) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        original_text = str(row.get("text", "")).strip()
        text = self._exclude_evaluation_text(original_text).strip()
        if not text:
            return [], []
        stored_context = int(row.get("train_context") or 0)
        context = int(self.model.cfg.context)
        # Once held-out content is removed, stored holdout indices no longer refer to
        # this text. Recompute the safe split rather than reusing stale indices.
        context_changed = stored_context != context or text != original_text
        chunks = self._encode_chunks(text, max_chunks=0 if all_chunks else self.config.runtime.max_training_chunks_per_memory, context=context)
        raw_holdout = row.get("holdout_indices")
        try:
            if raw_holdout and not context_changed:
                holdout_indices = {int(x) for x in json.loads(raw_holdout)}
            else:
                holdout_indices = set()
        except Exception:
            holdout_indices = set()
        if not holdout_indices and len(chunks) >= 2:
            _, fallback = self._split_train_holdout_indices(len(chunks), float(self.config.runtime.teacher_holdout_fraction))
            holdout_indices = set(fallback)
        train = [chunk for idx, chunk in enumerate(chunks) if idx not in holdout_indices]
        holdout = [chunk for idx, chunk in enumerate(chunks) if idx in holdout_indices]
        return train, holdout

    def _trajectory_from_rows(self, rows: list[dict]) -> tuple[list[torch.Tensor], list[int | None], list[float]]:
        tokens: list[torch.Tensor] = []
        previous_actions: list[int | None] = []
        errors: list[float] = []
        name_to_index = {name: i for i, name in enumerate(self.registry.names())}
        for row in rows:
            train_chunks, _ = self._chunks_for_memory_row(row)
            limit = int(self.config.runtime.max_training_chunks_per_memory)
            if limit > 0 and len(train_chunks) > limit:
                if limit == 1:
                    train_chunks = [train_chunks[0]]
                else:
                    indices = [min(len(train_chunks)-1, round(i*(len(train_chunks)-1)/(limit-1))) for i in range(limit)]
                    train_chunks = [train_chunks[i] for i in dict.fromkeys(indices)]
            action = name_to_index.get(row.get("action_name")) if row.get("action_name") else None
            error = float(row.get("prediction_error", 0.0))
            for chunk_index, chunk in enumerate(train_chunks):
                tokens.append(chunk)
                previous_actions.append(action if chunk_index == 0 else None)
                errors.append(error if chunk_index == 0 else 0.0)
        return tokens, previous_actions, errors

    def _chunk_bpb(self, chunks: list[torch.Tensor]) -> float:
        self.model.eval()
        numerator = denominator = 0.0
        with torch.inference_mode():
            for chunk in chunks:
                nll_bits, byte_count = self.model.language_bpb_totals(chunk)
                numerator += nll_bits
                denominator += byte_count
        return numerator / denominator if denominator > 0.0 else 8.0

    @staticmethod
    def _sequence_token_count(sequence: tuple[list[torch.Tensor], list[int | None], list[float]]) -> int:
        total = 0
        for chunk in sequence[0]:
            values = chunk.detach().reshape(-1)
            total += int(((values != PAD) & (values != 257) & (values != 258)).sum().item())
        return total

    def _batch_trajectory_sequences(
        self,
        sequences: list[tuple[list[torch.Tensor], list[int | None], list[float]]],
    ) -> float:
        last = 0.0
        batch_size = max(1, int(self.config.runtime.train_batch_size))
        accumulation_steps = (
            max(1, int(self.config.runtime.pretrain_gradient_accumulation_steps))
            if self._training_mode == "pretrain"
            else 1
        )

        self.optimizer.zero_grad(set_to_none=True)
        micro_batches = 0
        accumulated_tokens = 0
        for start in range(0, len(sequences), batch_size):
            batch = sequences[start:start + batch_size]
            token_count = sum(self._sequence_token_count(sequence) for sequence in batch)
            try:
                with self._autocast():
                    loss, _ = self.model.trajectory_loss_batched(batch)
                if not torch.isfinite(loss):
                    del loss
                    continue

                if accumulation_steps == 1:
                    value = self._optimizer_step(loss, token_count)
                    if value:
                        last = value
                    micro_batches = 0
                    accumulated_tokens = 0
                    continue

                value = self._backward_into_accumulator(loss, token_count, accumulation_steps)
                if value:
                    last = value
                micro_batches += 1
                accumulated_tokens += token_count

                is_last = start + batch_size >= len(sequences)
                if micro_batches >= accumulation_steps or is_last:
                    self._optimizer_step_from_accumulated(accumulated_tokens)
                    micro_batches = 0
                    accumulated_tokens = 0
            except torch.cuda.OutOfMemoryError:
                if self.device.type != "cuda":
                    raise
                self.optimizer.zero_grad(set_to_none=True)
                torch.cuda.empty_cache()
                micro_batches = 0
                accumulated_tokens = 0
                # A batch OOM should not discard an entire shard. Retry its individual
                # trajectories with the normal one-step path.
                for sequence in batch:
                    with self._autocast():
                        retry_loss, _ = self.model.trajectory_loss_batched([sequence])
                    if not torch.isfinite(retry_loss):
                        del retry_loss
                        continue
                    value = self._optimizer_step(retry_loss, self._sequence_token_count(sequence))
                    if value:
                        last = value

        if self.device.type == "cuda":
            torch.cuda.empty_cache()
        gc.collect()
        return last

    def _as_sequences(
        self,
        chunks: list[torch.Tensor],
        sequence_length: int,
    ) -> list[tuple[list[torch.Tensor], list[int | None], list[float]]]:
        seq_len = max(1, int(sequence_length))
        return [([*chunks[i:i + seq_len]], [None] * len(chunks[i:i + seq_len]), [0.0] * len(chunks[i:i + seq_len]))
                for i in range(0, len(chunks), seq_len)]

    def _build_replay_sequences(
        self,
        target_count: int,
        *,
        exclude_ids: set[int] | None = None,
    ) -> tuple[list[tuple[list[torch.Tensor], list[int | None], list[float]]], set[int]]:
        """Build balanced replay trajectories from previously learned lessons.

        FINAL8.4 could request many replay sequences but effectively cap the result at
        one sequence per lesson. FINAL8.5 samples multiple contiguous trajectories per
        lesson, then allocates them round-robin across the lesson pool so older lessons
        are represented before any one lesson dominates the replay stream.
        """
        target_count = max(0, int(target_count))
        if target_count <= 0:
            return [], set()

        excluded = {int(item) for item in (exclude_ids or set())}
        pool_setting = int(self.config.runtime.replay_lesson_pool)
        if pool_setting > 0:
            pool_limit = pool_setting
        else:
            pool_limit = max(1, self.memory.lesson_count())

        if pool_setting > 0:
            rows = self.memory.sample_validated_lessons(pool_limit)
        else:
            rows = self.memory.sample_validated_lessons(max(1, self.memory.lesson_count()))
        deduped: list[dict] = []
        seen_fingerprints: set[str] = set()
        for row in rows:
            if int(row["id"]) in excluded:
                continue
            fingerprint = str(row.get("content_fingerprint") or row.get("id"))
            if fingerprint in seen_fingerprints:
                continue
            seen_fingerprints.add(fingerprint)
            deduped.append(row)
        rows = deduped
        if not rows:
            return [], set()

        random.shuffle(rows)
        seq_len = max(1, int(self.config.runtime.teacher_sequence_length))
        per_lesson_limit = max(1, int(self.config.runtime.replay_max_sequences_per_lesson))

        candidates: list[
            tuple[int, list[tuple[list[torch.Tensor], list[int | None], list[float]]]]
        ] = []

        for row in rows:
            row_id = int(row["id"])
            chunks, _ = self._chunks_for_memory_row(row)
            if not chunks:
                continue

            if len(chunks) <= seq_len:
                choices = self._as_sequences(chunks, seq_len)
            else:
                starts = list(range(0, len(chunks) - seq_len + 1))
                random.shuffle(starts)
                choices = [
                    self._as_sequences(chunks[start:start + seq_len], seq_len)[0]
                    for start in starts[:per_lesson_limit]
                ]

            if choices:
                random.shuffle(choices)
                candidates.append((row_id, choices[:per_lesson_limit]))

        output: list[tuple[list[torch.Tensor], list[int | None], list[float]]] = []
        source_ids: set[int] = set()

        # Round-robin gives every eligible lesson an opportunity before a second
        # trajectory from any single lesson is selected.
        depth = 0
        while len(output) < target_count and candidates:
            progressed = False
            for row_id, choices in candidates:
                if depth < len(choices) and len(output) < target_count:
                    output.append(choices[depth])
                    source_ids.add(row_id)
                    progressed = True
            if not progressed:
                break
            depth += 1

        return output, source_ids

    def _train_sequence_collection(
        self,
        sequences: list[tuple[list[torch.Tensor], list[int | None], list[float]]],
        *,
        epochs: int,
        replay: bool = True,
        exclude_replay_ids: set[int] | None = None,
        minimum_replay_sequences: int = 0,
    ) -> float:
        if not sequences:
            self.last_replay_stats = {
                "requested": 0,
                "used": 0,
                "lesson_count": 0,
                "lesson_ids": [],
                "per_epoch": [],
            }
            return 0.0

        last = 0.0
        replay_per_epoch: list[int] = []
        replay_lesson_ids: set[int] = set()
        self.model.train()

        for _ in range(max(1, int(epochs))):
            order = list(sequences)
            replay_count = 0

            if replay and self.config.runtime.replay_mix_fraction > 0:
                fractional_count = int(
                    round(len(sequences) * float(self.config.runtime.replay_mix_fraction))
                )
                replay_count = max(
                    0,
                    fractional_count,
                    max(0, int(minimum_replay_sequences)),
                )
                replay_sequences, replay_ids = self._build_replay_sequences(
                    replay_count,
                    exclude_ids=exclude_replay_ids,
                )
                order.extend(replay_sequences)
                replay_count = len(replay_sequences)
                replay_lesson_ids.update(replay_ids)

            replay_per_epoch.append(replay_count)
            random.shuffle(order)

            for _ in range(max(1, int(self.config.runtime.teacher_steps_per_sequence))):
                value = self._batch_trajectory_sequences(order)
                if value:
                    last = value

        if self.device.type == "cuda":
            torch.cuda.empty_cache()
        gc.collect()
        self.last_language_loss = last
        self.last_replay_stats = {
            "requested": sum(replay_per_epoch),
            "used": sum(replay_per_epoch),
            "lesson_count": len(replay_lesson_ids),
            "lesson_ids": sorted(replay_lesson_ids),
            "per_epoch": replay_per_epoch,
        }
        return last

    def _train_chunk_sequences(self, chunks: list[torch.Tensor], sequence_length: int, steps: int) -> float:
        if not chunks:
            return 0.0
        sequences=self._as_sequences(chunks, sequence_length)
        old_steps=self.config.runtime.teacher_steps_per_sequence
        self.config.runtime.teacher_steps_per_sequence=max(1, int(steps))
        try:
            return self._train_sequence_collection(sequences, epochs=1, replay=False)
        finally:
            self.config.runtime.teacher_steps_per_sequence=old_steps

    def _train_recent_trajectory(self, steps: int) -> float:
        sequences=[]
        latest=self.memory.sample_sequence(self.config.runtime.sequence_length)
        if latest:
            batches, actions, errors=self._trajectory_from_rows(latest)
            if batches:
                sequences.append((batches, actions, errors))
        for _ in range(max(0, int(self.config.runtime.replay_sequences))):
            sample=self.memory.sample_sequence(self.config.runtime.sequence_length)
            if not sample:
                continue
            batches, actions, errors=self._trajectory_from_rows(sample)
            if batches:
                sequences.append((batches, actions, errors))
        if not sequences:
            return 0.0
        old=self.config.runtime.teacher_steps_per_sequence
        self.config.runtime.teacher_steps_per_sequence=max(1, int(steps))
        try:
            last=self._train_sequence_collection(sequences, epochs=1, replay=False)
        finally:
            self.config.runtime.teacher_steps_per_sequence=old
        return last

    def _train_text_sequence(self, text: str, steps: int) -> float:
        chunks=self._training_chunks(text, max_chunks=0)
        if not chunks:
            return 0.0
        old=self.config.runtime.teacher_steps_per_sequence
        self.config.runtime.teacher_steps_per_sequence=max(1, int(steps))
        try:
            last=self._train_sequence_collection(self._as_sequences(chunks, self.config.runtime.teacher_sequence_length), epochs=1, replay=False)
        finally:
            self.config.runtime.teacher_steps_per_sequence=old
        return last

    @staticmethod
    def reset_storage(root: Path) -> Path | None:
        """Archive the complete learned state, then remove active state files."""
        root = root.resolve()
        data = root / "data"
        if not data.exists():
            data.mkdir(parents=True, exist_ok=True)
            return None
        stamp = time.strftime("%Y%m%d_%H%M%S_") + f"{time.time_ns() % 1_000_000_000:09d}"
        archive_root = data / "archive" / stamp
        archive_root.mkdir(parents=True, exist_ok=False)
        moved = False
        # Archive all active runtime state, including future learned-state files that
        # are not yet known to this version. Keep older archives outside the move set.
        for path in list(data.iterdir()):
            if path.name == "archive" or path == archive_root:
                continue
            shutil.move(str(path), str(archive_root / path.name))
            moved = True
        return archive_root if moved else None

    def _teacher_chunks(self, text: str) -> list[torch.Tensor]:
        chunks = self._encode_chunks(text)
        limit = max(1, int(self.config.runtime.teacher_max_chunks))
        if len(chunks) <= limit:
            return chunks
        indices = [min(len(chunks) - 1, round(i * (len(chunks) - 1) / (limit - 1))) for i in range(limit)]
        return [chunks[i] for i in dict.fromkeys(indices)]

    def _evaluation_overlap_index(self) -> dict:
        """Cache exact/normalized hashes and word shingles from fixed evaluation text."""
        import hashlib

        evaluation_path = self._evaluation_path()
        try:
            text = evaluation_path.read_text(encoding="utf-8") if evaluation_path.exists() else ""
        except (OSError, UnicodeError):
            text = ""
        fingerprint = hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()
        if self._evaluation_overlap_cache is not None and self._evaluation_overlap_cache.get("fingerprint") == fingerprint:
            return self._evaluation_overlap_cache

        hash_manifest = evaluation_path.with_suffix(".paragraphs.json")
        hashes: set[str] = set()
        if hash_manifest.exists():
            try:
                payload = json.loads(hash_manifest.read_text(encoding="utf-8"))
                for item in payload.get("paragraphs", []):
                    if isinstance(item, dict):
                        hashes.update(str(item[key]) for key in ("sha256", "normalized_sha256") if item.get(key))
            except (OSError, ValueError, TypeError):
                # The raw evaluation text still provides usable hashes below.
                pass
        paragraphs = [self._normalize_overlap_text(p) for p in self._paragraphs(text)]
        for paragraph in paragraphs:
            if paragraph:
                hashes.add(hashlib.sha256(paragraph.encode("utf-8", errors="replace")).hexdigest())

        words_per_shingle = max(8, int(getattr(self.config.runtime, "evaluation_overlap_words", 16)))
        stride = max(1, words_per_shingle // 2)
        shingles: set[int] = set()
        for paragraph in paragraphs:
            words = re.findall(r"[a-z0-9]+", paragraph.lower())
            for index in range(0, max(0, len(words) - words_per_shingle + 1), stride):
                piece = " ".join(words[index:index + words_per_shingle]).encode("utf-8")
                shingles.add(int(hashlib.blake2b(piece, digest_size=8).hexdigest(), 16))
        self._evaluation_overlap_cache = {
            "fingerprint": fingerprint,
            "hashes": hashes,
            "paragraphs": paragraphs,
            "shingles": shingles,
            "words_per_shingle": words_per_shingle,
            "stride": stride,
        }
        return self._evaluation_overlap_cache

    def _evaluation_paragraph_hashes(self) -> set[str]:
        return set(self._evaluation_overlap_index()["hashes"])

    @staticmethod
    def _normalize_overlap_text(text: str) -> str:
        return re.sub(r"\s+", " ", text).strip()

    @staticmethod
    def _paragraphs(text: str) -> list[str]:
        return [part.strip() for part in re.split(r"\n\s*\n", text) if part.strip()]

    def _evaluation_paragraph_texts(self) -> list[str]:
        return list(self._evaluation_overlap_index()["paragraphs"])

    def _exclude_evaluation_text(self, text: str) -> str:
        index = self._evaluation_overlap_index()
        hashes: set[str] = index["hashes"]
        evaluation_paragraphs: list[str] = index["paragraphs"]
        evaluation_shingles: set[int] = index["shingles"]
        words_per_shingle = int(index["words_per_shingle"])
        stride = int(index["stride"])
        if not hashes and not evaluation_paragraphs:
            return text

        import hashlib
        match_fraction = float(getattr(self.config.runtime, "evaluation_overlap_match_fraction", 0.20))
        kept: list[str] = []
        for paragraph in self._paragraphs(text):
            normalized = self._normalize_overlap_text(paragraph)
            exact_digest = hashlib.sha256(paragraph.encode("utf-8", errors="replace")).hexdigest()
            normalized_digest = hashlib.sha256(normalized.encode("utf-8", errors="replace")).hexdigest()
            excluded = exact_digest in hashes or normalized_digest in hashes
            if not excluded and normalized and evaluation_paragraphs:
                for evaluation_paragraph in evaluation_paragraphs:
                    if len(evaluation_paragraph) < 80 or len(normalized) < 80:
                        continue
                    shorter = min(len(normalized), len(evaluation_paragraph))
                    longer = max(len(normalized), len(evaluation_paragraph))
                    if shorter / longer >= 0.50 and (
                        evaluation_paragraph in normalized or normalized in evaluation_paragraph
                    ):
                        excluded = True
                        break

            # Catch near duplicates that have been rewrapped, lightly edited, or joined
            # into different paragraph boundaries. This is a corpus-isolation heuristic,
            # not a proof of semantic originality; every source remains auditable.
            if not excluded and evaluation_shingles and normalized:
                words = re.findall(r"[a-z0-9]+", normalized.lower())
                candidates = []
                for offset in range(0, max(0, len(words) - words_per_shingle + 1), stride):
                    piece = " ".join(words[offset:offset + words_per_shingle]).encode("utf-8")
                    candidates.append(int(hashlib.blake2b(piece, digest_size=8).hexdigest(), 16))
                if candidates:
                    matches = sum(candidate in evaluation_shingles for candidate in candidates)
                    ratio = matches / len(candidates)
                    if matches >= 2 and ratio >= match_fraction:
                        excluded = True

            if not excluded:
                kept.append(paragraph)
        return "\n\n".join(kept).strip()

    def _train_dialogue_alignment(self, pairs: list[tuple[str, str]]) -> dict:
        """Train a small self-supervised context/response alignment objective.

        Positive pairs come only from observed conversation order. Other responses in
        the same mini-batch act as negatives. No intent labels, emotion labels, word
        definitions, or pretrained embeddings are introduced.
        """
        weight = max(0.0, float(self.config.runtime.dialogue_alignment_weight))
        if weight <= 0.0 or len(pairs) < 2:
            return {"loss": 0.0, "steps": 0, "pairs": len(pairs)}

        batch_size = max(2, int(self.config.runtime.dialogue_alignment_batch_size))
        order = list(range(len(pairs)))
        random.shuffle(order)
        old_training = self.model.training
        self.model.train()
        last = 0.0
        steps = 0
        used_pairs = 0
        try:
            for start in range(0, len(order), batch_size):
                selected = order[start:start + batch_size]
                if len(selected) < 2:
                    continue
                context_tokens: list[torch.Tensor] = []
                response_tokens: list[torch.Tensor] = []
                token_count = 0
                for idx in selected:
                    context, response = pairs[idx]
                    c = self._encode_text(context)
                    r = self._encode_text(response)
                    context_tokens.append(c)
                    response_tokens.append(r)
                    token_count += max(0, int(c.size(1) - 2)) + max(0, int(r.size(1) - 2))
                try:
                    with self._autocast():
                        loss = self.model.response_alignment_loss(
                            context_tokens,
                            response_tokens,
                            temperature=float(self.config.runtime.dialogue_alignment_temperature),
                        ) * weight
                    if not torch.isfinite(loss):
                        del loss
                        continue
                    value = self._optimizer_step(loss, token_count)
                    if value:
                        last = value
                    steps += 1
                    used_pairs += len(selected)
                except torch.cuda.OutOfMemoryError:
                    if self.device.type != "cuda" or len(selected) <= 2:
                        raise
                    self.optimizer.zero_grad(set_to_none=True)
                    torch.cuda.empty_cache()
                    # Retry with a minimal pair rather than dropping the entire
                    # communication-alignment phase on a 6 GB-class GPU.
                    pair = selected[:2]
                    context_tokens = [self._encode_text(pairs[idx][0]) for idx in pair]
                    response_tokens = [self._encode_text(pairs[idx][1]) for idx in pair]
                    token_count = sum(
                        max(0, int(item.size(1) - 2))
                        for item in context_tokens + response_tokens
                    )
                    with self._autocast():
                        loss = self.model.response_alignment_loss(
                            context_tokens, response_tokens,
                            temperature=float(self.config.runtime.dialogue_alignment_temperature),
                        ) * weight
                    if torch.isfinite(loss):
                        value = self._optimizer_step(loss, token_count)
                        if value:
                            last = value
                        steps += 1
                        used_pairs += 2
        finally:
            if not old_training:
                self.model.eval()
            if self.device.type == "cuda":
                torch.cuda.empty_cache()
            gc.collect()
        return {"loss": last, "steps": steps, "pairs": used_pairs}

    def _pending_unique_training_tokens(self, text: str, token_count: int | None = None) -> tuple[str, int]:
        canonical = self._normalize_overlap_text(text)
        raw = canonical.encode("utf-8", errors="replace")
        digest = hashlib.sha256(raw).hexdigest()
        if digest in self._unique_training_fingerprints:
            return digest, 0
        added = self.tokenizer.tokenize_count(canonical) if token_count is None else max(0, int(token_count))
        return digest, added

    def _commit_unique_training_tokens(self, digest: str, added: int) -> int:
        if digest in self._unique_training_fingerprints:
            return 0
        self._unique_training_fingerprints.add(str(digest))
        self.total_tokens += max(0, int(added))
        return max(0, int(added))

    def _validated_transfer_probe(self, exclude_ids: set[int] | None = None) -> list[torch.Tensor]:
        excluded = {int(x) for x in (exclude_ids or set())}
        rows = [
            row for row in self.memory.sample_validated_lessons(
                max(1, int(self.config.runtime.transfer_probe_candidate_pool))
            )
            if int(row["id"]) not in excluded
        ]
        deduped: list[dict] = []
        seen: set[str] = set()
        for row in rows:
            key = str(row.get("content_fingerprint") or row["id"])
            if key in seen:
                continue
            seen.add(key)
            deduped.append(row)
        rows = deduped
        probe: list[torch.Tensor] = []
        per_document = max(1, int(self.config.runtime.lesson_transfer_chunks_per_document))
        target = max(1, int(self.config.runtime.lesson_transfer_documents)) * per_document
        for row in rows:
            _, holdout = self._chunks_for_memory_row(row)
            if holdout:
                probe.extend(holdout[:per_document])
            if len(probe) >= target:
                break
        return probe[:target]

    def _alignment_eval(
        self,
        pairs: list[tuple[str, str]],
        *,
        max_pairs: int = 128,
        seed: int | None = None,
    ) -> dict:
        if len(pairs) < 2:
            return {
                "pairs": len(pairs), "top1": 0.0, "mrr": 0.0,
                "random_top1": 1.0 / max(1, len(pairs)),
                "random_mrr": 1.0 / max(1, len(pairs)),
            }
        unique: dict[str, tuple[str, str]] = {}
        for context, response in pairs:
            key = self._normalize_overlap_text(response)
            if key and key not in unique:
                unique[key] = (context, response)
        pool = list(unique.values())
        if len(pool) < 2:
            return {
                "pairs": len(pool), "top1": 0.0, "mrr": 0.0,
                "random_top1": 1.0 / max(1, len(pool)),
                "random_mrr": 1.0 / max(1, len(pool)),
            }
        rng = random.Random(seed) if seed is not None else random.Random()
        sample = rng.sample(pool, min(len(pool), max(2, int(max_pairs))))
        contexts = [self._encode_text(context) for context, _ in sample]
        responses = [self._encode_text(response) for _, response in sample]
        was_training = self.model.training
        try:
            self.model.eval()
            with torch.inference_mode():
                context_vec = torch.cat([self.model.pooled_representation(item) for item in contexts], dim=0)
                response_vec = torch.cat([self.model.pooled_representation(item) for item in responses], dim=0)
                scores = torch.matmul(context_vec, response_vec.transpose(0, 1))
                ranks: list[int] = []
                for i in range(scores.size(0)):
                    order = torch.argsort(scores[i], descending=True)
                    match = (order == i).nonzero(as_tuple=False)
                    ranks.append(int(match[0].item()) + 1 if match.numel() else scores.size(0))
        finally:
            if was_training:
                self.model.train()
        n = len(ranks)
        return {
            "pairs": n,
            "top1": sum(1 for rank in ranks if rank == 1) / max(1, n),
            "mrr": sum(1.0 / rank for rank in ranks) / max(1, n),
            "random_top1": 1.0 / max(1, n),
            "random_mrr": sum(1.0 / rank for rank in range(1, n + 1)) / max(1, n),
        }

    def teach_text(self, lesson: Lesson, *, epochs: int | None = None, steps_per_sequence: int | None = None, dialogue: bool = False, preserve_literal: bool = False) -> dict:
        raw_text = lesson.text if preserve_literal else lesson.text.strip()
        language_text, cleaning_meta = prepare_language_text(
            raw_text,
            strip_transcript_annotations=(False if preserve_literal else bool(self.config.runtime.teacher_strip_transcript_annotations)),
        )
        text = self._exclude_evaluation_text(language_text if preserve_literal else language_text.strip())
        max_chars = int(self.config.runtime.teacher_max_source_chars)
        if max_chars > 0:
            text = text[:max_chars]
        if len(text) < 8:
            raise ValueError("lesson text is too short to teach")
        all_chunks = self._encode_chunks(
            text,
            max_chunks=max(0, int(self.config.runtime.teacher_max_chunks)),
            context=self.model.cfg.context,
        )
        train_idx, holdout_idx = self._split_train_holdout_indices(
            len(all_chunks), float(self.config.runtime.teacher_holdout_fraction)
        )
        train_chunks = [all_chunks[i] for i in train_idx]
        holdout_chunks = [all_chunks[i] for i in holdout_idx]
        if not train_chunks:
            raise ValueError("lesson produced no training chunks")

        train_token_count = sum(max(0, int(chunk.size(1) - 2)) for chunk in train_chunks)
        unique_digest, unique_pending = self._pending_unique_training_tokens(text, train_token_count)

        before_holdout = self._chunk_bpb(holdout_chunks) if holdout_chunks else self._chunk_bpb(train_chunks)
        fixed_before = self.evaluate_fixed_reference()
        transfer_probe_before = self._validated_transfer_probe()
        transfer_before = self._chunk_bpb(transfer_probe_before) if transfer_probe_before else None

        lesson_model_snapshot = None
        lesson_optimizer_snapshot = None
        if bool(self.config.runtime.lesson_rollback_on_fixed_regression):
            lesson_model_snapshot = {name: value.detach().clone() for name, value in self.model.state_dict().items()}
            lesson_optimizer_snapshot = copy.deepcopy(self.optimizer.state_dict())
        previous_language_loss = self.last_language_loss

        row_id = self.memory.add(
            kind=lesson.kind,
            source=lesson.source,
            title=lesson.title,
            url=lesson.source if lesson.source.startswith(("http://", "https://")) else "",
            text=text,
            loss_before=before_holdout,
            loss_after=before_holdout,
            learning_progress=0.0,
            action_name="teacher_pending",
            prediction_error=before_holdout,
            train_context=self.model.cfg.context,
            holdout_indices=json.dumps(holdout_idx),
            accepted=False,
            replay_eligible=False,
            merge_existing=False,
        )
        self.memory.replace_chunks(row_id, self._memory_chunk_texts(text, context=self.model.cfg.context))

        seq_len = max(1, int(self.config.runtime.teacher_sequence_length))
        teach_epochs = max(1, int(epochs if epochs is not None else self.config.runtime.teacher_epochs))
        requested_steps = steps_per_sequence if steps_per_sequence is not None else self.config.runtime.teacher_steps_per_sequence
        current_sequences = self._as_sequences(train_chunks, seq_len)

        old_steps = self.config.runtime.teacher_steps_per_sequence
        self.config.runtime.teacher_steps_per_sequence = max(1, int(requested_steps))
        try:
            last = self._train_sequence_collection(
                current_sequences,
                epochs=teach_epochs,
                replay=True,
                exclude_replay_ids={row_id},
                minimum_replay_sequences=int(self.config.runtime.replay_min_sequences),
            )
            general_replay_stats = dict(self.last_replay_stats)
        finally:
            self.config.runtime.teacher_steps_per_sequence = old_steps

        dialogue_pairs: list[tuple[str, str]] = []
        dialogue_examples: list[str] = []
        dialogue_alignment_stats = {"loss": 0.0, "steps": 0, "pairs": 0}
        alignment_eval_before = {"pairs": 0, "top1": 0.0, "mrr": 0.0}
        alignment_eval_after = {"pairs": 0, "top1": 0.0, "mrr": 0.0, "random_top1": 0.0, "random_mrr": 0.0}
        alignment_eval_candidate = dict(alignment_eval_after)
        dialogue_fixed_gain_before = None
        dialogue_fixed_gain_after = None
        dialogue_fixed_gain_delta = 0.0
        dialogue_accepted = None
        dialogue_train_examples = 0

        if dialogue:
            dialogue_cap = max(1, int(self.config.runtime.dialogue_max_examples))
            parse_cap = dialogue_cap * max(1, int(self.config.runtime.dialogue_passes))
            dialogue_pairs, dialogue_eval_pairs = split_dialogue_response_pairs(
                raw_text,
                holdout_fraction=float(self.config.runtime.dialogue_holdout_fraction),
                max_train_examples=parse_cap,
                max_eval_examples=int(self.config.runtime.dialogue_alignment_eval_examples),
                seed=int(self.config.runtime.dialogue_alignment_eval_seed),
            )
            dialogue_examples = [format_dialogue_example(prompt, response) for prompt, response in dialogue_pairs]
            dialogue_accepted = True if dialogue_examples else None
            self.dialogue_examples_seen += len(dialogue_examples)

            if dialogue_examples:
                model_snapshot = {name: value.detach().clone() for name, value in self.model.state_dict().items()}
                optimizer_snapshot = copy.deepcopy(self.optimizer.state_dict())
                fixed_dialogue_before = self.evaluate_fixed_reference()
                dialogue_fixed_gain_before = fixed_dialogue_before.get("gain")
                # Measure alignment on the untouched dialogue holdout immediately before
                # any dialogue-specific training. This is the true pre-objective metric.
                alignment_eval_before = self._alignment_eval(
                    dialogue_eval_pairs,
                    max_pairs=int(self.config.runtime.dialogue_alignment_eval_examples),
                    seed=int(self.config.runtime.dialogue_alignment_eval_seed),
                )

                shuffled_examples = list(dialogue_examples)
                random.shuffle(shuffled_examples)
                max_passes = max(1, int(self.config.runtime.dialogue_passes))
                selected_examples = shuffled_examples[: dialogue_cap * max_passes]
                dialogue_train_examples = len(selected_examples)

                old_steps = self.config.runtime.teacher_steps_per_sequence
                old_replay_fraction = self.config.runtime.replay_mix_fraction
                old_lrs = [float(group["lr"]) for group in self.optimizer.param_groups]
                self.config.runtime.teacher_steps_per_sequence = 1
                try:
                    for group in self.optimizer.param_groups:
                        group["lr"] = float(group["lr"]) * float(self.config.runtime.dialogue_learning_rate_scale)
                    for pass_start in range(0, len(selected_examples), dialogue_cap):
                        batch_examples = selected_examples[pass_start:pass_start + dialogue_cap]
                        dialogue_sequences = []
                        for example in batch_examples:
                            example_chunks = self._encode_chunks(example, max_chunks=0, context=self.model.cfg.context)
                            if example_chunks:
                                dialogue_sequences.append((example_chunks, [None] * len(example_chunks), [0.0] * len(example_chunks)))
                        if not dialogue_sequences:
                            continue
                        replay_count = max(
                            int(self.config.runtime.conversation_replay_examples),
                            int(round(len(dialogue_sequences) * float(self.config.runtime.dialogue_replay_multiplier))),
                        )
                        self.config.runtime.replay_mix_fraction = replay_count / max(1, len(dialogue_sequences))
                        value = self._train_sequence_collection(
                            dialogue_sequences,
                            epochs=max(1, int(self.config.runtime.dialogue_train_epochs)),
                            replay=True,
                            minimum_replay_sequences=0,
                        )
                        if value:
                            last = value
                finally:
                    self.config.runtime.teacher_steps_per_sequence = old_steps
                    self.config.runtime.replay_mix_fraction = old_replay_fraction
                    for group, lr in zip(self.optimizer.param_groups, old_lrs):
                        group["lr"] = lr

                alignment_pairs: list[tuple[str, str]] = []
                seen_responses: set[str] = set()
                for pair in dialogue_pairs:
                    key = self._normalize_overlap_text(pair[1])
                    if not key or key in seen_responses:
                        continue
                    seen_responses.add(key)
                    alignment_pairs.append(pair)

                old_lrs = [float(group["lr"]) for group in self.optimizer.param_groups]
                try:
                    for group in self.optimizer.param_groups:
                        group["lr"] = float(group["lr"]) * float(self.config.runtime.dialogue_learning_rate_scale)
                    dialogue_alignment_stats = self._train_dialogue_alignment(alignment_pairs)
                    if dialogue_alignment_stats.get("loss"):
                        last = dialogue_alignment_stats["loss"]
                finally:
                    for group, lr in zip(self.optimizer.param_groups, old_lrs):
                        group["lr"] = lr

                alignment_eval_after = self._alignment_eval(
                    dialogue_eval_pairs,
                    max_pairs=int(self.config.runtime.dialogue_alignment_eval_examples),
                    seed=int(self.config.runtime.dialogue_alignment_eval_seed),
                )
                alignment_eval_candidate = dict(alignment_eval_after)
                fixed_dialogue_after = self.evaluate_fixed_reference()
                dialogue_fixed_gain_after = fixed_dialogue_after.get("gain")
                if dialogue_fixed_gain_before is not None and dialogue_fixed_gain_after is not None:
                    dialogue_fixed_gain_delta = float(dialogue_fixed_gain_after) - float(dialogue_fixed_gain_before)
                if dialogue_fixed_gain_delta < -float(self.config.runtime.dialogue_max_fixed_regression):
                    self.model.load_state_dict(model_snapshot)
                    self.optimizer.load_state_dict(optimizer_snapshot)
                    dialogue_accepted = False
                    last = previous_language_loss if previous_language_loss is not None else last

        self.last_dialogue_accepted = dialogue_accepted

        fixed_after = self.evaluate_fixed_reference()
        lesson_fixed_gain_before = fixed_before.get("gain")
        lesson_fixed_gain_after = fixed_after.get("gain")
        lesson_fixed_gain_delta = None
        if lesson_fixed_gain_before is not None and lesson_fixed_gain_after is not None:
            lesson_fixed_gain_delta = float(lesson_fixed_gain_after) - float(lesson_fixed_gain_before)

        after_holdout = self._chunk_bpb(holdout_chunks) if holdout_chunks else self._chunk_bpb(train_chunks)
        local_improvement = before_holdout - after_holdout
        transfer_probe_after = self._validated_transfer_probe(exclude_ids={row_id})
        transfer_after = self._chunk_bpb(transfer_probe_after) if transfer_probe_after else None
        transfer_improvement = (transfer_before - transfer_after) if transfer_before is not None and transfer_after is not None else None

        tracks = [
            (float(self.config.runtime.lesson_fixed_weight), lesson_fixed_gain_delta),
            (float(self.config.runtime.lesson_local_weight), local_improvement),
            (float(self.config.runtime.lesson_transfer_weight), transfer_improvement),
        ]
        available = [(weight, value) for weight, value in tracks if value is not None]
        total_weight = sum(max(0.0, weight) for weight, _ in available)
        aggregate_improvement = (
            sum(max(0.0, weight) * float(value) for weight, value in available) / total_weight
            if total_weight else 0.0
        )

        fixed_soft_ok = lesson_fixed_gain_delta is None or lesson_fixed_gain_delta >= -float(self.config.runtime.lesson_max_fixed_regression)
        local_soft_ok = local_improvement >= -float(self.config.runtime.lesson_max_local_regression)
        transfer_soft_ok = transfer_improvement is None or transfer_improvement >= -float(self.config.runtime.lesson_max_transfer_regression)
        aggregate_ok = aggregate_improvement >= -float(self.config.runtime.lesson_max_aggregate_regression)
        catastrophic_fixed_ok = lesson_fixed_gain_delta is None or lesson_fixed_gain_delta >= -float(self.config.runtime.lesson_hard_fixed_regression)
        catastrophic_local_ok = local_improvement >= -float(self.config.runtime.lesson_hard_local_regression)
        catastrophic_transfer_ok = transfer_improvement is None or transfer_improvement >= -float(self.config.runtime.lesson_hard_transfer_regression)
        catastrophic_ok = catastrophic_fixed_ok and catastrophic_local_ok and catastrophic_transfer_ok
        lesson_accepted = aggregate_ok and catastrophic_ok

        if bool(self.config.runtime.lesson_rollback_on_fixed_regression) and not lesson_accepted:
            if lesson_model_snapshot is not None and lesson_optimizer_snapshot is not None:
                self.model.load_state_dict(lesson_model_snapshot)
                self.optimizer.load_state_dict(lesson_optimizer_snapshot)
                lesson_accepted = False
                self.last_language_loss = previous_language_loss
                last = previous_language_loss if previous_language_loss is not None else 0.0
                fixed_after = self.evaluate_fixed_reference()
                after_holdout = before_holdout
                transfer_after = transfer_before
                local_improvement = 0.0
                transfer_improvement = 0.0 if transfer_before is not None else None
                aggregate_improvement = 0.0
                # The final lesson rollback restores the pre-lesson weights, so dialogue
                # cannot remain marked accepted even when its own intermediate gate passed.
                if dialogue_examples:
                    dialogue_accepted = False

        unique_added = self._commit_unique_training_tokens(unique_digest, unique_pending) if lesson_accepted else 0
        progress = before_holdout - after_holdout
        self.last_holdout_bpb = after_holdout
        self.last_teaching_loss = after_holdout
        self.last_lesson_id = row_id
        self.last_language_loss = last
        self.last_lesson_accepted = lesson_accepted
        self.last_lesson_fixed_gain_delta = lesson_fixed_gain_delta

        dialogue_fixed_gain_final = fixed_after.get("gain") if dialogue_examples else None
        if dialogue_examples and dialogue_fixed_gain_before is not None and dialogue_fixed_gain_final is not None:
            dialogue_fixed_gain_delta = float(dialogue_fixed_gain_final) - float(dialogue_fixed_gain_before)
        if dialogue_examples:
            alignment_eval_after = self._alignment_eval(
                dialogue_eval_pairs,
                max_pairs=int(self.config.runtime.dialogue_alignment_eval_examples),
                seed=int(self.config.runtime.dialogue_alignment_eval_seed),
            )
        self.last_dialogue_accepted = dialogue_accepted
        self.last_dialogue_fixed_gain_delta = dialogue_fixed_gain_delta if dialogue_examples else None

        self.memory.update_learning_metrics(
            row_id,
            loss_after=after_holdout,
            learning_progress=progress,
            prediction_error=after_holdout,
            action_name="teacher" if lesson_accepted else "teacher_rollback",
            holdout_indices=json.dumps(holdout_idx),
            train_context=self.model.cfg.context,
        )
        self.memory.update_learning_status(row_id, accepted=lesson_accepted, replay_eligible=lesson_accepted)
        self.memory.replace_chunks(row_id, self._memory_chunk_texts(text, context=self.model.cfg.context))
        self._advance_runtime_state(text, None)

        result = {
            "lesson_id": row_id,
            "title": lesson.title,
            "source": lesson.source,
            "chars": len(text),
            "raw_chars": len(raw_text),
            "cleaned_chars": len(text),
            "transcript_cleaning": cleaning_meta,
            "chunks": len(all_chunks),
            "train_chunks": len(train_chunks),
            "holdout_chunks": len(holdout_chunks),
            "epochs": teach_epochs,
            "loss_before": before_holdout,
            "loss_after": after_holdout,
            "holdout_bpb_before": before_holdout,
            "holdout_bpb_after": after_holdout,
            "learning_progress": progress,
            "train_loss": last,
            "dialogue_examples": len(dialogue_examples),
            "dialogue_train_examples": dialogue_train_examples,
            "dialogue_alignment_loss": float(dialogue_alignment_stats.get("loss", 0.0)),
            "dialogue_alignment_steps": int(dialogue_alignment_stats.get("steps", 0)),
            "dialogue_alignment_pairs": int(dialogue_alignment_stats.get("pairs", 0)),
            "dialogue_alignment_eval_before": alignment_eval_before,
            "dialogue_alignment_eval_after": alignment_eval_after,
            "dialogue_alignment_eval_candidate_after": alignment_eval_candidate,
            "dialogue_accepted": dialogue_accepted,
            "dialogue_fixed_gain_before": dialogue_fixed_gain_before,
            "dialogue_fixed_gain_after": dialogue_fixed_gain_after,
            "dialogue_fixed_gain_final": dialogue_fixed_gain_final,
            "dialogue_fixed_gain_delta": dialogue_fixed_gain_delta,
            "fixed_eval_bpb": fixed_after.get("model_bpb"),
            "fixed_baseline_bpb": fixed_after.get("baseline_bpb"),
            "fixed_baseline_kind": fixed_after.get("baseline_kind"),
            "fixed_eval_gain": fixed_after.get("gain"),
            "fixed_eval_gain_before": fixed_before.get("gain"),
            "lesson_fixed_gain_before": lesson_fixed_gain_before,
            "lesson_fixed_gain_after": lesson_fixed_gain_after,
            "lesson_fixed_gain_delta": lesson_fixed_gain_delta,
            "local_improvement": local_improvement,
            "transfer_bpb_before": transfer_before,
            "transfer_bpb_after": transfer_after,
            "transfer_improvement": transfer_improvement,
            "aggregate_improvement": aggregate_improvement,
            "lesson_acceptance_tracks": {
                "fixed_ok": fixed_soft_ok,
                "local_ok": local_soft_ok,
                "transfer_ok": transfer_soft_ok,
                "aggregate_ok": aggregate_ok,
            },
            "lesson_acceptance_diagnostics": {
                "fixed_soft_ok": fixed_soft_ok,
                "local_soft_ok": local_soft_ok,
                "transfer_soft_ok": transfer_soft_ok,
                "aggregate_ok": aggregate_ok,
                "catastrophic_ok": catastrophic_ok,
            },
            "lesson_accepted": lesson_accepted,
            "lesson_rollback_threshold": float(self.config.runtime.lesson_max_fixed_regression),
            "lesson_hard_fixed_regression": float(self.config.runtime.lesson_hard_fixed_regression),
            "reference_bytes": fixed_after.get("reference_bytes", 0),
            "evaluation_excluded_training_paths": fixed_after.get("excluded_training_paths", []),
            "unique_training_tokens_total": self.total_tokens,
            "processed_training_tokens_total": self.processed_tokens,
            "unique_tokens_added": unique_added,
            "presented_unique_tokens": unique_pending,
            "replay_sequences_used": int(general_replay_stats.get("used", 0)),
            "replay_lessons_used": int(general_replay_stats.get("lesson_count", 0)),
            "replay_per_epoch": list(general_replay_stats.get("per_epoch", [])),
        }
        self._record({"event": "lesson", **result})
        self.save()
        return result

    def _parquet_files(self, path: Path) -> list[Path]:
        root = path.expanduser().resolve()
        if root.is_file() and root.suffix.lower() == ".parquet":
            return [root]
        if root.is_dir():
            return sorted(child for child in root.rglob("*.parquet") if child.is_file() and not any(part.startswith(".") for part in child.relative_to(root).parts))
        return []

    def _estimate_parquet_microbatches(self, paths: list[Path], context: int, sequence_length: int, batch_size: int, *, respect_cap: bool = True) -> int:
        total_bytes = self._parquet_text_uncompressed_bytes(paths)
        budget = int(self.config.runtime.pretrain_max_source_bytes)
        if respect_cap and budget > 0:
            total_bytes = min(total_bytes, budget)
        payload = max(1, int(context) - 2)
        chunks = max(1, math.ceil(total_bytes / payload))
        sequences = max(1, math.ceil(chunks / max(1, int(sequence_length))))
        return max(1, math.ceil(sequences / max(1, int(batch_size))))

    def _pretrain_shard_update(
        self,
        shard: Lesson,
        raw_text: str,
        *,
        epoch_index: int,
        shard_index: int,
        total_shards: int,
        requested_epochs: int,
        sequence_length: int,
        transcript_cleaning: dict | None = None,
        parquet: bool = False,
    ) -> tuple[dict, int]:
        """Train and record one foundation-pretraining shard consistently across sources."""
        all_chunks = self._encode_chunks(raw_text, max_chunks=0, context=self.model.cfg.context)
        sequences = self._as_sequences(all_chunks, sequence_length)
        local_probe = all_chunks[: min(32, len(all_chunks))]
        local_before = self._chunk_bpb(local_probe) if local_probe else 0.0
        train_token_count = sum(max(0, int(chunk.size(1) - 2)) for chunk in all_chunks)
        unique_digest, unique_pending = self._pending_unique_training_tokens(raw_text, train_token_count)
        train_loss = self._train_sequence_collection(
            sequences, epochs=1, replay=False, minimum_replay_sequences=0
        )
        if not math.isfinite(float(train_loss)):
            raise RuntimeError("pretraining produced a non-finite training loss")
        local_after = self._chunk_bpb(local_probe) if local_probe else 0.0
        unique_added = self._commit_unique_training_tokens(unique_digest, unique_pending)
        local_gain = local_before - local_after
        fixed = (
            self.evaluate_fixed_reference(use_ema=False)
            if shard_index % max(1, int(self.config.runtime.pretrain_eval_every_shards)) == 0
            else None
        )
        raw_bytes = len(raw_text.encode("utf-8", errors="replace"))
        self.pretrain_shards_seen += 1
        self.pretrain_bytes_seen += raw_bytes
        row_id = None
        if bool(getattr(self.config.runtime, "pretrain_record_memory", True)):
            record_text = (
                f"Foundation pretraining experience: source={shard.source}; bytes={raw_bytes}; "
                f"local_bpb={local_after:.6f}; gain={local_gain:.6f}"
            )
            row_id = self.memory.add(
                kind="lesson",
                source=shard.source,
                title=f"[pretrain epoch {epoch_index + 1}] {shard.title}",
                url=shard.source,
                text=record_text,
                loss_before=local_before,
                loss_after=local_after,
                learning_progress=local_gain,
                action_name="pretrain",
                prediction_error=max(0.0, local_after),
                train_context=self.model.cfg.context,
                accepted=True,
                replay_eligible=False,
                merge_existing=False,
            )
        self.last_lesson_id = row_id
        self.last_lesson_accepted = True
        self.last_lesson_fixed_gain_delta = None if fixed is None else fixed.get("gain")
        self.last_teaching_loss = train_loss
        result = {
            "epoch": epoch_index + 1,
            "epochs": int(requested_epochs),
            "shard": shard_index,
            "shards": total_shards,
            "title": shard.title,
            "source": shard.source,
            "chars": len(raw_text),
            "local_bpb_before": local_before,
            "local_bpb_after": local_after,
            "learning_progress": local_gain,
            "train_loss": train_loss,
            "unique_tokens_added": unique_added,
            "fixed_eval_bpb": None if fixed is None else fixed.get("model_bpb"),
            "fixed_baseline_bpb": None if fixed is None else fixed.get("baseline_bpb"),
            "fixed_eval_gain": None if fixed is None else fixed.get("gain"),
            "fixed_eval_mode": "raw",
            "optimizer_steps": self.training_steps,
            "processed_training_tokens": self.processed_tokens,
            "transcript_cleaning": transcript_cleaning or {},
            "parquet": bool(parquet),
        }
        self._record({"event": "pretrain_shard", **result})
        self.save()
        return result, raw_bytes

    def _pretrain_parquet(
        self,
        paths: list[Path],
        requested_epochs: int,
        base_seed: int,
    ) -> list[dict]:
        """Stream Parquet text rows through pretraining without materializing the corpus."""
        from .curriculum import iter_parquet_lessons
        sequence_length = max(1, int(self.config.runtime.pretrain_sequence_length))
        batch_size = max(1, int(self.config.runtime.train_batch_size))
        accumulation = max(1, int(self.config.runtime.pretrain_gradient_accumulation_steps))
        uncapped_microbatches_per_epoch = self._estimate_parquet_microbatches(
            paths, self.model.cfg.context, sequence_length, batch_size, respect_cap=False
        )
        total_bytes = self._parquet_text_uncompressed_bytes(paths)
        full_run_bytes = max(1, total_bytes) * requested_epochs
        configured_cap = int(self.config.runtime.pretrain_max_source_bytes)
        effective_bytes = min(full_run_bytes, configured_cap) if configured_cap > 0 else full_run_bytes
        uncapped_total_microbatches = max(1, uncapped_microbatches_per_epoch * requested_epochs)
        if full_run_bytes > 0:
            fraction = min(1.0, float(effective_bytes) / float(full_run_bytes))
            scheduled_microbatches = max(1, math.ceil(uncapped_total_microbatches * fraction))
            estimate_per_epoch = max(1, math.ceil(uncapped_microbatches_per_epoch * fraction))
        else:
            scheduled_microbatches = uncapped_total_microbatches
            estimate_per_epoch = uncapped_microbatches_per_epoch
        total_steps = max(1, math.ceil(scheduled_microbatches / accumulation))
        self._configure_pretraining(total_steps)
        results: list[dict] = []
        original_sequence_length = self.config.runtime.teacher_sequence_length
        original_steps_per_sequence = self.config.runtime.teacher_steps_per_sequence
        self.config.runtime.teacher_sequence_length = sequence_length
        self.config.runtime.teacher_steps_per_sequence = 1
        remaining_cap: int | None = configured_cap if configured_cap > 0 else None
        shard_count_estimate = max(1, estimate_per_epoch)
        try:
            for epoch_index in range(requested_epochs):
                shard_index = 0
                ordered_paths = list(paths)
                random.Random(int(base_seed) + epoch_index).shuffle(ordered_paths)
                for path in ordered_paths:
                    if remaining_cap is not None and remaining_cap <= 0:
                        break
                    iterator = iter_parquet_lessons(
                        path,
                        shard_max_chars=int(self.config.runtime.teacher_shard_max_chars) or 1_000_000,
                        text_column=str(self.config.runtime.parquet_text_column or "text"),
                        max_source_bytes=remaining_cap,
                    )
                    for shard in iterator:
                        shard_index += 1
                        raw_text = shard.text
                        if not raw_text.strip():
                            continue
                        result, consumed = self._pretrain_shard_update(
                            shard,
                            raw_text,
                            epoch_index=epoch_index,
                            shard_index=shard_index,
                            total_shards=shard_count_estimate,
                            requested_epochs=requested_epochs,
                            sequence_length=sequence_length,
                            parquet=True,
                        )
                        results.append(result)
                        if remaining_cap is not None:
                            remaining_cap = max(0, remaining_cap - consumed)
                        if remaining_cap is not None and remaining_cap <= 0:
                            break
                    if remaining_cap is not None and remaining_cap <= 0:
                        break
                if remaining_cap is not None and remaining_cap <= 0:
                    break
        finally:
            self.config.runtime.teacher_sequence_length = original_sequence_length
            self.config.runtime.teacher_steps_per_sequence = original_steps_per_sequence
            if bool(self.config.runtime.pretrain_finalize_ema) and self._ema_shadow is not None:
                self._apply_ema_to_model()
            self.optimizer = self._make_optimizer(self.config, self.model)
            self._training_mode = "continual"
            self._ema_shadow = None
            self._schedule_peak_lrs = []
            self._schedule_total_steps = 0
            self._schedule_step_start = 0
            self._schedule_warmup_steps = 0
            self._schedule_decay_steps = 0
            self._schedule_min_lr_ratio = 1.0
            self.save()
            if self.device.type == "cuda":
                torch.cuda.empty_cache()
            gc.collect()
        self.evaluate_fixed_reference(use_ema=False)
        return [{"pretrain_microbatches_per_epoch": estimate_per_epoch, **item} for item in results]

    @staticmethod
    def _utf8_prefix(text: str, budget: int) -> str:
        if budget <= 0:
            return ""
        raw = text.encode("utf-8", errors="replace")
        if len(raw) <= budget:
            return text
        return raw[:budget].decode("utf-8", errors="ignore").strip()

    def _parquet_text_uncompressed_bytes(self, paths: list[Path]) -> int:
        try:
            import pyarrow.parquet as pq
        except ImportError as exc:
            raise ValueError("Parquet support needs 'pyarrow': pip install pyarrow") from exc
        total_bytes = 0
        for path in paths:
            parquet = pq.ParquetFile(path)
            names = list(parquet.schema_arrow.names)
            requested = str(self.config.runtime.parquet_text_column or "text")
            candidates = [requested, "text", "content", "body"]
            selected = next((name for name in candidates if name and name in names), None)
            if selected is None:
                for name in names:
                    try:
                        if str(parquet.schema_arrow.field(name).type).lower() in {"string", "large_string"}:
                            selected = name
                            break
                    except Exception:
                        continue
            if selected is None:
                continue
            column_index = parquet.schema_arrow.get_field_index(selected)
            total_bytes += sum(
                int(parquet.metadata.row_group(i).column(column_index).total_uncompressed_size)
                for i in range(parquet.metadata.num_row_groups)
            )
        return total_bytes

    @staticmethod
    def _estimate_pretrain_microbatches(text: str, context: int, sequence_length: int, batch_size: int) -> int:
        raw_bytes = len(text.encode("utf-8", errors="replace"))
        payload = max(1, int(context) - 2)
        chunks = max(1, math.ceil(raw_bytes / payload))
        sequences = max(1, math.ceil(chunks / max(1, int(sequence_length))))
        return max(1, math.ceil(sequences / max(1, int(batch_size))))

    @staticmethod
    def _interleave_pretrain_shards(shards: list[Lesson], seed: int) -> list[Lesson]:
        groups: dict[str, list[Lesson]] = {}
        for shard in shards:
            groups.setdefault(shard.source.split("#shard=", 1)[0], []).append(shard)
        rng = random.Random(int(seed))
        source_names = list(groups)
        rng.shuffle(source_names)
        for name in source_names:
            rng.shuffle(groups[name])

        output: list[Lesson] = []
        remaining = True
        while remaining:
            remaining = False
            for name in source_names:
                bucket = groups[name]
                if bucket:
                    output.append(bucket.pop(0))
                    remaining = True
        return output

    def _load_pretrain_shards(self, path: Path) -> tuple[list[Lesson], dict[str, dict]]:
        source_path = path.expanduser().resolve()
        skipped: list[str] = []
        lessons = list(iter_text_files(
            source_path,
            max_chars=self.config.runtime.teacher_max_source_chars,
            on_skip=lambda skipped_path, reason: skipped.append(f"{skipped_path.name}: {reason}"),
        ))
        eligible_lessons = []
        for lesson in lessons:
            if self._is_fixed_evaluation_source(lesson.source):
                skipped.append(f"{lesson.source}: excluded fixed evaluation asset")
                continue
            eligible_lessons.append(lesson)
        lessons = eligible_lessons
        if not lessons:
            detail = f" (skipped: {'; '.join(skipped[:5])})" if skipped else ""
            raise ValueError(f"no readable text files found in {source_path}{detail}")

        shards: list[Lesson] = []
        cleaning_meta: dict[str, dict] = {}
        shard_limit = max(0, int(self.config.runtime.teacher_shard_max_chars))
        for lesson in lessons:
            cleaned, meta = prepare_language_text(
                lesson.text,
                strip_transcript_annotations=bool(self.config.runtime.pretrain_strip_transcript_annotations),
            )
            cleaned = self._exclude_evaluation_text(cleaned.strip())
            cleaning_meta[lesson.source] = meta
            if len(cleaned) < 8:
                continue
            shards.extend(
                split_lesson_text(
                    cleaned,
                    shard_limit,
                    title=lesson.title,
                    source=lesson.source,
                    kind="lesson",
                )
            )
        if not shards:
            raise ValueError(f"all readable pretraining text was empty after preprocessing: {source_path}")
        return shards, {"cleaning": cleaning_meta, "skipped": skipped}

    def pretrain(self, path: Path, *, epochs: int | None = None, seed: int | None = None) -> list[dict]:
        """Run broad foundation pretraining over interleaved source shards.

        This phase intentionally bypasses continual-learning acceptance gates. Each shard is
        still recorded as an accepted, non-replay lesson so the readiness gate can distinguish
        experienced foundation material from later replayable lifelong-learning episodes.
        """
        requested_epochs = max(1, int(epochs if epochs is not None else self.config.runtime.pretrain_epochs))
        base_seed = int(seed if seed is not None else self.config.runtime.pretrain_seed)
        parquet_paths = self._parquet_files(path)
        if parquet_paths:
            return self._pretrain_parquet(parquet_paths, requested_epochs, base_seed)
        shards, metadata = self._load_pretrain_shards(path)
        sequence_length = max(1, int(self.config.runtime.pretrain_sequence_length))
        batch_size = max(1, int(self.config.runtime.train_batch_size))
        accumulation = max(1, int(self.config.runtime.pretrain_gradient_accumulation_steps))

        microbatches_per_epoch = sum(
            self._estimate_pretrain_microbatches(
                shard.text, self.model.cfg.context, sequence_length, batch_size
            )
            for shard in shards
        )
        optimizer_steps_per_epoch = sum(
            max(1, math.ceil(
                self._estimate_pretrain_microbatches(
                    shard.text, self.model.cfg.context, sequence_length, batch_size
                ) / accumulation
            ))
            for shard in shards
        )
        configured_cap = int(self.config.runtime.pretrain_max_source_bytes)
        total_text_bytes = sum(len(shard.text.encode("utf-8", errors="replace")) for shard in shards)
        full_run_bytes = max(1, total_text_bytes) * requested_epochs
        effective_bytes = min(full_run_bytes, configured_cap) if configured_cap > 0 else full_run_bytes
        fraction = min(1.0, float(effective_bytes) / float(full_run_bytes)) if full_run_bytes else 1.0
        total_steps = max(1, math.ceil(optimizer_steps_per_epoch * requested_epochs * fraction))
        self._configure_pretraining(total_steps)
        results: list[dict] = []
        original_sequence_length = self.config.runtime.teacher_sequence_length
        original_steps_per_sequence = self.config.runtime.teacher_steps_per_sequence
        self.config.runtime.teacher_sequence_length = sequence_length
        self.config.runtime.teacher_steps_per_sequence = 1
        remaining_cap: int | None = configured_cap if configured_cap > 0 else None

        try:
            for epoch_index in range(requested_epochs):
                ordered = self._interleave_pretrain_shards(shards, base_seed + epoch_index)
                for shard_index, shard in enumerate(ordered, 1):
                    raw_text = shard.text
                    if remaining_cap is not None:
                        raw_text = self._utf8_prefix(raw_text, remaining_cap)
                        if not raw_text:
                            break
                    result, consumed = self._pretrain_shard_update(
                        shard,
                        raw_text,
                        epoch_index=epoch_index,
                        shard_index=shard_index,
                        total_shards=len(ordered),
                        requested_epochs=requested_epochs,
                        sequence_length=sequence_length,
                        transcript_cleaning=metadata["cleaning"].get(shard.source.split("#shard=", 1)[0], {}),
                        parquet=False,
                    )
                    results.append(result)
                    if remaining_cap is not None:
                        remaining_cap = max(0, remaining_cap - consumed)
                        if remaining_cap <= 0:
                            break
                if remaining_cap is not None and remaining_cap <= 0:
                    break
        finally:
            self.config.runtime.teacher_sequence_length = original_sequence_length
            self.config.runtime.teacher_steps_per_sequence = original_steps_per_sequence
            if bool(self.config.runtime.pretrain_finalize_ema) and self._ema_shadow is not None:
                self._apply_ema_to_model()
            # The pretraining optimizer hyperparameters/moments belong to a different
            # optimization phase. Continue lifelong learning with a fresh optimizer.
            self.optimizer = self._make_optimizer(self.config, self.model)
            self._training_mode = "continual"
            self._ema_shadow = None
            self._schedule_peak_lrs = []
            self._schedule_total_steps = 0
            self._schedule_step_start = 0
            self._schedule_warmup_steps = 0
            self._schedule_decay_steps = 0
            self._schedule_min_lr_ratio = 1.0
            self.save()
            if self.device.type == "cuda":
                torch.cuda.empty_cache()
            gc.collect()

        # The final public evaluation must describe the final live (EMA-finalized) weights.
        self.evaluate_fixed_reference(use_ema=False)
        return [{"pretrain_microbatches_per_epoch": microbatches_per_epoch, **item} for item in results]

    def teach_url(self, url: str, *, epochs: int | None = None, dialogue: bool = False) -> dict:
        pages = self.web.crawl_same_domain(
            url,
            max_pages=int(getattr(self.config.runtime, "web_crawl_max_pages", 8)),
            max_depth=int(getattr(self.config.runtime, "web_crawl_max_depth", 2)),
            max_attempts=int(getattr(self.config.runtime, "web_crawl_max_attempts", 32)),
        )
        if not pages:
            detail = " | ".join(self.web.last_errors[-8:])
            raise RuntimeError(f"website crawl produced no readable pages: {url}" + (f" | errors: {detail}" if detail else ""))
        results = []
        all_links = 0
        shard_limit = max(0, int(self.config.runtime.teacher_shard_max_chars))
        for page in pages:
            page_results = []
            shards = split_lesson_text(
                page.text,
                shard_limit,
                title=page.title,
                source=page.url,
                kind="lesson",
            )
            for shard in shards:
                result = self.teach_text(shard, epochs=epochs, dialogue=dialogue)
                page_results.append(result)
                results.append(result)
            best_progress = max((float(item.get("learning_progress", 0.0)) for item in page_results), default=0.0)
            self.memory.add_links(page.links, page.url, depth=page.depth, priority=best_progress)
            all_links += len(page.links)
            if page_results:
                page_results[0]["links"] = len(page.links)
                page_results[0]["crawl_depth"] = page.depth
                page_results[0]["content_type"] = page.content_type
        first = dict(results[0])
        first["pages_taught"] = len(pages)
        first["lessons_taught"] = len(results)
        first["total_links"] = all_links
        first["crawl_results"] = results
        return first

    def teach_file(self, path: Path, *, epochs: int | None = None, dialogue: bool = False) -> list[dict]:
        results: list[dict] = []
        skipped: list[str] = []
        excluded = self._evaluation_path()
        source_path = path.expanduser().resolve()
        if source_path.is_file():
            if self._is_fixed_evaluation_source(source_path):
                lessons = []
            elif source_path.suffix.lower() == ".parquet":
                lessons = iter_text_files(
                    source_path,
                    max_chars=self.config.runtime.teacher_max_source_chars,
                    parquet_max_source_bytes=(
                        int(self.config.runtime.teach_parquet_max_source_bytes)
                        if int(self.config.runtime.teach_parquet_max_source_bytes) > 0 else None
                    ),
                    on_skip=lambda skipped_path, reason: skipped.append(f"{skipped_path.name}: {reason}"),
                )
            else:
                lessons = [read_text_file(source_path, max_chars=self.config.runtime.teacher_max_source_chars)]
        else:
            lessons = (
                lesson
                for lesson in iter_text_files(
                    source_path,
                    max_chars=self.config.runtime.teacher_max_source_chars,
                    parquet_max_source_bytes=(
                        int(self.config.runtime.teach_parquet_max_source_bytes)
                        if int(self.config.runtime.teach_parquet_max_source_bytes) > 0 else None
                    ),
                    on_skip=lambda skipped_path, reason: skipped.append(f"{skipped_path.name}: {reason}"),
                )
                if not self._is_fixed_evaluation_source(lesson.source)
            )
        shard_limit = max(0, int(self.config.runtime.teacher_shard_max_chars))
        for lesson in lessons:
            shards = split_lesson_text(
                lesson.text,
                shard_limit,
                title=lesson.title,
                source=lesson.source,
                kind=lesson.kind,
            )
            for shard in shards:
                results.append(self.teach_text(shard, epochs=epochs, dialogue=dialogue))
        if not results:
            detail = f" (skipped: {'; '.join(skipped[:5])})" if skipped else ""
            raise ValueError(f"no readable text files found in {path}{detail}")
        if skipped:
            results[0]["skipped_files"] = skipped
        return results

    def teach_input(self, source: str, *, epochs: int | None = None, dialogue: bool = False) -> list[dict]:
        """Resolve an explicit teaching source without ambiguous fallback behavior.

        Supported forms are URLs, existing file/directory paths, and literal text
        prefixed with ``text:``. A missing path is an error; it is never silently
        converted into lesson text.
        """
        cleaned = source.strip()
        if not cleaned:
            raise ValueError("empty teaching source")

        if cleaned.startswith(("http://", "https://")):
            return [self.teach_url(cleaned, epochs=epochs, dialogue=dialogue)]

        if cleaned.startswith("text:"):
            text = cleaned[5:].lstrip()
            if not text:
                raise ValueError("text: teaching source is empty")
            return [self.teach_text(Lesson(title="teacher text", source="teacher://text", text=text), epochs=epochs, dialogue=dialogue)]

        if cleaned.startswith(("file:", "path:")):
            cleaned = cleaned.split(":", 1)[1].lstrip()

        # A bare --teach / /learn argument denotes a resource.  We detect
        # Windows paths explicitly (independent of the host OS) so a missing
        # path such as C:\curriculum\missing.txt can never silently become
        # lesson text. Literal text must use the explicit text: prefix.
        candidate = Path(cleaned).expanduser()
        windows_drive_path = len(cleaned) >= 3 and cleaned[1] == ":" and cleaned[2] in ("\\", "/")
        unix_or_relative_path = cleaned.startswith(("~", "./", "../", ".\\", "..\\", "/", "\\\\"))
        path_like = windows_drive_path or unix_or_relative_path or "/" in cleaned or "\\" in cleaned

        try:
            exists = candidate.exists()
        except OSError as exc:
            raise ValueError(f"invalid teaching path: {cleaned}") from exc

        if exists:
            return self.teach_file(candidate, epochs=epochs, dialogue=dialogue)
        if path_like:
            raise FileNotFoundError(f"teaching path does not exist: {cleaned}")

        raise ValueError("literal teaching text must use the text: prefix")

    def lessons(self, limit: int = 12) -> list[dict]:
        return self.memory.recent_lessons(limit)

    def teach_text_exact(self, text: str, *, title: str = "pasted text", source: str = "teacher://paste", epochs: int | None = None, dialogue: bool = False) -> dict:
        """Teach literal text without stripping or path interpretation."""
        if not str(text):
            raise ValueError("empty teaching text")
        return self.teach_text(Lesson(title=title, source=source, text=str(text)), epochs=epochs, dialogue=dialogue, preserve_literal=True)

    def _learning_progress_reference(self) -> tuple[str, float] | None:
        refs = self.memory.recent(1, kinds=("external",))
        if not refs:
            return None
        text = refs[0]["text"]
        return text, self.evaluate_text_bpb(text)

    def _web_learning_anchor(self, fixed_reference: dict | None = None) -> dict | None:
        """Return a pinned fixed-reference anchor for cumulative web-learning gates.

        The anchor is tied to the evaluation corpus fingerprint. If the corpus changes,
        a stale anchor must never be reused because it would compare against the wrong
        scientific reference.
        """
        path = self.root / "data" / "web_learning_anchor.json"
        fixed = fixed_reference or self.evaluate_fixed_reference()
        fingerprint = str(fixed.get("fingerprint", "")) if isinstance(fixed, dict) else ""
        try:
            if path.exists():
                payload = json.loads(path.read_text(encoding="utf-8"))
                if (
                    isinstance(payload, dict)
                    and payload.get("reference_fingerprint")
                    and (not fingerprint or str(payload.get("reference_fingerprint")) == fingerprint)
                ):
                    return payload
        except (OSError, ValueError, TypeError):
            pass
        if not fixed.get("available"):
            return None
        payload = {
            "schema_version": 1,
            "created": time.time(),
            "reference_fingerprint": str(fixed.get("fingerprint", "")),
            "anchor_model_bpb": float(fixed.get("model_bpb")),
            "anchor_baseline_bpb": float(fixed.get("baseline_bpb", 0.0)),
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        temp = path.with_suffix(path.suffix + ".tmp")
        temp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        os.replace(temp, path)
        return payload

    def _advance_runtime_state(self, page_text: str, action_name: str | None) -> None:
        action_index = self.registry.index(action_name) if action_name in self.registry.names() else None
        self.model.eval()
        with torch.no_grad():
            for idx, tokens in enumerate(self._encode_chunks(page_text, max_chunks=0, context=self.model.cfg.context)):
                prev_action = action_index if idx == 0 else None
                prev_error = self.prediction_error if idx == 0 else 0.0
                _, self.state, _, _ = self.model.step(tokens, self.state, prev_action, prev_error)

    def _select_transfer_probe(self, exclude_url: str = "") -> list[torch.Tensor]:
        probe: list[torch.Tensor] = []
        rows = [row for row in self.memory.recent(max(1, int(self.config.runtime.transfer_probe_documents) * 4), kinds=("external",)) if str(row.get("url", "")) != str(exclude_url)]
        seen = 0
        for row in rows:
            _, holdout = self._chunks_for_memory_row(row)
            if not holdout:
                continue
            probe.extend(holdout[:max(1, int(self.config.runtime.transfer_probe_chunks_per_document))])
            seen += 1
            if seen >= int(self.config.runtime.transfer_probe_documents):
                break
        return probe

    def _ingest(self, page: Page, *, source: str, action_name: str) -> dict:
        """Learn from an external observation with replay and independent rollback.

        External web learning is treated as another developmental lesson. The page is
        preserved in memory even when its parameter update is rejected, but the live
        model returns to the last validated state when independent evaluation regresses
        beyond the configured threshold.
        """
        text = self._exclude_evaluation_text(page.text.strip())
        if len(text) < 32:
            return {"learned": False, "kind": "empty_or_short", "url": page.url, "title": page.title, "valid_transition": False}
        # Reassess the exact text remaining after evaluation-overlap filtering; a
        # page-level extraction report cannot vouch for this sanitized training slice.
        quality = WebExplorer.assess_content_quality(text)
        source_quality = getattr(page, "quality_report", None)
        if isinstance(source_quality, dict) and source_quality:
            quality["source_extraction_status"] = source_quality.get("status", "unknown")
        if quality.get("status") == "low_quality":
            return {
                "learned": False,
                "accepted": False,
                "kind": "low_content_quality",
                "url": page.url,
                "title": page.title,
                "content_quality": quality,
                "valid_transition": False,
            }
        all_chunks = self._encode_chunks(
            text,
            max_chunks=max(0, int(self.config.runtime.max_text_chunks_per_observation)),
            context=self.model.cfg.context,
        )
        train_idx, holdout_idx = self._split_train_holdout_indices(len(all_chunks), float(self.config.runtime.teacher_holdout_fraction))
        train_chunks = [all_chunks[i] for i in train_idx]
        holdout_chunks = [all_chunks[i] for i in holdout_idx]
        if not train_chunks:
            return {"learned": False, "kind": "no_train_chunks", "url": page.url, "title": page.title, "valid_transition": False}

        local_before = self._chunk_bpb(holdout_chunks) if holdout_chunks else self._chunk_bpb(train_chunks)
        transfer_probe = self._select_transfer_probe(page.url)
        transfer_before = self._chunk_bpb(transfer_probe) if transfer_probe else None
        fixed_before = self.evaluate_fixed_reference()
        fixed_anchor = self._web_learning_anchor(fixed_before)

        model_snapshot = None
        optimizer_snapshot = None
        runtime_snapshot = None
        if bool(self.config.runtime.web_learning_rollback_on_fixed_regression):
            model_snapshot = {name: value.detach().clone() for name, value in self.model.state_dict().items()}
            optimizer_snapshot = copy.deepcopy(self.optimizer.state_dict())
            runtime_snapshot = (
                self.state.detach().clone(),
                self.previous_action,
                float(self.prediction_error),
            )

        train_token_count = sum(max(0, int(chunk.size(1) - 2)) for chunk in train_chunks)
        unique_digest, unique_pending = self._pending_unique_training_tokens(text, train_token_count)

        memory_id = self.memory.add(
            kind="external", source=source, title=page.title, url=page.url, text=text,
            loss_before=local_before, loss_after=local_before, learning_progress=0.0,
            action_name=f"{action_name}_pending", prediction_error=local_before,
            train_context=self.model.cfg.context, holdout_indices=json.dumps(holdout_idx),
            accepted=False, replay_eligible=False, merge_existing=False,
        )

        seq_len = max(1, int(self.config.runtime.sequence_length))
        max_train = int(self.config.runtime.observation_train_chunks)
        if max_train > 0 and len(train_chunks) > max_train:
            picks = [min(len(train_chunks) - 1, round(i * (len(train_chunks) - 1) / (max_train - 1))) for i in range(max_train)] if max_train > 1 else [0]
            train_chunks = [train_chunks[i] for i in dict.fromkeys(picks)]
        sequences = [
            ([*train_chunks[i:i + seq_len]], [None] * len(train_chunks[i:i + seq_len]), [local_before] + [0.0] * (len(train_chunks[i:i + seq_len]) - 1))
            for i in range(0, len(train_chunks), seq_len)
        ]
        self.model.train()

        old_steps = self.config.runtime.teacher_steps_per_sequence
        self.config.runtime.teacher_steps_per_sequence = max(1, int(self.config.runtime.observation_train_epochs))
        try:
            if bool(self.config.runtime.web_learning_use_replay):
                last = self._train_sequence_collection(
                    sequences,
                    epochs=1,
                    replay=True,
                    exclude_replay_ids={memory_id},
                    minimum_replay_sequences=int(self.config.runtime.replay_min_sequences),
                )
            else:
                last = self._train_sequence_collection(sequences, epochs=1, replay=False)
            replay_stats = dict(self.last_replay_stats)
        finally:
            self.config.runtime.teacher_steps_per_sequence = old_steps

        local_after = self._chunk_bpb(holdout_chunks) if holdout_chunks else self._chunk_bpb(train_chunks)
        local_progress = local_before - local_after
        transfer_after = self._chunk_bpb(transfer_probe) if transfer_probe else None
        transfer_progress = (transfer_before - transfer_after) if transfer_before is not None and transfer_after is not None else 0.0
        self.last_holdout_bpb = local_after
        self.prediction_error = float(local_after)
        self._advance_runtime_state(text, action_name)

        fixed_after = self.evaluate_fixed_reference()
        fixed_gain_before = fixed_before.get("gain")
        fixed_gain_after = fixed_after.get("gain")
        fixed_gain_delta = None
        anchor_regression = None
        anchor_bpb = None
        if fixed_anchor and fixed_after.get("model_bpb") is not None:
            anchor_bpb = float(fixed_anchor.get("anchor_model_bpb"))
            anchor_regression = float(fixed_after.get("model_bpb")) - anchor_bpb
        if fixed_gain_before is not None and fixed_gain_after is not None:
            fixed_gain_delta = float(fixed_gain_after) - float(fixed_gain_before)

        fixed_soft_ok = fixed_gain_delta is None or fixed_gain_delta >= -float(self.config.runtime.web_learning_max_fixed_regression)
        local_soft_ok = local_progress >= -float(self.config.runtime.web_learning_max_local_regression)
        transfer_soft_ok = transfer_progress >= -float(self.config.runtime.web_learning_max_transfer_regression)
        tracks = [
            (float(self.config.runtime.web_learning_fixed_weight), fixed_gain_delta),
            (float(self.config.runtime.web_learning_local_weight), local_progress),
            (float(self.config.runtime.web_learning_transfer_weight), transfer_progress),
        ]
        available = [(max(0.0, w), v) for w, v in tracks if v is not None]
        total_weight = sum(w for w, _ in available)
        aggregate_improvement = sum(w * float(v) for w, v in available) / total_weight if total_weight else 0.0
        catastrophic_fixed_ok = fixed_gain_delta is None or fixed_gain_delta >= -float(self.config.runtime.web_learning_hard_fixed_regression)
        catastrophic_local_ok = local_progress >= -float(self.config.runtime.web_learning_hard_local_regression)
        catastrophic_transfer_ok = transfer_progress >= -float(self.config.runtime.web_learning_hard_transfer_regression)
        max_anchor_regression = float(getattr(self.config.runtime, "web_learning_max_anchor_regression", 0.02))
        anchor_ok = anchor_regression is None or anchor_regression <= max_anchor_regression
        accepted = (aggregate_improvement >= -float(self.config.runtime.web_learning_max_aggregate_regression)
                    and catastrophic_fixed_ok and catastrophic_local_ok and catastrophic_transfer_ok and anchor_ok)
        threshold = float(getattr(self.config.runtime, "web_learning_max_fixed_regression", 0.03))
        if (
            bool(self.config.runtime.web_learning_rollback_on_fixed_regression)
            and not accepted
            and model_snapshot is not None
            and optimizer_snapshot is not None
            and runtime_snapshot is not None
        ):
            self.model.load_state_dict(model_snapshot)
            self.optimizer.load_state_dict(optimizer_snapshot)
            self.state = runtime_snapshot[0]
            self.previous_action = runtime_snapshot[1]
            self.prediction_error = runtime_snapshot[2]
            accepted = False
            local_after = local_before
            local_progress = 0.0
            transfer_progress = 0.0
            last = 0.0
            fixed_after = self.evaluate_fixed_reference()
            fixed_gain_delta = (
                float(fixed_after.get("gain")) - float(fixed_gain_before)
                if fixed_after.get("gain") is not None and fixed_gain_before is not None
                else None
            )
            aggregate_improvement = 0.0

        self.memory.update_learning_metrics(
            memory_id,
            loss_after=local_after,
            learning_progress=transfer_progress if accepted else 0.0,
            prediction_error=local_after,
            action_name=action_name if accepted else f"{action_name}_rollback",
        )
        self.memory.update_learning_status(memory_id, accepted=accepted, replay_eligible=accepted)
        unique_added = self._commit_unique_training_tokens(unique_digest, unique_pending) if accepted else 0
        self.memory.add_links(page.links, page.url, depth=0, priority=max(0.0, transfer_progress) if accepted else 0.0)

        return {
            "learned": True, "accepted": accepted, "kind": "observation", "source": source,
            "title": page.title, "url": page.url, "loss_before": local_before, "loss_after": local_after,
            "learning_progress": transfer_progress if accepted else 0.0,
            "local_holdout_progress": local_progress, "transfer_progress": transfer_progress if accepted else 0.0,
            "train_loss": last, "links": len(page.links), "chars": len(text), "memory_id": memory_id,
            "holdout_bpb": local_after,
            "fixed_eval_bpb": fixed_after.get("model_bpb"),
            "fixed_baseline_bpb": fixed_after.get("baseline_bpb"),
            "fixed_eval_gain": fixed_after.get("gain"),
            "fixed_eval_gain_before": fixed_gain_before,
            "fixed_eval_gain_delta": fixed_gain_delta,
            "anchor_model_bpb": anchor_bpb,
            "anchor_regression": anchor_regression,
            "anchor_regression_limit": max_anchor_regression,
            "anchor_ok": anchor_ok,
            "rollback_threshold": threshold,
            "hard_fixed_regression": float(self.config.runtime.web_learning_hard_fixed_regression),
            "local_progress": local_progress,
            "transfer_progress": transfer_progress,
            "aggregate_improvement": aggregate_improvement,
            "acceptance_tracks": {
                "fixed_soft_ok": fixed_soft_ok,
                "local_soft_ok": local_soft_ok,
                "transfer_soft_ok": transfer_soft_ok,
                "catastrophic_ok": catastrophic_fixed_ok and catastrophic_local_ok and catastrophic_transfer_ok,
                "anchor_ok": anchor_ok,
            },
            "accepted_final": accepted,
            "content_quality": quality,
            "replay_sequences_used": int(replay_stats.get("used", 0)),
            "replay_lessons_used": int(replay_stats.get("lesson_count", 0)),
            "replay_per_epoch": list(replay_stats.get("per_epoch", [])),
            "unique_tokens_added": unique_added,
            "presented_unique_tokens": unique_pending,
            "valid_transition": accepted,
        }
    def _text_representation(self, text: str) -> torch.Tensor:
        tokens = self._encode_text(text)
        return self.model.text_representation(tokens)

    def _evaluation_terms(self) -> set[str]:
        try:
            text = load_reference(self.root, self.config.runtime.language_eval_path, 0)
        except FileNotFoundError:
            return set()
        return set(lexical_terms(text))

    def _informative_terms(self, message: str) -> list[str]:
        terms = lexical_terms(message)
        if not terms:
            return []
        background = self._evaluation_terms()
        # Once a fixed background corpus exists, rarity is estimated against the
        # whole learned corpus plus that independent reference. Before the corpus
        # exists, keep the lexical vocabulary intact and let candidate matching
        # require multiple informative terms.
        if not background:
            return terms
        total_docs = self.memory.document_count() + 1
        informative = []
        common_fraction = float(self.config.runtime.chat_common_term_doc_fraction)
        for term in terms:
            in_reference = term in background
            df = self.memory.term_document_frequency(term) + (1 if in_reference else 0)
            # Terms absent from the independent background are potentially domain-specific.
            # Terms present across at least half of the combined memory/background corpus are
            # treated as generic for retrieval ranking without naming a hard-coded stopword list.
            if (not in_reference) or df == 0 or (df / total_docs) < common_fraction:
                informative.append(term)
        return informative

    def _retrieve_experience(self, message: str, limit: int | None = None) -> list[dict]:
        limit=max(1, int(limit or self.config.runtime.chat_memory_results))
        terms=self._informative_terms(message)
        min_terms=max(1, int(self.config.runtime.chat_min_retrieval_terms))
        if len(terms) < min_terms:
            if not (len(terms) == 1 and len(terms[0]) >= int(self.config.runtime.chat_min_single_term_chars)):
                return []
        query=" ".join(terms)
        candidates=self.memory.search_chunks(query, max(limit * 8, 32))
        if not candidates:
            candidates=[row for row in self.memory.search(query, max(limit * 4, 12)) if row.get("kind") in {"external", "lesson"}]
        if not candidates:
            return []

        safe_candidates = []
        for row in candidates:
            item = dict(row)
            original_text = str(item.get("text", ""))
            safe_text = self._exclude_evaluation_text(original_text).strip()
            if len(safe_text) < 8:
                continue
            if safe_text != original_text:
                item["text"] = safe_text
                item["evaluation_overlap_filtered"] = True
            safe_candidates.append(item)
        candidates = safe_candidates
        if not candidates:
            return []

        query_vec=self._text_representation(message)
        token_sets={}
        for row in candidates:
            token_sets[int(row["id"])] = set(lexical_terms(str(row.get("text", ""))[:6000]))
        document_frequency={term: sum(1 for row_terms in token_sets.values() if term in row_terms) for term in terms}
        total_docs=max(1, len(candidates))
        scored=[]
        with torch.no_grad():
            for row in candidates:
                row_terms=token_sets[int(row["id"])]
                matched=sum(1 for term in terms if term in row_terms)
                if matched < max(1, int(self.config.runtime.chat_min_matching_terms)):
                    continue
                lexical=matched/max(1, len(terms))
                if lexical < float(self.config.runtime.chat_min_lexical_score):
                    continue
                vec=self._text_representation(str(row.get("text", ""))[: self.config.runtime.chat_memory_chars])
                semantic=max(0.0, float(torch.sum(query_vec * vec).item()))
                idf=sum(math.log((1 + total_docs)/(1 + document_frequency[term])) + 1.0 for term in terms if term in row_terms)
                score=0.60*lexical + 0.30*semantic + 0.10*(idf/max(1,len(terms)))
                scored.append((score, lexical, semantic, row))
        scored.sort(key=lambda item:(item[0], item[1], item[2]), reverse=True)
        results = []
        for score, lexical, semantic, row in scored[:limit]:
            item = dict(row)
            item["_retrieval"] = {
                "score": float(score), "lexical": float(lexical),
                "representation_similarity": float(semantic),
                "source": str(row.get("url") or row.get("source") or "local memory"),
                "method": "lexical_plus_learned_representation",
            }
            results.append(item)
        return results

    def _recent_communication_context(self, limit: int | None = None) -> str:
        limit=max(1, int(limit or 4))
        rows=self.memory.recent(limit*2, kinds=("conversation_user", "conversation_response"))
        rows=list(reversed(rows))
        parts=[]
        for row in rows[-limit*2:]:
            title=str(row.get("title", "")).strip()
            safe = self._exclude_evaluation_text(str(row.get("text", ""))).strip()
            text=" ".join(safe.split())
            if text:
                parts.append(f"{title}: {text[:240]}")
        return "\n".join(parts)

    def _prompt_with_context(self, message: str, rows: list[dict], *, reply_reserve: int = 128) -> str:
        evidence=[]
        for row in rows[: max(1, int(self.config.runtime.chat_memory_results))]:
            title=" ".join(str(row.get("title", "memory")).split())[:160]
            source=" ".join(str(row.get("url") or row.get("source") or "local memory").split())[:240]
            text=" ".join(str(row.get("text", "")).split())
            if text:
                # JSON quoting helps keep arbitrary retrieved page text clearly bounded as
                # data. It is not a security sandbox; callers must still treat web text as untrusted.
                quoted = json.dumps(text[:int(self.config.runtime.chat_memory_chars)], ensure_ascii=False)
                evidence.append(f"[Retrieved evidence; source={source}; title={title}]\n{quoted}\n[End retrieved evidence]")
        prior=self._recent_communication_context(4)
        context_parts=[]
        objective = self.objectives.current()
        if objective is not None:
            context_parts.append("Current objective:\n" + objective.text + ("\nKnowledge gaps: " + ", ".join(objective.knowledge_gaps) if objective.knowledge_gaps else ""))
        if evidence:
            context_parts.append("Learned material:\n" + "\n".join(evidence))
        if prior:
            context_parts.append("Recent interaction:\n" + prior)
        context="\n\n".join(context_parts)
        # Tokenize with the active tokenizer; character/byte counts are not reliable
        # once a learned BPE tokenizer is installed. Keep room for the generated reply.
        context_limit = max(8, int(self.model.cfg.context))
        reserve = min(max(1, int(reply_reserve)), max(1, context_limit // 2))
        prompt_limit = max(2, context_limit - reserve)
        suffix = format_turn_prompt(message)
        suffix_ids = self.tokenizer.encode(suffix, bos=False, eos=False)
        if len(suffix_ids) + 1 > prompt_limit:
            # Preserve the newest end of a too-long user message, and never allow the
            # prompt itself to overflow the context or cut off the Turn B marker.
            message_ids = self.tokenizer.encode(message, bos=False, eos=False)
            allowance = max(1, prompt_limit - len(self.tokenizer.encode(format_turn_prompt(""), bos=False, eos=False)) - 1)
            message = self.tokenizer.decode(message_ids[-allowance:])
            suffix = format_turn_prompt(message)
            suffix_ids = self.tokenizer.encode(suffix, bos=False, eos=False)
            while len(suffix_ids) + 1 > prompt_limit and message:
                message_ids = self.tokenizer.encode(message, bos=False, eos=False)
                message = self.tokenizer.decode(message_ids[1:])
                suffix = format_turn_prompt(message)
                suffix_ids = self.tokenizer.encode(suffix, bos=False, eos=False)
        context_allowance = max(0, prompt_limit - len(suffix_ids) - 1)
        context_ids = self.tokenizer.encode(context, bos=False, eos=False) if context else []
        if context_allowance and len(context_ids) > context_allowance:
            context_ids = context_ids[-context_allowance:]
        elif not context_allowance:
            context_ids = []
        context = self.tokenizer.decode(context_ids) if context_ids else ""
        prompt = (context + "\n\n" if context else "") + suffix
        # Defensive check against tokenizer edge cases: preserve the rightmost prompt
        # tokens only as a last resort, while keeping generation inside the context.
        prompt_ids = self.tokenizer.encode(prompt, bos=False, eos=False)
        if len(prompt_ids) + 1 > prompt_limit:
            prompt_ids = prompt_ids[-max(1, prompt_limit - 1):]
            prompt = self.tokenizer.decode(prompt_ids)
        return prompt

    def _communication_context(self, message: str, *, reply_reserve: int = 128) -> str:
        return self._prompt_with_context(message, self._retrieve_experience(message), reply_reserve=reply_reserve)

    def _workspace(self, limit: int = 6, chars: int = 800) -> str:
        rows = self.memory.recent(limit, kinds=("external", "conversation_user"))
        return "\n\n".join(f"{x['title']}\n{x['text'][:chars]}" for x in rows)

    def _language_ready(self) -> bool:
        tokens_ok = self.total_tokens >= int(self.config.runtime.language_ready_min_tokens)
        learned_units = self.memory.lesson_count() + int(self.pretrain_shards_seen)
        lessons_ok = learned_units >= int(self.config.runtime.language_ready_min_lessons)
        reference_ok = (
            self.last_eval_bpb is not None
            and self.eval_baseline_bpb is not None
            and self.eval_baseline_bpb > 0
            and self.eval_reference_bytes >= int(self.config.runtime.language_ready_min_eval_bytes)
        )
        gain = (self.eval_baseline_bpb - self.last_eval_bpb) if reference_ok else 0.0
        ratio = (self.last_eval_bpb / self.eval_baseline_bpb) if reference_ok else 1.0
        gain_ok = reference_ok and gain >= float(self.config.runtime.language_ready_min_bpb_gain)
        ratio_ok = reference_ok and ratio <= float(self.config.runtime.language_ready_max_baseline_ratio)
        return tokens_ok and lessons_ok and gain_ok and ratio_ok

    def _query(self) -> str:
        seed = self.memory.query_seed()
        if self._language_ready():
            context = self._workspace(4, 500)
            generated = self.model.generate(
                context + "\n\n" if context else seed,
                max_new_tokens=80,
                temperature=self.config.runtime.generation_temperature,
                top_k=self.config.runtime.generation_top_k,
                device=self.device,
                state=self.state,
            )
            query = " ".join(generated.replace("\n", " ").split())[: self.config.runtime.max_query_chars].strip(" .,:;!?-_\"")
            if len(query) >= 3 and any(ch.isalpha() for ch in query):
                return query
        return seed[: self.config.runtime.max_query_chars]

    def _choose_action(self, *, exclude_names: set[str] | None = None) -> tuple[int, str, float, torch.Tensor]:
        memory_count = self.memory.summary()["count"]
        items = self.registry.items()
        excluded = {str(name) for name in (exclude_names or set())}
        logits = self.model.action_logits(self.state).squeeze(0)
        valid = torch.zeros(len(items), dtype=torch.bool, device=self.device)
        for i, item in enumerate(items):
            valid[i] = item.name not in excluded and item.available() and memory_count >= item.requires_memory and (
                not item.requires_language or self._language_ready()
            )

        # FINAL20 model-based planning is generic: it sees only the learned latent
        # state, the current objective representation, action interfaces, uncertainty,
        # and the learned transition model. No world-specific action semantics are encoded.
        objective = self.objectives.current()
        if (
            objective is not None
            and bool(getattr(self.config.runtime, "planning_enabled", True))
            and self.planner.enabled
            and bool(valid.any())
        ):
            try:
                active_goal = self.planner.hierarchy.text(objective.id, objective.text)
                if not self.planner.hierarchy.get(objective.id):
                    self._ensure_objective_plan(objective)
                    active_goal = self.planner.hierarchy.text(objective.id, objective.text)
                planned = self.planner.plan(
                    self.state,
                    active_goal,
                    prediction_error=self.prediction_error,
                    valid_mask=valid,
                    previous_action=self.previous_action,
                )
                self._last_plan = planned
                if planned.get("enabled") and planned.get("action_index") is not None:
                    return (
                        int(planned["action_index"]),
                        "planned",
                        float(planned.get("entropy", 0.0)),
                        valid.detach().clone(),
                    )
            except Exception as exc:
                self._last_plan = {"enabled": False, "reason": "planner_error", "error": repr(exc)}

        if memory_count < self.config.runtime.minimum_bootstrap_memories:
            eligible = [
                i for i, item in enumerate(items)
                if item.external and item.produces_observation and bool(valid[i])
            ]
            if eligible:
                weights = torch.tensor(
                    [max(0.001, items[i].warmup_priority) for i in eligible],
                    device=self.device,
                )
                probs = weights / weights.sum()
                idx = eligible[int(torch.multinomial(probs, 1).item())]
                entropy = float((-(probs * probs.clamp_min(1e-8).log()).sum()).detach().cpu())
                self._last_policy_mask = valid.detach().clone()
                return idx, "warmup", entropy, valid.detach().clone()

        eligible = valid.nonzero(as_tuple=False).flatten().tolist()
        if not eligible:
            fallback = [
                i for i, item in enumerate(items)
                if item.external and item.produces_observation and item.available()
            ] or list(range(len(items)))
            idx = random.choice(fallback)
            mask = torch.zeros(len(items), dtype=torch.bool, device=self.device)
            mask[idx] = True
            self._last_policy_mask = mask.detach().clone()
            return idx, "fallback", 0.0, mask.detach().clone()
        if random.random() < self.config.runtime.policy_epsilon:
            idx = random.choice(eligible)
            self._last_policy_mask = valid.detach().clone()
            return idx, "epsilon", 0.0, valid.detach().clone()
        masked = logits.clone()
        masked[~valid] = float("-inf")
        probs = torch.softmax(masked, dim=-1)
        entropy = float((-(probs[valid] * probs[valid].clamp_min(1e-8).log()).sum()).detach().cpu())
        idx = int(torch.distributions.Categorical(probs=probs).sample().item())
        self._last_policy_mask = valid.detach().clone()
        return idx, "policy", entropy, valid.detach().clone()

    def _learn_policy(self, state_before: torch.Tensor, action: int, signal: float, valid_transition: bool, selection_mode: str, valid_mask: torch.Tensor | None = None) -> None:
        if selection_mode not in {"policy", "planned"} or not valid_transition or not math.isfinite(signal):
            return False
        items = self.registry.items()
        if valid_mask is None:
            memory_count = self.memory.summary()["count"]
            valid_mask = torch.tensor([
                item.available() and memory_count >= item.requires_memory and (not item.requires_language or self._language_ready())
                for item in items
            ], dtype=torch.bool, device=self.device)
        if action < 0 or action >= len(items) or not items[action].policy_train or not bool(valid_mask[action]):
            return False
        logits = self.model.action_logits(state_before).squeeze(0)
        mask = valid_mask.to(device=logits.device, dtype=torch.bool)
        masked = logits.masked_fill(~mask, float("-inf"))
        log_probs = F.log_softmax(masked, dim=-1)
        probs = log_probs.exp()
        clipped = max(-self.config.runtime.train_progress_clip, min(self.config.runtime.train_progress_clip, float(signal)))
        advantage = clipped - self.policy_baseline
        log_prob = log_probs[action]
        entropy = -(probs[mask] * log_probs[mask]).sum()
        loss = -(torch.tensor(advantage, device=self.device) * log_prob) - self.config.runtime.policy_entropy_weight * entropy
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.action_head.parameters(), 1.0)
        self.optimizer.step()
        decay = float(self.config.runtime.policy_baseline_decay)
        self.policy_baseline = decay * self.policy_baseline + (1.0 - decay) * clipped
        return True


    def _load_perception_state(self) -> None:
        path = self.paths.data / "perception.pt"
        if not path.exists():
            return
        try:
            payload = torch.load(path, map_location=self.device, weights_only=True)
            if isinstance(payload, dict):
                self.perception.vision.load_state_dict(payload.get("vision", {}), strict=False)
                self.perception.audio.load_state_dict(payload.get("audio", {}), strict=False)
                self.perception.object_discovery.load_state_dict(payload.get("object_discovery", {}), strict=False)
                if payload.get("vision_optimizer"):
                    try:
                        self.perception.vision_optimizer.load_state_dict(payload["vision_optimizer"])
                    except Exception:
                        pass
                if payload.get("audio_optimizer"):
                    try:
                        self.perception.audio_optimizer.load_state_dict(payload["audio_optimizer"])
                    except Exception:
                        pass
                self.perception.vision_steps = int(payload.get("vision_steps", 0))
                self.perception.audio_steps = int(payload.get("audio_steps", 0))
        except Exception:
            pass

    def _save_perception_state(self) -> None:
        path = self.paths.data / "perception.pt"
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "vision": self.perception.vision.state_dict(),
            "object_discovery": self.perception.object_discovery.state_dict(),
            "audio": self.perception.audio.state_dict(),
            "vision_optimizer": self.perception.vision_optimizer.state_dict(),
            "audio_optimizer": self.perception.audio_optimizer.state_dict(),
            "vision_steps": self.perception.vision_steps,
            "audio_steps": self.perception.audio_steps,
        }, path)

    def _planning_generator(self, prompt: str) -> str:
        return self.model.generate(
            prompt,
            max_new_tokens=160,
            temperature=float(self.config.runtime.generation_temperature),
            top_k=int(self.config.runtime.generation_top_k),
            device=self.device,
            state=self.state,
        )

    def _ensure_objective_plan(self, objective) -> list[dict]:
        generate = None
        if (
            bool(getattr(self.config.runtime, "planning_decompose_on_objective", True))
            and self._language_ready()
        ):
            generate = self._planning_generator
        steps = self.planner.hierarchy.ensure(
            objective.id,
            objective.text,
            generator=generate,
            max_steps=int(getattr(self.config.runtime, "planning_subgoals", 6)),
        )
        return [asdict(step) if hasattr(step, "__dataclass_fields__") else dict(step) for step in steps]

    def planning_status(self) -> dict:
        objective = self.objectives.current()
        if objective is None:
            return {"enabled": bool(self.planner.enabled), "objective": None, "steps": [], "plan": {}}
        steps = self._ensure_objective_plan(objective)
        active = self.planner.hierarchy.text(objective.id, objective.text)
        memory_count = self.memory.summary()["count"]
        items = self.registry.items()
        valid = torch.tensor(
            [
                item.available()
                and memory_count >= item.requires_memory
                and (not item.requires_language or self._language_ready())
                for item in items
            ],
            dtype=torch.bool,
            device=self.device,
        )
        plan = self.planner.plan(
            self.state,
            active,
            prediction_error=self.prediction_error,
            valid_mask=valid,
            previous_action=self.previous_action,
        )
        return {
            "enabled": bool(self.planner.enabled),
            "objective": objective.text,
            "active_subgoal": active,
            "steps": steps,
            "plan": plan,
            "planner_training_steps": self.planner.training_steps,
        }

    def set_objective(self, text: str, *, priority: float = 1.0, verifier_type: str | None = None, completion_threshold: float | None = None, required_consecutive: int | None = None) -> dict:
        objective = self.objectives.add(
            text,
            priority=priority,
            verifier_type=str(verifier_type or getattr(self.config.runtime, "objective_verifier_default", "manual")),
            completion_threshold=float(completion_threshold if completion_threshold is not None else getattr(self.config.runtime, "objective_completion_threshold", 0.95)),
            required_consecutive=int(required_consecutive if required_consecutive is not None else getattr(self.config.runtime, "objective_required_consecutive", 1)),
        )
        steps = self._ensure_objective_plan(objective)
        self._record({
            "event": "objective_created",
            "objective": objective.text,
            "objective_id": objective.id,
            "planning_steps": len(steps),
        })
        return {
            "id": objective.id,
            "text": objective.text,
            "status": objective.status,
            "priority": objective.priority,
            "verifier_type": objective.verifier_type,
            "completion_threshold": objective.completion_threshold,
            "required_consecutive": objective.required_consecutive,
            "planning_steps": len(steps),
            "plan": [asdict(step) for step in self.planner.hierarchy.get(objective.id)],
        }

    def objectives_status(self) -> list[dict]:
        from dataclasses import asdict
        return [asdict(item) for item in self.objectives.all()]

    def learn_multimodal_pairs(self, pairs: list[dict], *, steps: int | None = None) -> dict:
        """Train cross-modal alignment from explicitly paired observations.

        Each item may contain ``text`` and one or both of ``image``/``audio``.
        Pairing is supplied by the teacher/environment; no semantic labels are assumed.
        """
        usable = [item for item in pairs if isinstance(item, dict) and str(item.get("text", "")).strip()]
        if len(usable) < 2:
            return {"trained": False, "reason": "need_at_least_two_pairs", "pairs": len(usable), "steps": 0}
        text_tokens = [self._encode_text(str(item["text"])) for item in usable]
        batches = []
        for item in usable:
            vision = None
            audio = None
            if item.get("image") is not None:
                image = self.perception.vision.prepare_image(item["image"]).to(self.device)
                with torch.no_grad():
                    vision = self.perception.vision.encode_raw(image)
            if item.get("audio") is not None:
                waveform = self.perception._fixed_audio_window(torch.as_tensor(item["audio"], dtype=torch.float32)).unsqueeze(0).to(self.device)
                with torch.no_grad():
                    audio = self.perception.audio.encode_raw(waveform)
            batches.append((vision, audio))
        vision_batch = torch.cat([v for v, _ in batches], dim=0) if all(v is not None for v, _ in batches) else None
        audio_batch = torch.cat([a for _, a in batches], dim=0) if all(a is not None for _, a in batches) else None
        total_steps = max(1, int(steps if steps is not None else getattr(self.config.runtime, "dialogue_train_epochs", 1)))
        self.model.train()
        last = None
        try:
            for _ in range(total_steps):
                self.optimizer.zero_grad(set_to_none=True)
                loss = self.model.cross_modal_alignment_loss(
                    torch.cat(text_tokens, dim=0), vision_embeddings=vision_batch, audio_embeddings=audio_batch,
                    temperature=float(getattr(self.config.runtime, "dialogue_alignment_temperature", 0.12)),
                )
                if not torch.isfinite(loss):
                    return {"trained": False, "reason": "non_finite_alignment_loss", "pairs": len(usable), "steps": 0}
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()
                last = float(loss.detach().cpu())
        finally:
            self.model.eval()
        return {"trained": True, "pairs": len(usable), "steps": total_steps, "loss": last}

    def observe_image(self, source: str | Path, *, alias: str = "", train: bool = True) -> dict:
        result = self.perception.observe_image(source, alias=alias, train=train)
        observation_text = (
            f"visual observation entity={result['entity_id']} score={result['score']:.4f} "
            f"new={result['new']} objects={result.get('object_count', 0)} alias={alias or 'none'}"
        )
        memory_id = self.memory.add(
            kind="sensory_vision", source="vision", title="visual observation", url="", text=observation_text,
        )
        novelty = 0.0
        try:
            tensor = self.perception.vision.prepare_image(source).to(self.device)
            with torch.no_grad():
                embedding = self.perception.last_scene_embedding
                if embedding is None:
                    embedding = self.perception.vision(tensor)[0]
                if self._last_visual_embedding is not None:
                    novelty = max(0.0, 1.0 - float(torch.sum(embedding * self._last_visual_embedding).item()))
                self._last_visual_embedding = embedding.detach()
                self.state = self.model.integrate_perception(embedding, self.state, "vision").detach()
            cognitive_event_id = self._record_cognitive_event(
                "visual_observation", modality="vision", source=str(source),
                entity_id=str(result.get("entity_id", "")) or None, novelty=novelty, salience=novelty,
                detail={"recognized": bool(result.get("recognized", False)), "score": float(result.get("score", 0.0)), "new": bool(result.get("new", False)),
                        "object_count": int(result.get("object_count", 0)),
                        "objects": [{k: v for k, v in item.items() if k not in {"embedding"}} for item in result.get("objects", [])]},
            )
        except Exception as exc:
            cognitive_event_id = self._record_cognitive_event(
                "visual_observation_error", modality="vision", source=str(source),
                salience=0.0, novelty=0.0, detail={"error": repr(exc)},
            )
        self._record({"event": "vision", "memory_id": memory_id, "cognitive_event_id": cognitive_event_id, "novelty": novelty, **result})
        self.perception.flush()
        self._save_perception_state()
        self.save()
        return {**result, "memory_id": memory_id, "cognitive_event_id": cognitive_event_id, "novelty": novelty}

    def recognize_image(self, source: str | Path) -> dict | None:
        return self.perception.recognize_image(source)

    def hear_audio(self, source: str | Path, *, train: bool = True) -> dict:
        result = self.perception.observe_audio(source, train=train)
        memory_id = self.memory.add(
            kind="sensory_audio", source="microphone", title="audio observation", url="",
            text=f"audio observation sample_rate={result['sample_rate']} embedding_dim={result['embedding_dim']}",
        )
        waveform, _ = self.perception.audio.load_wav(source)
        with torch.no_grad():
            embedding = self.perception.audio(waveform.to(self.device))[0]
            self.state = self.model.integrate_perception(embedding, self.state, "audio").detach()
        cognitive_event_id = self._record_cognitive_event(
            "audio_observation", modality="audio", source="microphone",
            novelty=min(1.0, float(result.get("norm", 0.0))), salience=min(1.0, float(result.get("norm", 0.0))),
            detail={"sample_rate": int(result.get("sample_rate", 0)), "embedding_dim": int(result.get("embedding_dim", 0))},
        )
        self._record({"event": "audio", "memory_id": memory_id, "cognitive_event_id": cognitive_event_id, **result})
        self.perception.flush()
        self._save_perception_state()
        self.save()
        return {**result, "memory_id": memory_id, "cognitive_event_id": cognitive_event_id}

    def see_camera(self, *, device_index: int = 0, frames: int = 0, alias: str = "", show: bool = False) -> list[dict]:
        results = self.perception.camera_stream(device_index=device_index, frames=frames, alias=alias, show=show)
        for result in results[-1:]:
            self.memory.add(
                kind="sensory_vision", source="camera", title="camera observation", url="",
                text=f"camera observation entity={result['entity_id']} score={result['score']:.4f} new={result['new']}",
            )
        if results:
            self._record({"event": "camera", "frames": len(results), "last": results[-1]})
            self.perception.flush()
            self._save_perception_state()
            self.save()
        return results

    def speak(self, text: str, *, speaker: str = "", output: str | Path | None = None, play: bool | None = None) -> dict:
        text = " ".join(str(text).split()).strip()
        learned_preferred = bool(getattr(self.config.runtime, "tts_prefer_learned", True))
        playback = bool(getattr(self.config.runtime, "tts_playback", True)) if play is None else bool(play)
        if learned_preferred and self.tts.ready:
            speaker_key = str(speaker or getattr(self.config.runtime, "tts_default_speaker", "") or self._last_speaker_id or "").strip()
            embedding = self.speakers.embedding_for(speaker_key) if speaker_key else None
            if embedding is None and self._last_speaker_id:
                embedding = self.speakers.embedding_for(self._last_speaker_id)
            if embedding is None:
                result = self.voice.speak(text)
                self._record({"event": "speech_output", "chars": len(text), **result})
                return result
            waveform = self.tts.synthesize_from_embedding(text, embedding)
            destination = Path(output).expanduser().resolve() if output else (self.paths.data / "tts_output.wav")
            saved = self.tts.write_wav(waveform, destination)
            result = {"spoken": True, "backend": "learned_tts", "path": str(saved), "speaker": speaker_key or None}
            if playback:
                result["playback"] = self.voice.play_waveform(waveform, sample_rate=self.tts.sample_rate)
            self._record({"event": "speech_output", "chars": len(text), **result})
            return result
        result = self.voice.speak(text)
        self._record({"event": "speech_output", "chars": len(text), **result})
        return result

    def train_tts(self, manifest: str | Path, *, epochs: int = 1, max_examples: int = 0) -> dict:
        result = self.tts.train_manifest(manifest, epochs=epochs, max_examples=max_examples)
        self._record({"event": "tts_training", **result})
        self.save()
        return result

    def synthesize_voice(
        self,
        text: str,
        *,
        speaker_sample: str | Path | None = None,
        speaker: str = "",
        output: str | Path | None = None,
        play: bool = False,
    ) -> dict:
        if not self.tts.ready:
            raise RuntimeError("learned TTS is not ready; train a paired multi-speaker TTS manifest first")
        embedding = None
        if speaker_sample is not None:
            waveform, rate = self.perception.audio.load_wav(speaker_sample)
            embedding = self._speaker_embedding(waveform, sample_rate=rate)
        else:
            speaker_key = str(speaker or self._last_speaker_id or getattr(self.config.runtime, "tts_default_speaker", "")).strip()
            if speaker_key:
                embedding = self.speakers.embedding_for(speaker_key)
            if embedding is None:
                raise ValueError("provide --voice-sample or an enrolled --voice-speaker alias/id")
        waveform = self.tts.synthesize_from_embedding(text, embedding)
        destination = Path(output).expanduser().resolve() if output else (self.paths.data / "tts_output.wav")
        saved = self.tts.write_wav(waveform, destination)
        playback_result = self.voice.play_waveform(waveform, sample_rate=self.tts.sample_rate) if play else None
        result = {
            "ready": True,
            "backend": "learned_tts",
            "path": str(saved),
            "samples": int(waveform.numel()),
            "seconds": float(waveform.numel() / self.tts.sample_rate),
            "speaker": speaker or self._last_speaker_id or None,
            "playback": playback_result,
        }
        self._record({"event": "voice_synthesis", "chars": len(text), **result})
        return result

    def train_speech(self, manifest: str | Path, *, epochs: int = 1, max_examples: int = 0) -> dict:
        result = self.speech.train_manifest(manifest, epochs=epochs, max_examples=max_examples)
        self._record({"event": "speech_training", **result})
        self.save()
        return result

    def transcribe_audio(self, source: torch.Tensor | str | Path) -> str:
        if isinstance(source, (str, Path)):
            waveform, rate = self.perception.audio.load_wav(source)
        else:
            waveform, rate = source.float().reshape(-1).cpu(), self.perception.audio.sample_rate
        return self.speech.transcribe(waveform, sample_rate=rate)

    def _speaker_embedding(self, waveform: torch.Tensor, *, sample_rate: int | None = None) -> torch.Tensor:
        value = waveform.float().reshape(-1).cpu()
        rate = int(sample_rate or self.perception.audio.sample_rate)
        target_rate = int(self.perception.audio.sample_rate)
        if rate != target_rate:
            target = max(1, int(round(value.numel() * target_rate / max(1, rate))))
            value = F.interpolate(value.view(1, 1, -1), size=target, mode="linear", align_corners=False).view(-1)
        with torch.no_grad():
            self.perception.audio.eval()
            return self.perception.audio(value.to(self.device))[0].detach()

    def observe_speaker(self, source: torch.Tensor | str | Path, *, alias: str = "", update: bool = True) -> dict:
        if isinstance(source, (str, Path)):
            waveform, rate = self.perception.audio.load_wav(source)
        else:
            waveform, rate = source.float().reshape(-1).cpu(), self.perception.audio.sample_rate
        embedding = self._speaker_embedding(waveform, sample_rate=rate)
        result = self.speakers.observe(embedding, alias=alias, update=update)
        self._last_speaker_id = result.get("speaker_id")
        self._last_speaker_score = float(result.get("score", 0.0))
        self._record_cognitive_event(
            "speaker_observation", modality="audio", source="microphone",
            entity_id=str(result.get("speaker_id", "")) or None,
            novelty=1.0 if result.get("new") else max(0.0, 1.0 - float(result.get("score", 0.0))),
            salience=float(result.get("score", 0.0)),
            detail={"known": not bool(result.get("new")), "aliases": result.get("aliases", [])},
        )
        self.speakers.flush()
        return result

    def enroll_speaker(self, source: str | Path, alias: str) -> dict:
        waveform, rate = self.perception.audio.load_wav(source)
        embedding = self._speaker_embedding(waveform, sample_rate=rate)
        result = self.speakers.enroll(embedding, alias)
        self._last_speaker_id = result.get("speaker_id")
        self._last_speaker_score = float(result.get("score", 0.0))
        self._record({"event": "speaker_enrollment", **result})
        self.speakers.flush()
        self.save()
        return result

    def identify_speaker(self, source: torch.Tensor | str | Path) -> dict:
        if isinstance(source, (str, Path)):
            waveform, rate = self.perception.audio.load_wav(source)
        else:
            waveform, rate = source.float().reshape(-1).cpu(), self.perception.audio.sample_rate
        embedding = self._speaker_embedding(waveform, sample_rate=rate)
        result = self.speakers.identify(embedding)
        if result.get("speaker_id"):
            self._last_speaker_id = result["speaker_id"]
            self._last_speaker_score = float(result.get("score", 0.0))
        return result

    @staticmethod
    def _parse_dialogue_feedback(text: str) -> tuple[str, str] | None:
        raw = str(text)
        return parse_dialogue_example(raw)

    def _online_dialogue_learn(self, message: str, reply: str) -> dict:
        """Perform a small accepted-turn update with rollback gates.

        The response bytes are the only supervised targets. Recent positively accepted turns
        are mixed in as replay so a single interaction does not immediately overwrite the
        learned language behavior.
        """
        runtime = self.config.runtime
        if not bool(getattr(runtime, "dialogue_online_learning_enabled", True)):
            return {"enabled": False, "accepted": False, "reason": "disabled"}
        if len(str(reply).strip()) < 2 or len(str(message).strip()) < 1:
            return {"enabled": True, "accepted": False, "reason": "too_short"}

        pairs = [(str(message).strip(), str(reply).strip())]
        replay_limit = max(0, int(getattr(runtime, "dialogue_online_replay_examples", 4)))
        if replay_limit:
            for row in self.memory.recent(replay_limit * 3, kinds=("dialogue_feedback",)):
                if not bool(row.get("accepted")):
                    continue
                parsed = self._parse_dialogue_feedback(str(row.get("text", "")))
                if parsed and parsed not in pairs:
                    pairs.append(parsed)
                if len(pairs) >= replay_limit + 1:
                    break

        def mean_response_loss() -> float:
            self.model.eval()
            values: list[float] = []
            with torch.no_grad():
                for prompt, response in pairs:
                    value = self.model.response_loss(
                        format_turn_prompt(prompt), response, device=self.device
                    )
                    if torch.isfinite(value):
                        values.append(float(value.detach().cpu()))
            return float(sum(values) / max(1, len(values)))

        before_model = {name: tensor.detach().cpu().clone() for name, tensor in self.model.state_dict().items()}
        before_optimizer = copy.deepcopy(self.optimizer.state_dict())
        before_mode = self.model.training
        before_loss = mean_response_loss()
        fixed_before = self.evaluate_fixed_reference()
        old_lrs = [float(group["lr"]) for group in self.optimizer.param_groups]
        scale = max(1e-4, float(getattr(runtime, "dialogue_online_learning_rate_scale", 0.15)))
        steps = max(1, int(getattr(runtime, "dialogue_online_steps", 1)))
        last_loss = before_loss
        accepted = False
        try:
            self.model.train()
            for group, base_lr in zip(self.optimizer.param_groups, old_lrs):
                group["lr"] = base_lr * scale
            for _ in range(steps):
                losses = []
                for prompt, response in pairs:
                    losses.append(
                        self.model.response_loss(
                            format_turn_prompt(prompt), response, device=self.device
                        )
                    )
                if not losses:
                    break
                loss = torch.stack(losses).mean()
                if not torch.isfinite(loss):
                    raise FloatingPointError("online dialogue loss became non-finite")
                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), float(runtime.gradient_clip))
                self.optimizer.step()
                last_loss = float(loss.detach().cpu())

            after_loss = mean_response_loss()
            fixed_after = self.evaluate_fixed_reference()
            fixed_gain_delta = 0.0
            if fixed_before.get("available") and fixed_after.get("available"):
                fixed_gain_delta = float(fixed_after["gain"] - fixed_before["gain"])
            response_delta = after_loss - before_loss
            max_fixed_regression = float(getattr(runtime, "dialogue_online_max_fixed_regression", 0.02))
            max_response_regression = float(getattr(runtime, "dialogue_online_max_response_loss_regression", 0.02))
            accepted = bool(
                math.isfinite(after_loss)
                and response_delta <= max_response_regression
                and fixed_gain_delta >= -max_fixed_regression
            )
            if not accepted:
                self.model.load_state_dict(before_model)
                self.optimizer.load_state_dict(before_optimizer)
        except Exception as exc:
            self.model.load_state_dict(before_model)
            self.optimizer.load_state_dict(before_optimizer)
            return {
                "enabled": True, "accepted": False, "reason": "exception", "error": repr(exc),
                "before_loss": before_loss, "after_loss": None, "pairs": len(pairs),
            }
        finally:
            for group, lr in zip(self.optimizer.param_groups, old_lrs):
                group["lr"] = lr
            self.model.train(before_mode)

        result = {
            "enabled": True,
            "accepted": accepted,
            "pairs": len(pairs),
            "steps": steps,
            "before_loss": before_loss,
            "after_loss": after_loss,
            "response_loss_delta": after_loss - before_loss,
            "fixed_gain_delta": fixed_gain_delta,
            "fixed_available": bool(fixed_before.get("available") and fixed_after.get("available")),
            "replayed": max(0, len(pairs) - 1),
        }
        self._last_online_dialogue = result
        self.last_dialogue_accepted = accepted
        self.last_dialogue_fixed_gain_delta = float(fixed_gain_delta)
        self._record({"event": "online_dialogue_learning", **result})
        return result

    def listen_microphone(self, *, seconds: float = 5.0, sample_rate: int | None = None) -> dict:
        try:
            import sounddevice as sd
        except ImportError as exc:
            raise RuntimeError("microphone listening requires sounddevice") from exc
        rate = max(8000, int(sample_rate or getattr(self.config.runtime, "stream_sample_rate", 16000)))
        duration = max(0.25, float(seconds))
        frames = max(256, int(round(duration * rate)))
        recorded = sd.rec(frames, samplerate=rate, channels=1, dtype="float32", blocking=True)
        audio = torch.from_numpy(recorded.reshape(-1))
        session = StreamingSpeechSession(
            self.speech,
            sample_rate=rate,
            window_seconds=float(getattr(self.config.runtime, "stream_window_seconds", 3.0)),
            update_seconds=float(getattr(self.config.runtime, "stream_update_seconds", 0.75)),
            vad_threshold=float(getattr(self.config.runtime, "stream_vad_threshold", 0.02)),
            end_silence_chunks=int(getattr(self.config.runtime, "stream_end_silence_chunks", 4)),
            min_speech_seconds=float(getattr(self.config.runtime, "stream_min_speech_seconds", 0.25)),
            max_utterance_seconds=float(getattr(self.config.runtime, "stream_max_utterance_seconds", 20.0)),
        )
        update = session.push(audio, sample_rate=rate)
        if not update.get("final"):
            update = session.flush()
        transcript = str(update.get("final") or update.get("partial") or update.get("stable") or "")
        speaker = self.observe_speaker(update.get("audio") if update.get("audio") is not None else audio, update=False) if transcript else {"speaker_id": None, "known": False}
        reply = self.communicate(transcript, speaker_id=speaker.get("speaker_id"), source="microphone") if transcript else ""
        return {"transcript": transcript, "speaker": speaker, "reply": reply, "rms": update.get("rms", 0.0)}

    def initiate_communication(self, trigger: str, *, speak: bool | None = None) -> dict:
        """Generate an unsolicited conversational turn from internal/sensory context.

        The trigger supplies observations, not a canned response. The language model decides
        whether there is anything worth saying from its learned state and memory.
        """
        if not self._chat_generation_ready():
            return {"initiated": False, "reason": "language_not_ready"}
        prompt = self._prompt_with_context(
            "Internal observation requiring no direct user reply:\n" + " ".join(str(trigger).split()) + "\nTurn B:",
            self._retrieve_experience(trigger, limit=max(1, int(self.config.runtime.chat_memory_results // 2))),
        )
        generated, new_state, metrics = self.model.generate_candidates(
            prompt,
            candidate_count=max(1, int(self.config.runtime.chat_candidate_count)),
            max_new_tokens=int(self.config.runtime.chat_max_response_tokens),
            temperature=self.config.runtime.chat_temperature,
            top_k=self.config.runtime.chat_top_k,
            device=self.device,
            state=self.state,
        )
        reply = self._clean_generated_reply(generated)
        if not self._generated_text_is_usable(reply, trigger):
            return {"initiated": False, "reason": "no_usable_generation", "generation_metrics": metrics}
        self.state = new_state.detach()
        memory_id = self.memory.add(
            kind="conversation_initiative", source="internal", title="Aletheia initiated communication", url="",
            text=reply, accepted=True, replay_eligible=True, merge_existing=False,
        )
        self._last_feedback_target = {
            "kind": "chat", "message": trigger, "reply": reply, "memory_id": memory_id, "created": time.time(),
        }
        should_speak = bool(self.config.runtime.initiative_speak if speak is None else speak)
        speech = self.speak(reply) if should_speak else {"spoken": False, "reason": "speech_disabled"}
        self._last_initiative_time = time.monotonic()
        self._record({"event": "initiative", "trigger": trigger[:1000], "reply_chars": len(reply), "spoken": speech.get("spoken", False), "generation_metrics": metrics})
        self.save()
        return {"initiated": True, "reply": reply, "spoken": speech, "memory_id": memory_id, "generation_metrics": metrics}

    def run_companion(self, *, camera_index: int = 0, max_seconds: int | None = None, show: bool = False) -> None:
        """Run a local multimodal communication loop with incremental ASR and speaker memory."""
        try:
            import cv2
        except ImportError as exc:
            raise RuntimeError("companion mode requires opencv-python") from exc
        try:
            import sounddevice as sd
        except ImportError as exc:
            raise RuntimeError("companion mode requires sounddevice") from exc

        capture = cv2.VideoCapture(int(camera_index))
        if not capture.isOpened():
            raise RuntimeError(f"unable to open camera device {camera_index}")

        camera_interval = max(0.05, float(self.config.runtime.presence_camera_interval))
        audio_seconds = max(0.15, min(1.0, float(self.config.runtime.presence_audio_seconds)))
        sample_rate = max(8000, int(getattr(self.config.runtime, "stream_sample_rate", self.config.runtime.presence_audio_sample_rate)))
        max_seconds_value = max(0, int(max_seconds if max_seconds is not None else self.config.runtime.presence_max_seconds))
        session = StreamingSpeechSession(
            self.speech,
            sample_rate=sample_rate,
            window_seconds=float(getattr(self.config.runtime, "stream_window_seconds", 3.0)),
            update_seconds=float(getattr(self.config.runtime, "stream_update_seconds", 0.75)),
            vad_threshold=float(getattr(self.config.runtime, "stream_vad_threshold", self.config.runtime.presence_audio_activity_threshold)),
            end_silence_chunks=int(getattr(self.config.runtime, "stream_end_silence_chunks", 4)),
            min_speech_seconds=float(getattr(self.config.runtime, "stream_min_speech_seconds", 0.25)),
            max_utterance_seconds=float(getattr(self.config.runtime, "stream_max_utterance_seconds", 20.0)),
        )
        start = time.monotonic()
        last_frame = start - camera_interval
        last_audio = start - audio_seconds
        print("Aletheia companion mode started. Press Ctrl+C to stop.")
        try:
            while True:
                now = time.monotonic()
                if max_seconds_value and now - start >= max_seconds_value:
                    break
                visual_novelty = 0.0
                audio_activity = 0.0
                audio_partial = ""
                audio_final = ""
                current_visual_embedding = None
                current_audio_embedding = None
                current_visual_entity: str | None = None
                spoke_for_utterance = False

                if now - last_frame >= camera_interval:
                    ok, frame = capture.read()
                    if ok:
                        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        tensor = self.perception.vision.prepare_image(rgb).to(self.device)
                        visual_result = self.perception.observe_image(rgb, alias="", train=True)
                        current_visual_entity = str(visual_result.get("entity_id", "")) or None
                        with torch.no_grad():
                            embedding = (self.perception.last_scene_embedding if self.perception.last_scene_embedding is not None
                                         else self.perception.vision(tensor)[0]).detach()
                        if self._last_visual_embedding is not None:
                            visual_novelty = max(0.0, 1.0 - float(torch.sum(embedding * self._last_visual_embedding).item()))
                        self._last_visual_embedding = embedding
                        current_visual_embedding = embedding
                        should_record_visual = (
                            self._last_recorded_visual_time == 0.0
                            or visual_novelty >= float(getattr(self.config.runtime, "perception_min_visual_delta", 0.02))
                            or (now - self._last_recorded_visual_time) >= max(5.0, camera_interval * 10.0)
                        )
                        if should_record_visual:
                            self.memory.add(
                                kind="sensory_vision", source="camera", title="companion visual observation", url="",
                                text=f"visual novelty={visual_novelty:.4f} entity={current_visual_entity or 'none'}", merge_existing=False,
                            )
                            self._record_cognitive_event(
                                "visual_stream", modality="vision", source="camera", entity_id=current_visual_entity,
                                novelty=visual_novelty, salience=visual_novelty,
                                detail={"companion": True, "object_count": int(visual_result.get("object_count", 0)),
                                        "objects": visual_result.get("objects", [])},
                            )
                            self._last_recorded_visual_time = now
                        if show:
                            cv2.imshow("Aletheia", frame)
                            if cv2.waitKey(1) & 0xFF == ord("q"):
                                break
                        last_frame = now

                if now - last_audio >= audio_seconds:
                    try:
                        frames = max(256, int(sample_rate * audio_seconds))
                        recorded = sd.rec(frames, samplerate=sample_rate, channels=1, dtype="float32", blocking=True)
                        audio = torch.from_numpy(recorded.reshape(-1))
                        update = session.push(audio, sample_rate=sample_rate)
                        audio_activity = float(update.get("rms", 0.0))
                        audio_partial = str(update.get("partial") or update.get("stable") or "")
                        audio_final = str(update.get("final") or "")
                        self.perception.observe_audio(audio, train=True)
                        with torch.no_grad():
                            current_audio_embedding = self.perception.audio(audio.to(self.device))[0].detach()

                        if audio_activity >= float(getattr(self.config.runtime, "stream_vad_threshold", self.config.runtime.presence_audio_activity_threshold)):
                            detail = {"partial_chars": len(audio_partial), "final_chars": len(audio_final)}
                            self._record_cognitive_event(
                                "audio_stream", modality="audio", source="microphone",
                                novelty=min(1.0, audio_activity * 10.0), salience=min(1.0, audio_activity * 10.0),
                                detail=detail,
                            )

                        if audio_partial:
                            # Only telemetry gets the evolving transcript. Persistent memory is
                            # written once the utterance boundary is detected.
                            self._record({"event": "stream_transcript", "partial": audio_partial[-500:]})

                        final_audio = update.get("audio")
                        if audio_final and isinstance(final_audio, torch.Tensor):
                            speaker = self.observe_speaker(final_audio, update=True)
                            speaker_id = str(speaker.get("speaker_id", "")) or None
                            if speaker_id and current_visual_entity:
                                self.speakers.link(speaker_id, current_visual_entity)
                            self.memory.add(
                                kind="sensory_audio", source="microphone", title="spoken utterance", url="",
                                text=(
                                    f"speaker_id={speaker_id or 'unknown'} speaker_score={float(speaker.get('score', 0.0)):.4f} "
                                    f"transcript={audio_final[:2000]}"
                                ), merge_existing=False,
                            )
                            self._record_cognitive_event(
                                "spoken_utterance", modality="audio", source="microphone", entity_id=speaker_id,
                                novelty=1.0, salience=1.0,
                                detail={"transcript_chars": len(audio_final), "speaker_score": float(speaker.get("score", 0.0)), "visual_entity": current_visual_entity},
                            )
                            if self.speech.ready and audio_final.strip():
                                raw_reply = self.communicate(audio_final, speaker_id=speaker_id, source="microphone")
                                clean_reply = self._strip_response_tag(raw_reply)
                                if raw_reply.startswith("[generated]") or raw_reply.startswith("[quoted]"):
                                    spoken = self.speak(clean_reply)
                                    print(f"ALETHEIA> {clean_reply}")
                                    spoke_for_utterance = bool(spoken.get("spoken"))
                                elif raw_reply:
                                    print(f"ALETHEIA> {clean_reply}")
                                self._last_initiative_time = time.monotonic()

                    except Exception as exc:
                        self._record({"event": "companion_audio_error", "error": repr(exc)})
                    last_audio = now

                if current_visual_embedding is not None or current_audio_embedding is not None:
                    with torch.no_grad():
                        self.state, attention = self.model.integrate_multimodal(
                            self.state,
                            vision_embedding=current_visual_embedding,
                            audio_embedding=current_audio_embedding,
                        )
                    attention_values = attention.detach().float().cpu().tolist() if attention.numel() else []
                    self._record_cognitive_event(
                        "multimodal_fusion", modality="multimodal", source="companion",
                        novelty=max(visual_novelty, min(1.0, audio_activity * 10.0)),
                        salience=max(visual_novelty, min(1.0, audio_activity * 10.0)),
                        detail={"attention": attention_values, "transcript": audio_final[-500:] if audio_final else audio_partial[-500:]},
                    )

                combined = (
                    float(self.config.runtime.presence_visual_weight) * visual_novelty
                    + float(self.config.runtime.presence_audio_weight) * min(1.0, audio_activity * 10.0)
                )
                cooldown = max(1.0, float(self.config.runtime.presence_initiative_cooldown_seconds))
                threshold = max(0.0, float(self.config.runtime.initiative_min_visual_novelty))
                if (
                    not spoke_for_utterance
                    and not audio_final
                    and combined >= threshold
                    and time.monotonic() - self._last_initiative_time >= cooldown
                ):
                    trigger = (
                        f"sensory state changed; visual_novelty={visual_novelty:.4f}; "
                        f"audio_activity={audio_activity:.4f}; combined_novelty={combined:.4f}; "
                        f"partial_transcript={audio_partial[:500]}; elapsed_seconds={time.monotonic() - start:.1f}"
                    )
                    initiative = self.initiate_communication(trigger)
                    if initiative.get("initiated"):
                        print(f"ALETHEIA> {initiative['reply']}")
                time.sleep(min(0.15, camera_interval))
        except KeyboardInterrupt:
            print("\nAletheia companion mode stopped by operator.")
        finally:
            try:
                tail = session.flush()
                if tail.get("final"):
                    self._record({"event": "stream_transcript_final", "text": tail["final"][-1000:]})
            except Exception:
                pass
            capture.release()
            if show:
                cv2.destroyAllWindows()
            self.perception.flush()
            self.speakers.flush()
            self.save()

    def _cap_pursue_objective(self) -> tuple[float, dict]:
        """Delegate an objective attempt to a learned/planned action rather than hard-wiring search."""
        objective = self.objectives.current()
        if objective is None:
            return 0.0, {"kind": "objective", "valid_transition": False, "reason": "no_active_objective"}
        self._ensure_objective_plan(objective)
        active_goal = self.planner.hierarchy.text(objective.id, objective.text)
        before_progress = float(objective.progress)
        try:
            delegated_index, mode, entropy, _ = self._choose_action(exclude_names={"pursue_objective"})
            capability = self.registry.get(delegated_index)
            signal, delegated = capability.run()
        except Exception as exc:
            objective.attempts += 1
            self.objectives.update(objective.id, attempts=objective.attempts, last_action="objective_dispatch_error")
            return 0.0, {
                "kind": "objective", "valid_transition": False, "reason": "dispatch_error", "error": repr(exc),
                "objective": objective.text, "active_subgoal": active_goal,
            }
        objective.attempts += 1
        objective.last_action = capability.name
        known_after = self._workspace(32, 1500)
        objective.knowledge_gaps = self.objectives.infer_gaps(objective, known_after, limit=12)
        self.objectives.update(
            objective.id, attempts=objective.attempts, last_action=objective.last_action,
            knowledge_gaps=objective.knowledge_gaps, notes=objective.notes[-40:],
        )
        result = dict(delegated or {})
        result.update({
            "kind": "objective",
            "objective": objective.text,
            "active_subgoal": active_goal,
            "delegated_action": capability.name,
            "delegated_selection_mode": mode,
            "delegated_entropy": entropy,
            "subgoals": self.planner.hierarchy.status(objective.id),
            "gaps": objective.knowledge_gaps,
            "evidence": float(max(0.0, min(1.0, float(signal)))),
            "objective_progress_before": before_progress,
            "objective_progress_after": float(objective.progress),
            "objective_progress_source": objective.progress_source,
        })
        return float(signal), result

    @staticmethod
    def _strip_response_tag(reply: str) -> str:
        return re.sub(r"^\s*\[(?:generated|experimental|quoted|not_ready|evidence|no_answer)\]\s*", "", reply or "").strip()

    def record_objective_outcome(
        self,
        outcome: float,
        *,
        source: str = "environment",
        note: str = "",
        persist: bool = True,
        verified: bool = False,
        evidence: dict | None = None,
    ) -> dict:
        """Record a generic observed outcome for the most recent objective transition.

        The caller supplies only an observed score in [0, 1]. No environment-specific
        meaning is encoded here; environments and human feedback may use the same interface.
        """
        objective = self.objectives.current()
        if objective is None:
            return {"updated": False, "reason": "no_active_objective"}
        value = float(outcome)
        if not math.isfinite(value):
            raise ValueError("objective outcome must be finite")
        value = max(0.0, min(1.0, value))
        if self._last_action_state is None or self._last_action_state_after is None or self._last_action_index is None:
            return {"updated": False, "reason": "no_transition_context", "objective_id": objective.id}

        if not verified:
            # Similarity/prediction/observation-quality scores are candidates, not labels
            # for task success. Keep them in telemetry, but never train the learned
            # outcome estimator or advance objective progress from an unverified score.
            if note.strip():
                objective.notes.append(f"unverified_outcome_candidate[{source}]: {note.strip()}")
            self.objectives.update(objective.id, notes=objective.notes[-40:])
            result = {
                "updated": False,
                "reason": "outcome_not_independently_verified",
                "objective_id": objective.id,
                "candidate_outcome": value,
                "source": str(source),
                "progress": objective.progress,
                "outcome_count": objective.outcome_count,
                "verified": False,
                "value_learning": {"trained": False, "reason": "unverified_outcome"},
            }
            self._record({"event": "objective_outcome_candidate", **result})
            if persist:
                self.save()
            return result

        self._ensure_objective_plan(objective)
        planning_goal = self.planner.hierarchy.text(objective.id, objective.text)
        goal_vector = self.planner._goal_vector(planning_goal).detach()
        action_embedding = self.model.action_embedding(
            torch.tensor([int(self._last_action_index)], dtype=torch.long, device=self.device)
        ).detach()
        self.objective_value.record(
            self._last_action_state,
            self._last_action_state_after,
            goal_vector,
            action_embedding,
            value,
            objective_id=objective.id,
            source=str(source),
            persist=bool(persist),
        )
        value_train = self.objective_value.fit(
            int(getattr(self.config.runtime, "objective_value_training_steps", 2))
        )
        local_outcomes = self.objective_value.outcome_stats(objective.id)
        estimated, source_name = self.objective_value.estimate(
            self._last_action_state,
            self._last_action_state_after,
            goal_vector,
            action_embedding,
            fallback=float(local_outcomes["mean"]),
        )
        if estimated is not None and math.isfinite(float(estimated)):
            objective.progress = max(0.0, min(0.99, float(estimated)))
            objective.progress_source = source_name
        objective.outcome_count = int(local_outcomes["count"])
        objective.outcome_mean = float(local_outcomes["mean"])
        if note.strip():
            objective.notes.append(f"outcome[{source}]: {note.strip()}")
        self.objectives.update(
            objective.id,
            progress=objective.progress,
            outcome_count=objective.outcome_count,
            outcome_mean=objective.outcome_mean,
            progress_source=objective.progress_source,
            notes=objective.notes[-40:],
        )
        verification = self.objectives.verify(
            objective.id,
            outcome=value,
            human_confirmed=(str(source) == "human_feedback" and value >= 0.5),
            evidence=evidence or {},
        )
        result = {
            "updated": True,
            "objective_id": objective.id,
            "outcome": value,
            "source": str(source),
            "progress": objective.progress,
            "progress_source": objective.progress_source,
            "outcome_count": objective.outcome_count,
            "outcome_mean": objective.outcome_mean,
            "value_learning": value_train,
            "verified": bool(verified),
            "verification": verification,
        }
        self._record({"event": "objective_outcome", **result})
        if persist:
            self.save()
        return result

    def objective_feedback(self, positive: bool, note: str = "") -> dict:
        objective = self.objectives.current()
        target = dict(self._last_feedback_target or {})
        kind = target.get("kind", "none")
        if objective is None and kind == "none":
            return {"updated": False, "reason": "no_feedback_target"}

        planning_goal = None
        planner_learning = {"accepted": False, "reason": "not_applicable"}
        hierarchy_feedback = {"updated": False, "reason": "no_objective"}
        target_label = (
            target.get("action_name")
            if kind == "autonomous"
            else "chat_response"
            if kind == "chat"
            else "none"
        )
        value_train = {"trained": False, "reason": "no_autonomous_transition"}

        if objective is not None:
            self._ensure_objective_plan(objective)
            planning_goal = self.planner.hierarchy.text(objective.id, objective.text)
            feedback_note = ("positive" if positive else "negative") + f" [{target_label}]"
            if note.strip():
                feedback_note += f": {note.strip()}"
            objective.notes.append(feedback_note)
            objective.last_action = "human_feedback"
            self.objectives.update(
                objective.id,
                progress=objective.progress,
                last_action=objective.last_action,
                notes=objective.notes[-40:],
            )
            if kind == "autonomous":
                outcome_result = self.record_objective_outcome(
                    1.0 if positive else 0.0,
                    source="human_feedback",
                    verified=True,
                    evidence={"note": note.strip()},
                )
                value_train = outcome_result.get(
                    "value_learning", {"trained": False, "reason": "no_autonomous_transition"}
                )
            if kind == "autonomous":
                hierarchy_feedback = self.planner.hierarchy.feedback(objective.id, positive)

        policy_updated = False
        if kind == "autonomous" and self._last_action_state is not None and self._last_action_index is not None:
            try:
                self._learn_policy(
                    self._last_action_state,
                    self._last_action_index,
                    1.0 if positive else -1.0,
                    True,
                    self._last_action_selection_mode or "policy",
                    self._last_policy_mask,
                )
                policy_updated = True
            except Exception:
                policy_updated = False
            if planning_goal:
                try:
                    planner_learning = self.planner.learn_feedback(
                        self._last_action_state,
                        planning_goal,
                        self._last_action_index,
                        target=1.0 if positive else 0.0,
                        steps=int(getattr(self.config.runtime, "planning_feedback_steps", 1)),
                    )
                except Exception as exc:
                    planner_learning = {"accepted": False, "reason": "planner_learning_error", "error": repr(exc)}
        elif kind == "chat":
            message = str(target.get("message", ""))
            reply = self._strip_response_tag(str(target.get("reply", "")))
            if message and reply:
                self.memory.add(
                    kind="dialogue_feedback", source="human_feedback",
                    title="positive chat feedback" if positive else "rejected chat feedback", url="",
                    text=f"Turn A: {message}\nTurn B: {reply}",
                    accepted=bool(positive), replay_eligible=bool(positive), merge_existing=False,
                )
                if positive:
                    online_learning = self._online_dialogue_learn(message, reply)
                else:
                    online_learning = {
                        "enabled": bool(getattr(self.config.runtime, "dialogue_online_learning_enabled", True)),
                        "accepted": False,
                        "reason": "negative_feedback",
                    }
            else:
                online_learning = {
                    "enabled": bool(getattr(self.config.runtime, "dialogue_online_learning_enabled", True)),
                    "accepted": False,
                    "reason": "missing_turn",
                }
        else:
            online_learning = {"enabled": False, "accepted": False, "reason": "no_chat_target"}

        if kind != "chat":
            online_learning = {"enabled": False, "accepted": False, "reason": "not_chat"}

        self._record({
            "event": "objective_feedback",
            "objective_id": objective.id if objective is not None else None,
            "positive": positive,
            "note": note,
            "target_kind": kind,
            "target_label": target_label,
            "objective_progress_source": objective.progress_source if objective is not None else "unobserved",
            "objective_outcome_count": objective.outcome_count if objective is not None else 0,
            "objective_outcome_mean": objective.outcome_mean if objective is not None else 0.0,
            "value_learning": value_train if objective is not None else {"trained": False, "reason": "no_objective"},
            "policy_updated": policy_updated,
            "online_learning": online_learning,
            "planner_learning": planner_learning,
            "planning_goal": planning_goal,
            "hierarchy_feedback": hierarchy_feedback,
        })
        self.save()
        return {
            "updated": True,
            "positive": positive,
            "progress": objective.progress if objective is not None else None,
            "objective_id": objective.id if objective is not None else None,
            "target_kind": kind,
            "target_label": target_label,
            "objective_progress_source": objective.progress_source if objective is not None else "unobserved",
            "objective_outcome_count": objective.outcome_count if objective is not None else 0,
            "objective_outcome_mean": objective.outcome_mean if objective is not None else 0.0,
            "value_learning": value_train if objective is not None else {"trained": False, "reason": "no_objective"},
            "policy_updated": policy_updated,
            "online_learning": online_learning,
            "planner_learning": planner_learning,
            "planning_goal": planning_goal,
            "hierarchy_feedback": hierarchy_feedback,
        }

    def evolve_source(self, relative_path: str, instruction: str) -> dict:
        result = self.evolution.propose_source_change(self, relative_path, instruction)
        self._record({"event": "source_evolution", **result})
        if result.get("accepted"):
            self.save()
        return result

    def evolve_generation(self, *, population_size: int | None = None, auto_promote: bool | None = None) -> dict:
        """Clone the complete learned state, benchmark a candidate population, and select a successor."""
        result = self.generational_evolution.run_generation(
            self, population_size=population_size, auto_promote=auto_promote
        )
        self._record({
            "event": "generation_evolution",
            "generation": result.get("generation"),
            "promoted": result.get("promoted", False),
            "winner": (result.get("winner") or {}).get("candidate_id") if isinstance(result.get("winner"), dict) else None,
            "restart_required": result.get("restart_required", False),
        })
        return result

    def evolution_status(self) -> dict:
        current = self.generational_evolution.current_generation()
        return {
            "current": current,
            "lineage": self.generational_evolution.lineage()[-32:],
        }

    def _cap_evolve(self) -> tuple[float, dict]:
        objective = self.objectives.current()
        # This path currently scores only language BPB. Do not use that metric to
        # tune policy/perception/audio/speech/TTS/planning learning rates; each needs
        # its own independently held-out task metric before it can be optimized here.
        options = ["learning_rate"] if hasattr(self.config.runtime, "learning_rate") else []
        if not options:
            return 0.0, {"kind": "evolution", "valid_transition": False, "reason": "no_runtime_mutations_configured"}
        parameter = random.choice(options)
        try:
            result = self.evolution.evolve_runtime_parameter(self, parameter)
            signal = 1.0 if result.accepted else 0.0
            if objective is not None and result.accepted:
                self.objectives.update(objective.id, progress=objective.progress, last_action=f"evolve:{parameter}")
            return signal, {"kind": "evolution", **asdict(result), "valid_transition": True}
        except Exception as exc:
            return 0.0, {"kind": "evolution", "error": repr(exc), "valid_transition": False}

    def _objective_action_index(self, action_name: str) -> int | None:
        if self._last_action_index is not None:
            try:
                if self.registry.items()[int(self._last_action_index)].name == str(action_name):
                    return int(self._last_action_index)
            except Exception:
                pass
        try:
            return int(self.registry.index(str(action_name)))
        except Exception:
            return None

    def _predict_environment_outcome_embedding(
        self, action_name: str, state_before: torch.Tensor | None = None
    ) -> torch.Tensor | None:
        """Predict the next learned observation before an external action happens.

        This must be computed before web/browser ingestion can update Aletheia's
        parameters; otherwise prediction verification would compare an observation
        against a prediction produced by post-update weights.
        """
        state = state_before if state_before is not None else self.state
        action_index = self._objective_action_index(action_name)
        if action_index is None:
            return None
        try:
            with torch.no_grad():
                return self.model.predict_next_observation(
                    state.detach(), int(action_index), float(self.prediction_error)
                ).detach()
        except Exception:
            return None

    def _open_world_verification(
        self,
        *,
        action_name: str,
        observed_text: str,
        query: str = "",
        prior_rows: list[dict] | None = None,
        exclude_url: str = "",
        state_before: torch.Tensor | None = None,
        include_self_qa: bool = False,
        predicted_observation: torch.Tensor | None = None,
        observed_embedding: torch.Tensor | None = None,
        prior_embeddings: list[torch.Tensor] | None = None,
    ) -> tuple[float | None, dict]:
        if not bool(getattr(self.config.runtime, "open_world_enabled", True)):
            return None, {}
        text = self._exclude_evaluation_text(str(observed_text or "").strip()).strip()
        if len(text) < int(getattr(self.config.runtime, "open_world_min_source_chars", 96)):
            return None, {}
        if state_before is None:
            state_before = self._last_action_state.detach().clone() if self._last_action_state is not None else self.state.detach().clone()
        prediction_match = None
        actual_vec = observed_embedding
        with torch.no_grad():
            if actual_vec is None:
                actual_vec = self._text_representation(text[: int(getattr(self.config.runtime, "open_world_max_prediction_chars", 4000))])
            predicted = predicted_observation
            if predicted is None:
                predicted = self._predict_environment_outcome_embedding(action_name, state_before)
            if predicted is not None and actual_vec is not None:
                prediction_match = self.open_world_outcomes.similarity(predicted, actual_vec)

        candidates: list[torch.Tensor] = list(prior_embeddings or [])
        rows = prior_rows
        if rows is None and query:
            try:
                rows = self.memory.search_chunks(query, max(3, int(getattr(self.config.runtime, "open_world_consistency_sources", 3)) * 4))
            except Exception:
                rows = []
        if not candidates:
            for row in (rows or []):
                url = str(row.get("url", ""))
                if exclude_url and url and url == exclude_url:
                    continue
                row_text = self._exclude_evaluation_text(str(row.get("text", ""))).strip()[: int(getattr(self.config.runtime, "open_world_max_prediction_chars", 4000))]
                if len(row_text.strip()) < 32:
                    continue
                try:
                    candidates.append(self._text_representation(row_text))
                except Exception:
                    continue
                if len(candidates) >= max(1, int(getattr(self.config.runtime, "open_world_consistency_sources", 3))):
                    break
        cross_source = self.open_world_outcomes.mean_consistency(actual_vec, candidates) if candidates else None

        self_qa = None
        if include_self_qa and bool(getattr(self.config.runtime, "open_world_self_qa_enabled", False)) and self._language_ready() and int(getattr(self.config.runtime, "open_world_self_qa_examples", 1)) > 0:
            excerpt = text[:2000]
            try:
                question = self.model.generate(
                    "Generate one short factual question answerable only from this material.\nSOURCE:\n" + excerpt + "\nQUESTION:",
                    max_new_tokens=72, temperature=0.6, top_k=32, device=self.device, state=state_before,
                ).strip()
                question = re.sub(r"^Question:\s*", "", question, flags=re.I).strip()
                if question:
                    answer = self.model.generate(
                        "Answer this question from learned memory only. Do not quote the source.\nQUESTION:\n" + question + "\nANSWER:",
                        max_new_tokens=96, temperature=0.6, top_k=32, device=self.device, state=self.state,
                    ).strip()
                    if answer:
                        answer_vec = self._text_representation(answer[:1600])
                        self_qa = self.open_world_outcomes.similarity(answer_vec, actual_vec)
            except Exception:
                self_qa = None
        signals = OpenWorldSignals(
            prediction_match=prediction_match,
            cross_source_consistency=cross_source,
            reconstruction_match=None,
            self_qa_match=self_qa,
        )
        score = signals.aggregate()
        payload = {"open_world_signals": signals.as_dict(), "observation_quality": score, "verified": False}
        return score, payload

    def _attach_open_world_verification(self, result: dict, *, action_name: str, observed_text: str, query: str = "", prior_rows: list[dict] | None = None, exclude_url: str = "", state_before: torch.Tensor | None = None, include_self_qa: bool = True, predicted_observation: torch.Tensor | None = None, observed_embedding: torch.Tensor | None = None, prior_embeddings: list[torch.Tensor] | None = None) -> tuple[float | None, dict]:
        score, payload = self._open_world_verification(
            action_name=action_name, observed_text=observed_text, query=query,
            prior_rows=prior_rows, exclude_url=exclude_url, state_before=state_before,
            include_self_qa=include_self_qa, predicted_observation=predicted_observation,
            observed_embedding=observed_embedding, prior_embeddings=prior_embeddings,
        )
        if score is not None:
            result.update(payload)
            result["observation_quality"] = float(score)
            result["objective_outcome_generated"] = False
        elif payload:
            result.update(payload)
        return score, result

    @staticmethod
    def _text_duplicate_score(left: str, right: str) -> float:
        def shingles(text: str) -> set[str]:
            tokens = re.findall(r"\w+", re.sub(r"\s+", " ", str(text)).casefold())
            return {" ".join(tokens[i:i+5]) for i in range(max(0, len(tokens) - 4))}
        a, b = shingles(left), shingles(right)
        if not a or not b:
            return 0.0
        return len(a & b) / max(1, len(a | b))

    def _near_duplicate_external(self, text: str) -> tuple[bool, float, str]:
        threshold = float(getattr(self.config.runtime, "open_world_duplicate_similarity_threshold", 0.98))
        safe_incoming = self._exclude_evaluation_text(str(text or "")).strip()
        if len(safe_incoming) < 32:
            return False, 0.0, ""
        for row in self.memory.recent(100, kinds=("external",)):
            existing = self._exclude_evaluation_text(str(row.get("text", ""))).strip()
            if len(existing) < 32:
                continue
            score = self._text_duplicate_score(safe_incoming[:12000], existing[:12000])
            if score >= threshold:
                return True, score, str(row.get("url", ""))
        return False, 0.0, ""

    def _cap_discover(self) -> tuple[float, dict]:
        state_before = self.state.detach().clone()
        predicted = self._predict_environment_outcome_embedding("discover", state_before)
        page = self.web.discover()
        safe_page_text = self._exclude_evaluation_text(page.text).strip()
        if len(safe_page_text) < 32:
            return 0.0, {"kind": "evaluation_overlap_filtered", "valid_transition": False}
        observed = self._text_representation(safe_page_text[: int(getattr(self.config.runtime, "open_world_max_prediction_chars", 4000))])
        result = self._ingest(page, source="discovery", action_name="discover")
        score, result = self._attach_open_world_verification(
            result, action_name="discover", observed_text=page.text, exclude_url=page.url,
            state_before=state_before, predicted_observation=predicted, observed_embedding=observed,
        )
        return float(score if score is not None else 0.0), result

    def _cap_search(self, query_override: str | None = None) -> tuple[float, dict]:
        query = query_override or self._query()
        if len(query) < 3:
            return 0.0, {"kind": "empty_query", "valid_transition": False, "query": query}
        state_before = self._last_action_state.detach().clone() if self._last_action_state is not None else self.state.detach().clone()
        predicted = self._predict_environment_outcome_embedding("search", state_before)
        links = self.web.search(query, self.config.runtime.search_result_limit)
        self.memory.add_links(links, "search://" + query, depth=0, priority=0.0)
        if not links:
            return 0.0, {"kind": "search_empty", "valid_transition": False, "query": query, "results": 0, "errors": list(self.web.last_errors)}
        try:
            prior_rows = self.memory.search_chunks(query, max(3, int(getattr(self.config.runtime, "open_world_consistency_sources", 3)) * 4))
            prior_embeddings = []
            for row in prior_rows:
                safe_text = self._exclude_evaluation_text(str(row.get("text", ""))).strip()
                if len(safe_text) >= 32 and str(row.get("url", "")):
                    prior_embeddings.append(self._text_representation(safe_text[: int(getattr(self.config.runtime, "open_world_max_prediction_chars", 4000))]))
        except Exception:
            prior_rows, prior_embeddings = [], []
        for candidate in links:
            if self.memory.is_blocked(candidate):
                continue
            try:
                page = self.web.open(candidate)
            except WebAccessError as exc:
                self.memory.block_url(exc.url, exc.reason, source="search")
                continue
            duplicate, duplicate_score, duplicate_url = self._near_duplicate_external(page.text)
            safe_page_text = self._exclude_evaluation_text(page.text).strip()
            if len(safe_page_text) < 32:
                return 0.0, {"kind": "evaluation_overlap_filtered", "valid_transition": False, "query": query}
            observed = self._text_representation(safe_page_text[: int(getattr(self.config.runtime, "open_world_max_prediction_chars", 4000))])
            if duplicate:
                duplicate_memory_id = self.memory.add(
                    kind="external_duplicate", source="search", title=page.title, url=page.url, text=page.text,
                    accepted=False, replay_eligible=False, merge_existing=False, action_name="search_duplicate",
                )
                return 0.0, {
                    "kind": "search_duplicate", "valid_transition": False, "query": query,
                    "results": len(links), "errors": list(self.web.last_errors),
                    "duplicate": True, "duplicate_score": duplicate_score, "duplicate_of": duplicate_url,
                    "duplicate_memory_id": duplicate_memory_id, "observation_quality": 0.0,
                }
            result = self._ingest(page, source="search", action_name="search")
            result.update({"kind": "search", "query": query, "results": len(links), "errors": list(self.web.last_errors)})
            score, result = self._attach_open_world_verification(
                result, action_name="search", observed_text=page.text, query=query, prior_rows=prior_rows,
                exclude_url=page.url, state_before=state_before, include_self_qa=False,
                predicted_observation=predicted, observed_embedding=observed, prior_embeddings=prior_embeddings,
            )
            return float(score if score is not None else 0.0), result
        return 0.0, {"kind": "search_blocked", "valid_transition": False, "query": query, "results": len(links), "errors": list(self.web.last_errors)}

    def _cap_follow(self) -> tuple[float, dict]:
        candidate = self.memory.next_link()
        if not candidate:
            return 0.0, {"kind": "frontier_empty", "valid_transition": False}
        self.memory.mark_visit(candidate["url"])
        try:
            prior_rows = self.memory.search_chunks(str(candidate.get("parent_url", "")), max(3, int(getattr(self.config.runtime, "open_world_consistency_sources", 3))))
        except Exception:
            prior_rows = []
        state_before = self._last_action_state.detach().clone() if self._last_action_state is not None else self.state.detach().clone()
        predicted = self._predict_environment_outcome_embedding("follow", state_before)
        try:
            page = self.web.open(candidate["url"])
        except WebAccessError as exc:
            self.memory.block_url(exc.url, exc.reason, source="frontier")
            return 0.0, {"kind": "access_blocked", "valid_transition": False, "url": exc.url, "reason": exc.reason}
        safe_page_text = self._exclude_evaluation_text(page.text).strip()
        if len(safe_page_text) < 32:
            return 0.0, {"kind": "evaluation_overlap_filtered", "valid_transition": False, "url": page.url}
        observed = self._text_representation(safe_page_text[: int(getattr(self.config.runtime, "open_world_max_prediction_chars", 4000))])
        result = self._ingest(page, source="frontier", action_name="follow")
        result["parent_url"] = candidate["parent_url"]
        score, result = self._attach_open_world_verification(result, action_name="follow", observed_text=page.text, prior_rows=prior_rows, exclude_url=page.url, state_before=state_before, predicted_observation=predicted, observed_embedding=observed)
        return float(score if score is not None else 0.0), result

    def _cap_revisit(self) -> tuple[float, dict]:
        rows = [row for row in self.memory.recent(40, kinds=("external",)) if row.get("url") and not self.memory.is_blocked(row["url"])]
        if not rows:
            return 0.0, {"kind": "no_revisitable_memory", "valid_transition": False}
        random.shuffle(rows)
        state_before = self._last_action_state.detach().clone() if self._last_action_state is not None else self.state.detach().clone()
        predicted = self._predict_environment_outcome_embedding("revisit", state_before)
        for item in rows:
            try:
                prior_rows = [row for row in rows if str(row.get("url", "")) != str(item.get("url", ""))][: int(getattr(self.config.runtime, "open_world_consistency_sources", 3))]
                prior_embeddings = []
                for row in prior_rows:
                    safe_prior = self._exclude_evaluation_text(str(row.get("text", ""))).strip()
                    if len(safe_prior) >= 32:
                        prior_embeddings.append(self._text_representation(safe_prior[: int(getattr(self.config.runtime, "open_world_max_prediction_chars", 4000))]))
                page = self.web.open(item["url"])
            except WebAccessError as exc:
                self.memory.block_url(exc.url, exc.reason, source="revisit")
                continue
            safe_page_text = self._exclude_evaluation_text(page.text).strip()
            if len(safe_page_text) < 32:
                continue
            observed = self._text_representation(safe_page_text[: int(getattr(self.config.runtime, "open_world_max_prediction_chars", 4000))])
            result = self._ingest(page, source="revisit", action_name="revisit")
            score, result = self._attach_open_world_verification(result, action_name="revisit", observed_text=page.text, prior_rows=prior_rows, exclude_url=page.url, state_before=state_before, predicted_observation=predicted, observed_embedding=observed, prior_embeddings=prior_embeddings)
            return float(score if score is not None else 0.0), result
        return 0.0, {"kind": "all_revisits_blocked", "valid_transition": False}

    def _cap_reflect(self) -> tuple[float, dict]:
        if not self._language_ready():
            return 0.0, {"kind": "language_not_ready", "valid_transition": False}
        context = self._workspace()
        thought = self.model.generate(
            context + "\n\n" if context else "",
            max_new_tokens=180,
            temperature=self.config.runtime.generation_temperature,
            top_k=self.config.runtime.generation_top_k,
            device=self.device,
        ).strip()
        if not thought:
            return 0.0, {"kind": "empty_reflection", "valid_transition": False}
        self.memory.add(kind="internal", source="reflection", title="internal thought", url="", text=thought)
        return 0.0, {"kind": "reflection", "text": thought, "valid_transition": True}

    def browser_open(self, url: str) -> dict:
        if self.browser is None:
            raise BrowserEnvironmentError("browser environment is disabled")
        observation = self.browser.navigate(url)
        return self._record_browser_observation(observation, action={"action": "navigate", "url": observation.url})

    def browser_observe(self) -> dict:
        if self.browser is None:
            raise BrowserEnvironmentError("browser environment is disabled")
        observation = self.browser.observe(screenshot=True)
        return self._record_browser_observation(observation, action={"action": "observe"})

    def browser_action(self, action: dict) -> dict:
        if self.browser is None:
            raise BrowserEnvironmentError("browser environment is disabled")
        before = self.browser.last_observation.url if self.browser.last_observation else ""
        observation = self.browser.action(action)
        payload = self._record_browser_observation(observation, action=action)
        payload["before_url"] = before
        payload["after_url"] = observation.url
        return payload

    def browser_status(self) -> dict:
        if self.browser is None:
            return {"enabled": False}
        status = self.browser.state_summary()
        status["enabled"] = True
        return status

    def _browser_action_from_model(self, observation: BrowserObservation) -> dict:
        objective = self.objectives.current()
        objective_text = objective.text if objective is not None else ""
        compact = observation.compact(
            max_text=int(getattr(self.config.runtime, "browser_dom_max_chars", 12000)),
            max_elements=int(getattr(self.config.runtime, "browser_dom_max_elements", 100)),
        )
        prompt = (
            "Select one generic browser interaction for the current observation. "
            "Do not assume facts that are absent from the observation. "
            "Return exactly one JSON object and no other text. "
            "Allowed actions: observe, click, type, press, select, scroll, back, forward, reload, wait. "
            "For click/type/press/select, use only an element ref shown in the observation. "
            "For type, include a text field; for press include key; for select include value; "
            "for scroll include delta_y; for wait include milliseconds. "
            "Do not use passwords or secrets. "
        )
        if objective_text:
            prompt += f"OBJECTIVE:\n{objective_text}\n"
        prompt += f"OBSERVATION:\n{compact}\nACTION JSON:\n"
        raw = self.model.generate(
            prompt,
            max_new_tokens=max(64, int(getattr(self.config.runtime, "browser_action_max_chars", 1200)) // 4),
            temperature=self.config.runtime.generation_temperature,
            top_k=self.config.runtime.generation_top_k,
            device=self.device,
            state=self.state,
        ).strip()
        match = re.search(r"\{.*\}", raw, flags=re.DOTALL)
        if not match:
            return {"action": "observe", "reason": "model_output_not_json"}
        try:
            payload = json.loads(match.group(0))
        except json.JSONDecodeError:
            return {"action": "observe", "reason": "model_output_invalid_json"}
        if not isinstance(payload, dict):
            return {"action": "observe", "reason": "model_output_not_object"}
        kind = str(payload.get("action", "observe")).strip().lower()
        allowed = {"observe", "click", "type", "press", "select", "scroll", "back", "forward", "reload", "wait"}
        if kind not in allowed:
            return {"action": "observe", "reason": "unsupported_model_action", "raw_action": kind}
        if kind in {"click", "type", "press", "select"}:
            refs = {item.ref: item for item in observation.elements}
            ref = str(payload.get("ref", ""))
            element = refs.get(ref)
            if element is None:
                return {"action": "observe", "reason": "stale_or_unknown_ref", "ref": ref}
            if int(observation.observation_id) > 0:
                payload["observation_id"] = int(observation.observation_id)
            if str(element.signature):
                payload["signature"] = str(element.signature)
        return payload

    def _record_browser_observation(self, observation: BrowserObservation, *, action: dict) -> dict:
        compact = observation.compact(
            max_text=int(getattr(self.config.runtime, "browser_dom_max_chars", 12000)),
            max_elements=int(getattr(self.config.runtime, "browser_dom_max_elements", 100)),
        )
        memory_id = self.memory.add(
            kind="browser_observation",
            source="browser",
            title=observation.title or observation.url,
            url=observation.url,
            text=compact,
            action_name="browser_interact",
            prediction_error=0.0,
        )
        vision = None
        if observation.screenshot_b64:
            try:
                import base64
                import io
                import numpy as np
                from PIL import Image
                image = Image.open(io.BytesIO(base64.b64decode(observation.screenshot_b64))).convert("RGB")
                vision = self.perception.observe_image(np.asarray(image), train=False)
            except Exception as exc:
                vision = {"error": repr(exc)}
        with torch.no_grad():
            tokens = self._encode_text(compact[: int(getattr(self.config.runtime, "browser_dom_max_chars", 12000))])
            _, self.state, _, _ = self.model.step(
                tokens,
                self.state,
                self.previous_action,
                self.prediction_error,
            )
        event_id = self._record_cognitive_event(
            "browser_observation",
            modality="browser",
            source="browser",
            salience=min(1.0, len(observation.elements) / 32.0),
            novelty=1.0 if self._last_browser_observation.get("url") != observation.url else 0.25,
            detail={
                "url": observation.url,
                "title": observation.title,
                "element_count": len(observation.elements),
                "navigation_id": observation.navigation_id,
                "memory_id": memory_id,
                "vision": {k: v for k, v in (vision or {}).items() if k != "content_hash"},
                "action": action,
            },
        )
        payload = {
            "kind": "browser_observation",
            "valid_transition": True,
            "url": observation.url,
            "title": observation.title,
            "text_chars": len(observation.text),
            "element_count": len(observation.elements),
            "navigation_id": observation.navigation_id,
            "memory_id": memory_id,
            "cognitive_event_id": event_id,
            "screenshot": bool(observation.screenshot_b64),
            "action": action,
            "vision": vision,
        }
        self._last_browser_observation = payload
        self._last_browser_action = dict(action)
        self._record({"event": "browser", **payload})
        return payload

    def _cap_browser_interact(self) -> tuple[float, dict]:
        if self.browser is None:
            return 0.0, {"kind": "browser_disabled", "valid_transition": False}
        observation = self.browser.last_observation
        if observation is None:
            start_url = str(getattr(self.config.runtime, "browser_start_url", "")).strip()
            if not start_url:
                return 0.0, {"kind": "browser_not_open", "valid_transition": False}
            result = self.browser_open(start_url)
            return 0.0, result
        if not self._language_ready():
            result = self.browser_observe()
            result.update({"kind": "browser_language_not_ready", "valid_transition": True})
            return 0.0, result
        action = self._browser_action_from_model(observation)
        state_before = self._last_action_state.detach().clone() if self._last_action_state is not None else self.state.detach().clone()
        predicted = self._predict_environment_outcome_embedding("browser_interact", state_before)
        try:
            result = self.browser_action(action)
            current = self.browser.last_observation
            observed_text = current.compact(
                max_text=int(getattr(self.config.runtime, "browser_dom_max_chars", 12000)),
                max_elements=int(getattr(self.config.runtime, "browser_dom_max_elements", 100)),
            ) if current is not None else ""
            observed_embedding = self._text_representation(observed_text[: int(getattr(self.config.runtime, "open_world_max_prediction_chars", 4000))])
            score, result = self._attach_open_world_verification(
                result, action_name="browser_interact", observed_text=observed_text,
                query=str(self.objectives.current().text if self.objectives.current() else ""),
                state_before=state_before, include_self_qa=False, predicted_observation=predicted,
                observed_embedding=observed_embedding,
            )
        except BrowserEnvironmentError as exc:
            return 0.0, {
                "kind": "browser_action_error",
                "valid_transition": False,
                "action": action,
                "error": str(exc),
            }
        result.update({"kind": "browser_interact", "valid_transition": bool(result.get("valid_transition", True))})
        return float(score if score is not None else 0.0), result

    def _cap_consolidate(self) -> tuple[float, dict]:
        rows = self.memory.recent_replayable(1, kinds=("external",))
        if not rows:
            return 0.0, {"kind": "no_external_memory", "valid_transition": False}
        row = rows[0]
        chunks = self._training_chunks(str(row.get("text", "")), int(row.get("id", 0)), max_chunks=0)
        train_chunks, holdout_chunks = self._split_train_holdout(chunks, float(self.config.runtime.teacher_holdout_fraction))
        if not train_chunks or not holdout_chunks:
            return 0.0, {"kind": "no_holdout_for_consolidation", "valid_transition": False}
        before = float(row.get("loss_after", self._chunk_bpb(holdout_chunks)))
        last = self._train_chunk_sequences(train_chunks, self.config.runtime.sequence_length, self.config.runtime.train_steps_per_observation)
        after = self._chunk_bpb(holdout_chunks)
        progress = before - after
        self.last_holdout_bpb = after
        self.memory.add(
            kind="external", source=row.get("source", ""), title=row.get("title", ""), url=row.get("url", ""),
            text=row.get("text", ""), loss_before=before, loss_after=after, learning_progress=progress,
            action_name=row.get("action_name"), prediction_error=after,
        )
        return progress, {"kind": "consolidate", "loss": last, "holdout_before": before, "holdout_after": after, "learning_progress": progress, "valid_transition": True}

    _EXTERNAL_ACTION_CAPABILITIES = frozenset({"discover", "search", "follow", "revisit", "browser_interact", "pursue_objective"})

    def _autonomy_guard(self) -> tuple[bool, str, int]:
        """Enforce the human kill switch and hourly external-action budget."""
        runtime = self.config.runtime
        kill_path = self.root / str(getattr(runtime, "autonomy_kill_switch_path", "data/STOP_ALETHEIA"))
        try:
            if kill_path.exists():
                return False, "kill_switch_present", 0
        except OSError:
            return False, "kill_switch_check_failed", 0

        if not bool(getattr(runtime, "autonomy_external_action_budget_enabled", True)):
            return True, "budget_disabled", 0
        limit = max(1, int(getattr(runtime, "autonomy_max_external_actions_per_hour", 120)))
        ledger = self.root / str(getattr(runtime, "autonomy_action_ledger_path", "data/autonomy_actions.jsonl"))
        cutoff = time.time() - 3600.0
        recent = 0
        try:
            if ledger.exists():
                # The ledger is append-only; read from the tail to keep checks cheap.
                with ledger.open("rb") as handle:
                    handle.seek(0, os.SEEK_END)
                    size = handle.tell()
                    handle.seek(max(0, size - 256 * 1024), os.SEEK_SET)
                    data = handle.read().decode("utf-8", errors="replace")
                for line in data.splitlines():
                    try:
                        row = json.loads(line)
                        if str(row.get("status", "")) == "started" and float(row.get("time", 0.0)) >= cutoff:
                            recent += 1
                    except (ValueError, TypeError, json.JSONDecodeError):
                        continue
        except OSError:
            return False, "action_ledger_unreadable", recent
        if recent >= limit:
            return False, "external_action_hourly_budget_exhausted", recent
        return True, "ok", recent

    def _ledger_external_action(self, *, capability: str, status: str, cycle: int, detail: dict | None = None) -> None:
        path = self.root / str(getattr(self.config.runtime, "autonomy_action_ledger_path", "data/autonomy_actions.jsonl"))
        path.parent.mkdir(parents=True, exist_ok=True)
        row = {"time": time.time(), "cycle": int(cycle), "capability": str(capability), "status": str(status), "detail": dict(detail or {})}
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")

    def autonomous_cycle(self, *, train_steps: int | None = None) -> dict:
        allowed, guard_reason, recent_external_actions = self._autonomy_guard()
        if not allowed:
            event = {
                "event": "cycle_blocked",
                "cycle": self.cycles,
                "capability": "safety_guard",
                "valid_transition": False,
                "outcome_available": False,
                "guard_reason": guard_reason,
                "recent_external_actions": recent_external_actions,
            }
            self._record(event)
            return event
        if train_steps is not None:
            self.config.runtime.train_steps_per_observation = max(1, int(train_steps))
        current_objective = self.objectives.current()
        objective_progress_before = current_objective.progress if current_objective is not None else None
        action, selection_mode, policy_entropy, valid_mask = self._choose_action()
        capability = self.registry.get(action)
        state_before = self.state.detach().clone()
        self._last_action_state = state_before.detach().clone()
        self._last_action_selection_mode = selection_mode
        self._last_action_index = action
        self._last_action_state_after = None
        objective_for_target = self.objectives.current()
        target_goal = None
        if objective_for_target is not None:
            try:
                target_goal = self.planner.hierarchy.text(objective_for_target.id, objective_for_target.text)
            except Exception:
                target_goal = objective_for_target.text
        self._last_feedback_target = {
            "kind": "autonomous",
            "action_index": action,
            "action_name": self.registry.items()[action].name if 0 <= action < len(self.registry.items()) else str(action),
            "objective_id": objective_for_target.id if objective_for_target is not None else None,
            "objective_goal": target_goal,
            "created": time.time(),
        }
        self._last_policy_mask = valid_mask.detach().clone() if valid_mask is not None else None
        signal = 0.0
        result: dict
        valid_transition = False
        external_action = bool(capability.external) or capability.name in self._EXTERNAL_ACTION_CAPABILITIES
        blocked_external = False
        if external_action:
            allowed_now, reason_now, recent_now = self._autonomy_guard()
            if not allowed_now:
                blocked_external = True
                result = {
                    "kind": "external_action_blocked",
                    "error": reason_now,
                    "valid_transition": False,
                    "external_action_budget": recent_now,
                }
                self._ledger_external_action(
                    capability=capability.name, status="blocked", cycle=self.cycles, detail=result
                )
            else:
                self._ledger_external_action(capability=capability.name, status="started", cycle=self.cycles)
        try:
            if not blocked_external:
                signal, result = capability.run()
                valid_transition = bool(result.get("valid_transition", result.get("learned", True)))
        except Exception as exc:
            result = {"kind": "environment_error", "error": repr(exc), "valid_transition": False}
            # Network/service failures do not train the policy.
        if external_action and not blocked_external:
            self._ledger_external_action(
                capability=capability.name, status="finished", cycle=self.cycles,
                detail={"valid_transition": valid_transition, "kind": result.get("kind", "")},
            )
        self._last_action_state_after = self.state.detach().clone()
        if valid_transition and self.objectives.current() is not None and result.get("objective_outcome") is not None:
            try:
                outcome_result = self.record_objective_outcome(
                    float(result["objective_outcome"]),
                    source=str(result.get("outcome_source", "environment")),
                    note=json.dumps(result.get("outcome_evidence", {}), sort_keys=True),
                    persist=False,
                    verified=bool(result.get("verified", False)),
                    evidence=dict(result.get("outcome_evidence", {})),
                )
                result["objective_outcome_record"] = outcome_result
                signal = float(result["objective_outcome"])
            except Exception as exc:
                result["objective_outcome_record"] = {"updated": False, "reason": "outcome_record_error", "error": repr(exc)}
        objective_outcome_present = result.get("objective_outcome") is not None
        outcome_available = objective_outcome_present and bool(result.get("verified", False))
        if outcome_available:
            self._learn_policy(state_before, action, signal, valid_transition, selection_mode, valid_mask)
        self.previous_action = action
        self.cycles += 1
        objective_progress_after = (
            current_objective.progress if current_objective is not None else None
        )
        event = {
            "event": "cycle",
            "cycle": self.cycles,
            "capability": capability.name,
            "selection_mode": selection_mode,
            "policy_entropy": policy_entropy,
            "signal": float(signal),
            "valid_transition": valid_transition,
            "outcome_available": bool(result.get("objective_outcome") is not None and result.get("verified", False)),
            "unverified_outcome_candidate_available": bool(result.get("objective_outcome") is not None and not result.get("verified", False)),
            "prediction_error": float(self.prediction_error),
            "objective_progress_before": objective_progress_before,
            "objective_progress_after": objective_progress_after,
            "plan": dict(self._last_plan),
            "memory": self.memory.summary(),
            "unique_tokens": self.total_tokens,
            "processed_tokens": self.processed_tokens,
            "result": result,
        }
        cognitive_event_id = self._record_cognitive_event(
            "autonomous_transition", modality="internal", source="agent_runtime", action_name=capability.name,
            salience=abs(float(signal)), novelty=abs(float(self.prediction_error)),
            detail={"cycle": self.cycles, "selection_mode": selection_mode, "valid_transition": valid_transition},
        )
        event["cognitive_event_id"] = cognitive_event_id
        if self.cycles % self.config.runtime.telemetry_every == 0:
            self._record(event)
        if self.cycles % self.config.runtime.checkpoint_every == 0:
            self.save()
        return event

    @staticmethod
    def _generated_text_is_usable(text: str, prompt: str = "") -> bool:
        compact=" ".join(text.split()).strip()
        if len(compact)<8:
            return False
        if compact.lower().startswith(("turn a:", "turn b:", "source:", "title:")):
            return False
        if "[quoted]" in compact.lower() or "[no_answer_yet]" in compact.lower():
            return False
        prompt_terms=set(lexical_terms(prompt))
        output_terms=set(lexical_terms(compact))
        if prompt_terms and len(compact)>20 and len(output_terms & prompt_terms)/max(1,len(output_terms))>0.92:
            return False
        letters=sum(ch.isalpha() for ch in compact)
        if letters<5:
            return False
        longest=1
        run=1
        for left,right in zip(compact,compact[1:]):
            if left==right:
                run+=1; longest=max(longest,run)
            else:
                run=1
        return longest<max(8,len(compact)//3)

    @staticmethod
    def _experimental_text_is_usable(text: str) -> bool:
        """Reject clearly corrupted early-model output without asserting semantic correctness.

        This lightweight integrity check is deliberately separate from scientific language
        readiness: readable-looking text can still be wrong, and only held-out evaluation can
        establish learning progress. Unicode letters are allowed for multilingual output.
        """
        compact = " ".join(str(text).split()).strip()
        if not compact or "\ufffd" in compact:
            return False
        for character in compact:
            category = unicodedata.category(character)
            if category == "Cc" and not character.isspace():
                return False
            if category == "Cs":
                return False
        visible = [character for character in compact if not character.isspace()]
        if not visible:
            return False
        wordlike = sum(character.isalnum() or unicodedata.category(character).startswith("M") for character in visible)
        if wordlike / len(visible) < 0.60:
            return False
        # When a model emits whitespace-separated text, insist on more than a single
        # accidental fragment. Scripts commonly written without spaces remain supported.
        if any(character.isspace() for character in compact) and len(lexical_terms(compact)) < 2:
            return False
        return True

    def _chat_generation_ready(self) -> bool:
        """Return formal evaluation readiness (not whether a user may attempt a chat)."""
        return self._language_ready()

    def _clean_generated_reply(self, text: str) -> str:
        reply=text.replace("Turn A:", "").replace("Turn B:", "").replace("Response:", "")
        reply=re.sub(r"\s+", " ", reply).strip()
        return reply.strip(" \t\r\n")

    def _evidence_response(self, message: str, rows: list[dict]) -> str:
        terms=set(self._informative_terms(message))
        candidates=[]
        for row in rows:
            text=" ".join(str(row.get("text", "")).split())
            sentences=[part.strip() for part in re.split(r"(?<=[.!?])\s+",text) if part.strip()]
            for sentence in sentences:
                st={t.lower() for t in re.findall(r"[A-Za-z0-9]{3,}",sentence)}
                overlap=len(terms & st)
                if overlap>0:
                    candidates.append((overlap/max(1,len(terms)), overlap, sentence, str(row.get("title", "memory"))))
        candidates.sort(key=lambda item:(item[0],item[1]), reverse=True)
        if not candidates:
            return ""
        _, _, sentence, title = candidates[0]
        return f"[quoted] From {title}: {sentence}"

    def communicate(self, message: str, max_tokens: int = 256, *, speaker_id: str | None = None, source: str = "human_chat") -> str:
        message=" ".join(message.strip().split())
        if not message:
            return ""
        before=self.evaluate_text_loss(message)
        self.memory.add(
            kind="conversation_user", source=source, title="user message", url="",
            text=message, loss_before=before, loss_after=before, learning_progress=0.0,
            merge_existing=False,
        )
        # A bare user message is not treated as a training target. Aletheia only
        # learns it as language when a supervised/teacher sequence supplies the continuation.
        retrieved=self._retrieve_experience(message)
        with torch.no_grad():
            _, self.state, _, _ = self.model.step(
                self._encode_text(message),
                self.state,
                self.previous_action,
                self.prediction_error,
            )
        prompt=self._communication_context(message, reply_reserve=max_tokens)
        reply=""
        mode="unknown"
        generation_metrics={}
        formally_ready = self._chat_generation_ready()
        allow_early_chat = bool(getattr(self.config.runtime, "chat_allow_unready_generation", True))
        if formally_ready or allow_early_chat:
            # Keep low-maturity chat cheap: one candidate and a bounded reply. This is
            # only a best-effort user-initiated conversation path; it does not promote
            # the model to language-ready or enable autonomous initiation/browser actions.
            generation_limit = max(1, int(max_tokens))
            candidate_count = max(1, int(self.config.runtime.chat_candidate_count))
            if not formally_ready:
                generation_limit = min(
                    generation_limit,
                    max(1, int(getattr(self.config.runtime, "chat_unready_max_response_tokens", 64))),
                )
                candidate_count = 1
            generated,new_state,generation_metrics=self.model.generate_candidates(
                prompt,
                candidate_count=candidate_count,
                max_new_tokens=generation_limit,
                temperature=self.config.runtime.chat_temperature,
                top_k=self.config.runtime.chat_top_k,
                device=self.device,
                state=self.state,
            )
            generated=self._clean_generated_reply(generated)
            usable = self._generated_text_is_usable(generated, message)
            if usable and not formally_ready:
                usable = self._experimental_text_is_usable(generated)
            if usable:
                reply=f"[generated] {generated}" if formally_ready else f"[experimental] {generated}"
                mode="generated" if formally_ready else "experimental_generation"
                self.state=new_state.detach()

        if not reply and retrieved:
            reply=self._evidence_response(message,retrieved)
            mode="evidence" if reply else "no_answer"
        if not reply:
            reply="[not_ready] I do not have enough learned language or relevant experience to answer that yet."
            mode="no_answer"

        response_memory_id = self.memory.add(
            kind="conversation_response", source="aletheia", title="Aletheia response", url="", text=reply,
            merge_existing=False,
        )
        self._last_feedback_target = {
            "kind": "chat",
            "message": message,
            "reply": reply,
            "memory_id": response_memory_id,
            "speaker_id": speaker_id,
            "source": source,
            "created": time.time(),
        }
        cognitive_event_id = self._record_cognitive_event(
            "conversation_turn", modality="text", source="human_chat",
            salience=min(1.0, len(message.encode("utf-8", errors="replace")) / 512.0),
            novelty=1.0 if not retrieved else 0.25,
            detail={"response_mode": mode, "retrieved_count": len(retrieved), "generation": generation_metrics, "speaker_id": speaker_id, "source": source},
        )
        self._record({
            "event":"communication",
            "cognitive_event_id": cognitive_event_id,
            "message_chars":len(message),
            "reply_chars":len(reply),
            "retrieved_memories":[int(row.get("memory_id",row.get("id",0))) for row in retrieved],
            "response_mode":mode,
            "chat_generation_ready":self._chat_generation_ready(),
            "language_ready":self._language_ready(),
            "tokenizer": {"kind": getattr(self.tokenizer, "name", "byte"), "vocab_size": int(self.tokenizer.vocab_size), "path": str(self.tokenizer_path)},
            "language_bpb":self.last_eval_bpb,
            "baseline_bpb":self.eval_baseline_bpb,
            "generation_metrics":generation_metrics,
        })
        self._save_runtime_state()
        self._chat_turns_since_checkpoint += 1
        checkpoint_interval = max(1, int(getattr(self.config.runtime, "checkpoint_every", 25)))
        if self._chat_turns_since_checkpoint >= checkpoint_interval:
            self.save()
            self._chat_turns_since_checkpoint = 0
        return reply

    def _record_cognitive_event(
        self, event_kind: str, *, modality: str = "internal", source: str = "",
        action_name: str | None = None, entity_id: str | None = None,
        salience: float = 0.0, novelty: float = 0.0, prediction_error: float | None = None,
        detail: dict | None = None,
    ) -> int:
        """Persist compact experience metadata while keeping raw sensor payloads separate."""
        self._cognitive_step += 1
        if prediction_error is None:
            prediction_error = self.prediction_error
        try:
            with torch.no_grad():
                uncertainty = float(self.model.transition_uncertainty(self.state).mean().detach().cpu())
        except Exception:
            uncertainty = 0.0
        self._last_attention_uncertainty = uncertainty
        computed_salience = max(float(salience), uncertainty / (1.0 + uncertainty))
        event_id = self.memory.add_cognitive_event(
            event_kind=event_kind, modality=modality, source=source, action_name=action_name,
            entity_id=entity_id, salience=computed_salience, novelty=float(max(0.0, novelty)),
            prediction_error=float(max(0.0, prediction_error)),
            detail={**(detail or {}), "cognitive_step": self._cognitive_step, "uncertainty": uncertainty,
                    "attention_worthy": computed_salience >= float(getattr(self.config.runtime, "cognitive_attention_threshold", 0.08))},
            limit=int(getattr(self.config.runtime, "cognitive_event_limit", 50000)),
        )
        self._last_cognitive_event_id = event_id
        return event_id

    def _record(self, event: dict) -> None:
        with self.paths.telemetry.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps({"time": time.time(), **event}, ensure_ascii=False) + "\n")

    def save(self) -> None:
        self.perception.flush()
        self.model.save(self.paths.model, self.optimizer)
        self.planner.save()
        self.objective_value.save()
        self._save_perception_state()
        self._save_runtime_state()

    def run(self, cycles: int, train_steps: int | None = None) -> None:
        print("Aletheia — from-scratch open-world developmental core")
        print(f"device={self.device}")
        print(f"torch={torch.__version__} | cuda_available={torch.cuda.is_available()} | cuda_version={torch.version.cuda}")
        if self.device.type == "cpu":
            print("NOTICE: Aletheia is running on CPU. Install a CUDA-enabled PyTorch build for long training runs on an NVIDIA GPU.")
        print(f"parameters={self.parameter_count():,}")
        print(f"capabilities={self.registry.names()}")
        print(f"memory={self.memory.summary()} | lessons={self.memory.lesson_count()} | unique_tokens={self.total_tokens} | processed_tokens={self.processed_tokens}")
        print(f"cognitive={self.memory.cognitive_summary(int(getattr(self.config.runtime, 'cognitive_recent_events', 128)))}")
        if self.last_eval_bpb is not None:
            print(f"language_eval_bpb={self.last_eval_bpb:.4f} | baseline_bpb={self.eval_baseline_bpb:.4f} | gain={self.eval_baseline_bpb-self.last_eval_bpb:.4f} | ready={self._language_ready()}")
        recovery = getattr(self, "_checkpoint_recovery_message", None)
        if recovery:
            print("CHECKPOINT: incompatible prior checkpoint preserved; started a fresh model")
            print(f"DETAIL: {recovery}")
        for _ in range(max(0, int(cycles))):
            event = self.autonomous_cycle(train_steps=train_steps)
            result = event["result"]
            print(f"\n=== cycle {event['cycle']} | {event['capability']} | mode={event['selection_mode']} | signal={event['signal']:.6f} | entropy={event['policy_entropy']:.3f} ===")
            if result.get("kind") == "environment_error":
                print("ERROR:", result.get("error"))
                if result.get("errors"):
                    print("DETAILS:", result["errors"])
            elif result.get("kind") == "search":
                print(f"SEARCH: {result.get('query', '')} | results={result.get('results', 0)}")
                if result.get("title"):
                    print(f"LEARNED: {result['title'][:100]} | progress={result.get('learning_progress', 0):.6f}")
            elif result.get("kind") == "reflection":
                print("THOUGHT:", result.get("text", "")[:300])
            elif result.get("kind") == "consolidate":
                print(f"CONSOLIDATE: loss={result.get('loss', 0):.4f} | progress={result.get('learning_progress', 0):.6f}")
            elif result.get("learned"):
                print(f"LEARNED: {result.get('title', '')[:100]} | loss={result.get('loss_after', 0):.4f} | progress={result.get('learning_progress', 0):.6f} | links={result.get('links', 0)}")
            else:
                print("EVENT:", result)
            print("MEMORY:", self.memory.summary())
            print("TOKENS:", self.total_tokens)

    def run_forever(
        self,
        train_steps: int | None = None,
        *,
        sleep_seconds: float = 0.0,
        autosave_on_error: bool = True,
        max_consecutive_errors: int = 8,
    ) -> None:
        print("Aletheia autonomous developmental runtime started. Press Ctrl+C to stop.")
        consecutive_errors = 0
        backoff = max(0.1, float(sleep_seconds))
        while True:
            try:
                event = self.autonomous_cycle(train_steps=train_steps)
                consecutive_errors = 0
                backoff = max(0.0, float(sleep_seconds))
                print(
                    f"\n=== cycle {event['cycle']} | {event['capability']} | "
                    f"signal={event['signal']:.6f} | valid={event['valid_transition']} ==="
                )
                if backoff > 0:
                    time.sleep(backoff)
            except KeyboardInterrupt:
                print("\nAletheia runtime stopped by operator.")
                break
            except Exception as exc:
                consecutive_errors += 1
                self._record({"event": "runtime_error", "error": repr(exc), "consecutive_errors": consecutive_errors})
                if autosave_on_error:
                    try:
                        self.save()
                    except Exception:
                        pass
                if consecutive_errors >= max(1, int(max_consecutive_errors)):
                    raise RuntimeError(
                        f"Aletheia stopped after {consecutive_errors} consecutive runtime errors; inspect experience.jsonl"
                    ) from exc
                backoff = min(60.0, max(0.1, backoff * 2.0 if backoff else 0.5))
                time.sleep(backoff)

    def close(self) -> None:
        self.save()
        self.memory.close()
        try:
            if self.browser is not None:
                self.browser.close()
        finally:
            self.web.close()
