from __future__ import annotations

import json
from dataclasses import dataclass, fields, field
from pathlib import Path

from .tokenizer import VOCAB_SIZE


@dataclass
class ModelConfig:
    vocab_size: int = VOCAB_SIZE
    dim: int = 384
    layers: int = 8
    heads: int = 8
    ffn: int = 1536
    recurrent_dim: int = 768
    context: int = 1536
    dropout: float = 0.05
    gradient_checkpointing: bool = True
    perception_dim: int = 256
    world_model_hidden: int = 1024
    world_attention_heads: int = 8
    world_prediction_weight: float = 0.15
    world_uncertainty_weight: float = 0.05


@dataclass
class CapabilityConfig:
    description: str
    external: bool = False
    produces_observation: bool = False
    requires_memory: int = 0
    requires_language: bool = False
    warmup_priority: float = 0.0
    policy_train: bool = True


@dataclass
class RuntimeConfig:
    learning_rate: float = 3e-4
    policy_learning_rate: float = 1e-4
    policy_entropy_weight: float = 0.005
    weight_decay: float = 1e-2
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    replay_sequences: int = 8
    replay_mix_fraction: float = 0.30
    replay_min_sequences: int = 4
    replay_lesson_pool: int = 0
    replay_max_sequences_per_lesson: int = 4
    lesson_rollback_on_fixed_regression: bool = True
    lesson_max_fixed_regression: float = 0.15
    lesson_max_local_regression: float = 0.10
    lesson_max_transfer_regression: float = 0.10
    lesson_max_aggregate_regression: float = 0.03
    lesson_hard_fixed_regression: float = 0.25
    lesson_hard_local_regression: float = 0.20
    lesson_hard_transfer_regression: float = 0.20
    lesson_fixed_weight: float = 0.50
    lesson_local_weight: float = 0.25
    lesson_transfer_weight: float = 0.25
    lesson_transfer_documents: int = 4
    lesson_transfer_chunks_per_document: int = 2
    dialogue_max_examples: int = 2048
    dialogue_passes: int = 4
    dialogue_alignment_eval_examples: int = 128
    dialogue_alignment_eval_seed: int = 1337
    dialogue_holdout_fraction: float = 0.20
    web_learning_rollback_on_fixed_regression: bool = True
    web_learning_max_fixed_regression: float = 0.03
    web_learning_max_anchor_regression: float = 0.02
    web_learning_max_local_regression: float = 0.10
    web_learning_max_transfer_regression: float = 0.10
    web_learning_max_aggregate_regression: float = 0.03
    web_learning_hard_fixed_regression: float = 0.25
    web_learning_hard_local_regression: float = 0.20
    web_learning_hard_transfer_regression: float = 0.20
    web_learning_fixed_weight: float = 0.50
    web_learning_local_weight: float = 0.25
    web_learning_transfer_weight: float = 0.25
    web_learning_use_replay: bool = True
    sequence_length: int = 4
    train_context: int = 1536
    max_training_chunks_per_memory: int = 0
    train_batch_size: int = 1
    mixed_precision: bool = True
    gradient_clip: float = 1.0
    observation_train_chunks: int = 0
    observation_train_epochs: int = 1
    teacher_max_chunks: int = 0
    teacher_sequence_length: int = 4
    teacher_epochs: int = 6
    teacher_steps_per_sequence: int = 1
    teacher_max_source_chars: int = 0
    teacher_strip_transcript_annotations: bool = True
    teacher_shard_max_chars: int = 1000000
    teacher_holdout_fraction: float = 0.10
    memory_limit: int = 20000
    memory_kind_limits: dict[str, int] = field(default_factory=dict)
    memory_eviction_order: list[str] = field(default_factory=list)
    train_steps_per_observation: int = 2
    minimum_bootstrap_memories: int = 8
    policy_epsilon: float = 0.10
    device: str = "auto"
    web_timeout: float = 20.0
    web_max_page_chars: int = 100000
    web_min_host_interval: float = 3.0
    web_max_retries: int = 4
    web_max_redirects: int = 5
    max_text_chunks_per_observation: int = 0
    generation_temperature: float = 0.85
    generation_top_k: int = 48
    checkpoint_every: int = 25
    telemetry_every: int = 1
    max_query_chars: int = 180
    max_frontier: int = 10000
    user_agent: str = "Aletheia/1.0 developmental research learner"
    policy_baseline_decay: float = 0.95
    train_progress_clip: float = 5.0
    search_result_limit: int = 5
    chat_train_steps: int = 1
    chat_memory_results: int = 6
    chat_memory_chars: int = 420
    chat_context_chars: int = 1500
    chat_temperature: float = 0.75
    chat_top_k: int = 40
    chat_max_response_tokens: int = 256
    # Best-effort interactive chat may run before benchmark-based language readiness.
    # Unready outputs are explicitly labeled experimental and never unlock autonomous speech.
    chat_allow_unready_generation: bool = True
    chat_unready_max_response_tokens: int = 64
    chat_fallback_sentences: int = 1
    chat_candidate_count: int = 3
    chat_min_retrieval_terms: int = 2
    chat_min_single_term_chars: int = 8
    chat_recent_lessons: int = 12
    language_ready_min_tokens: int = 500000
    language_ready_min_lessons: int = 5
    language_ready_min_bpb_gain: float = 0.25
    language_ready_min_eval_bytes: int = 50000
    dialogue_train_epochs: int = 1
    conversation_replay_examples: int = 16
    dialogue_replay_multiplier: float = 2.0
    dialogue_learning_rate_scale: float = 0.25
    dialogue_max_fixed_regression: float = 0.02
    dialogue_alignment_weight: float = 0.15
    dialogue_alignment_batch_size: int = 4
    dialogue_alignment_temperature: float = 0.12
    language_eval_path: str = "evaluation/heldout.txt"
    language_eval_max_chars: int = 0
    evaluation_bootstrap_samples: int = 1000
    evaluation_seed: int = 1337
    language_eval_every: int = 25
    user_message_train_steps: int = 1
    language_ready_max_baseline_ratio: float = 0.90
    chat_min_lexical_score: float = 0.20
    chat_min_matching_terms: int = 1
    chat_common_term_doc_fraction: float = 0.50
    transfer_probe_documents: int = 4
    transfer_probe_chunks_per_document: int = 2
    policy_failure_penalty: float = 0.0
    evaluation_corpus_target_bytes: int = 80000
    evaluation_exclude_paths: list[str] = field(default_factory=list)
    evaluation_overlap_words: int = 16
    evaluation_overlap_match_fraction: float = 0.20
    transfer_probe_candidate_pool: int = 64
    # Initial language-foundation pretraining is deliberately separate from
    # continual-learning acceptance gates. It interleaves shards from all supplied
    # sources and uses its own schedule/regularization recipe.
    pretrain_learning_rate: float = 6e-4
    pretrain_weight_decay: float = 0.10
    pretrain_beta1: float = 0.9
    pretrain_beta2: float = 0.95
    pretrain_gradient_accumulation_steps: int = 8
    pretrain_warmup_steps: int = 100
    pretrain_decay_steps: int = 0
    pretrain_min_lr_ratio: float = 0.10
    pretrain_sequence_length: int = 4
    pretrain_epochs: int = 2
    pretrain_seed: int = 1337
    pretrain_eval_every_shards: int = 1
    pretrain_ema_decay: float = 0.999
    pretrain_finalize_ema: bool = True
    pretrain_strip_transcript_annotations: bool = True
    pretrain_record_memory: bool = True
    perception_embedding_dim: int = 256
    perception_batch_size: int = 32
    perception_buffer_size: int = 256
    perception_learning_rate: float = 2e-4
    perception_buffer_persist_interval: int = 32
    perception_min_visual_delta: float = 0.02
    perception_identity_reindex_interval: int = 8
    perception_identity_exemplars: int = 4
    vision_object_slots: int = 6
    vision_object_slot_dim: int = 128
    vision_object_iterations: int = 3
    vision_object_max: int = 6
    vision_object_min_mass: float = 0.02
    vision_tracker_max_tracks: int = 32
    vision_tracker_match_threshold: float = 0.55
    vision_tracker_max_age: int = 6
    vision_tracker_ema: float = 0.25
    vision_entity_threshold: float = 0.84
    vision_entity_ema: float = 0.15
    vision_entity_max_entities: int = 5000
    web_crawl_max_pages: int = 8
    web_crawl_max_depth: int = 2
    web_crawl_max_attempts: int = 32
    web_crawl_exclude_patterns: list[str] = field(default_factory=list)
    parquet_text_column: str = "text"
    pretrain_max_source_bytes: int = 0
    web_max_download_bytes: int = 50 * 1024 * 1024
    web_max_pdf_chars: int = 5_000_000
    web_plain_text_default_encoding: str = "utf-8"
    teach_parquet_max_source_bytes: int = 0
    initiative_enabled: bool = False
    initiative_interval_seconds: float = 30.0
    initiative_min_visual_novelty: float = 0.05
    initiative_speak: bool = True
    presence_camera_interval: float = 0.5
    presence_audio_seconds: float = 1.0
    presence_audio_sample_rate: int = 16000
    presence_audio_activity_threshold: float = 0.02
    presence_visual_weight: float = 0.7
    presence_audio_weight: float = 0.3
    presence_initiative_cooldown_seconds: float = 45.0
    presence_max_seconds: int = 0
    speech_learning_rate: float = 2e-4
    speech_ready_min_steps: int = 100
    speech_ready_max_cer: float = 0.35
    speech_ready_min_eval_examples: int = 8
    speech_holdout_fraction: float = 0.20
    speech_batch_size: int = 4
    speech_eval_max_examples: int = 256
    speech_augmentation: bool = True
    speech_checkpoint_quarantine: bool = True
    stream_sample_rate: int = 16000
    stream_window_seconds: float = 3.0
    stream_update_seconds: float = 0.75
    stream_vad_threshold: float = 0.02
    stream_end_silence_chunks: int = 4
    stream_min_speech_seconds: float = 0.25
    stream_max_utterance_seconds: float = 20.0
    speaker_threshold: float = 0.78
    speaker_max_exemplars: int = 8
    dialogue_online_learning_enabled: bool = True
    dialogue_online_steps: int = 1
    dialogue_online_learning_rate_scale: float = 0.15
    dialogue_online_replay_examples: int = 4
    dialogue_online_max_fixed_regression: float = 0.02
    dialogue_online_max_response_loss_regression: float = 0.02
    tts_learning_rate: float = 2e-4
    tts_holdout_fraction: float = 0.20
    tts_ready_min_steps: int = 25
    tts_ready_max_loss: float = 1.0
    tts_ready_min_eval_examples: int = 8
    tts_sample_rate: int = 16000
    tts_n_fft: int = 400
    tts_hop_length: int = 160
    tts_n_mels: int = 80
    tts_speaker_dim: int = 256
    tts_text_dim: int = 128
    tts_encoder_hidden: int = 192
    tts_decoder_hidden: int = 256
    tts_attention_dim: int = 128
    tts_griffin_lim_iterations: int = 32
    tts_max_synthesis_frames: int = 240
    tts_prefer_learned: bool = True
    tts_playback: bool = True
    tts_default_speaker: str = ""
    tts_checkpoint_quarantine: bool = True
    cognitive_event_limit: int = 50000
    cognitive_attention_threshold: float = 0.08
    cognitive_recent_events: int = 128
    identity_threshold: float = 0.92
    evolution_source_bpb_tolerance: float = 0.0001
    evolution_test_timeout: int = 300
    evolution_population_size: int = 6
    evolution_candidate_mutations: list[str] = field(default_factory=lambda: [
        "baseline", "model_noise", "vision_noise", "audio_noise", "planner_noise", "tts_noise"
    ])
    evolution_runtime_mutations: list[str] = field(default_factory=lambda: [
        "learning_rate", "policy_learning_rate", "perception_learning_rate",
        "speech_learning_rate", "tts_learning_rate", "planning_learning_rate"
    ])
    evolution_runtime_scales: list[float] = field(default_factory=lambda: [0.85, 1.15])
    evolution_mutation_scale: float = 0.005
    evolution_generation_seed: int = 1337
    evolution_eval_seeds: list[int] = field(default_factory=lambda: [1337, 2027, 3037])
    evolution_runtime_min_bpb_gain: float = 0.005
    evolution_min_improvement: float = 0.01
    evolution_max_metric_regression: float = 0.05
    evolution_benchmark_max_items: int = 64
    evolution_benchmark_repeats: int = 3
    evolution_candidate_train_steps: int = 3
    evolution_benchmark_holdout_fraction: float = 0.20
    evolution_keep_candidates: int = 2
    evolution_keep_generations: int = 3
    evolution_auto_promote: bool = False
    evolution_benchmark_weights: dict[str, float] = field(default_factory=lambda: {
        "language_bpb": 0.35,
        "vision_ssl_loss": 0.20,
        "audio_ssl_loss": 0.15,
        "speech_cer": 0.15,
        "tts_holdout_loss": 0.15,
    })
    planning_enabled: bool = True
    planning_horizon: int = 3
    planning_beam_width: int = 6
    planning_branch_factor: int = 4
    planning_goal_weight: float = 0.65
    planning_alignment_weight: float = 0.20
    planning_policy_weight: float = 0.15
    planning_uncertainty_weight: float = 0.08
    planning_discount: float = 0.90
    planning_learning_rate: float = 2e-4
    planning_value_hidden: int = 256
    planning_subgoals: int = 6
    planning_decompose_on_objective: bool = True
    planning_feedback_steps: int = 1
    objective_value_learning_rate: float = 0.0002
    objective_value_hidden: int = 192
    objective_value_batch_size: int = 32
    objective_value_buffer_size: int = 4096
    objective_value_min_examples: int = 32
    objective_value_training_steps: int = 2
    objective_verifier_default: str = "manual"
    objective_completion_threshold: float = 0.95
    objective_required_consecutive: int = 1
    evolution_require_hard_sandbox: bool = True
    evolution_self_edit_allowlist: list[str] = field(default_factory=lambda: ["aletheia/learning_adapter.py", "aletheia/experimental.py"])
    evolution_source_context_chars: int = 4200
    evolution_require_change: bool = True
    evolution_confirmation_holdout: bool = True
    evolution_min_confirmation_improvement: float = 0.005
    evolution_task_benchmark_repeats: int = 2
    evolution_require_human_approval: bool = True
    autonomy_kill_switch_path: str = "data/STOP_ALETHEIA"
    autonomy_max_external_actions_per_hour: int = 120
    autonomy_action_ledger_path: str = "data/autonomy_actions.jsonl"
    autonomy_external_action_budget_enabled: bool = True
    browser_persistent_profile: bool = False
    browser_allow_domains: list[str] = field(default_factory=list)
    browser_allow_form_submit: bool = False
    browser_allow_downloads: bool = False
    browser_allow_file_uploads: bool = False
    browser_allow_authentication: bool = False
    browser_dns_cache_seconds: float = 30.0
    # Remote Chromium browsing requires an egress proxy whose destination policy is
    # enforced at connection time plus host firewall rules that prevent direct bypass.
    browser_egress_proxy_url: str = ""
    browser_egress_proxy_policy_confirmed: bool = False
    open_world_self_qa_enabled: bool = False
    open_world_duplicate_similarity_threshold: float = 0.98
    open_world_require_independent_evidence: bool = True
    browser_enabled: bool = False
    browser_headless: bool = True
    browser_executable_path: str = ""
    browser_user_data_dir: str = "data/browser_profile"
    browser_timeout_ms: int = 20000
    browser_respect_robots: bool = True
    browser_screenshot_max_bytes: int = 8 * 1024 * 1024
    browser_dom_max_chars: int = 12000
    browser_dom_max_elements: int = 100
    browser_action_max_chars: int = 1200
    browser_start_url: str = ""
    tokenizer_path: str = "data/tokenizer.json"
    tokenizer_vocab_size: int = 2048
    tokenizer_min_frequency: int = 2
    tokenizer_train_max_bytes: int = 16_000_000
    tokenizer_max_segments: int = 75000
    tokenizer_reset_readiness_on_change: bool = True
    open_world_enabled: bool = True
    open_world_max_prediction_chars: int = 4000
    open_world_consistency_sources: int = 3
    open_world_self_qa_examples: int = 1
    open_world_min_source_chars: int = 96


@dataclass
class WebSearchProvider:
    name: str
    url: str
    query_parameter: str = "q"
    result_selector: str = ""
    result_attribute: str = "href"
    enabled: bool = True


@dataclass
class DiscoverySource:
    name: str
    url: str
    type: str = "page"
    enabled: bool = True


@dataclass
class WebConfig:
    search_providers: list[WebSearchProvider]
    bootstrap_discovery: list[DiscoverySource]
    allowed_schemes: list[str]


@dataclass
class AgentConfig:
    capabilities: dict[str, CapabilityConfig]
    model: ModelConfig
    runtime: RuntimeConfig
    web: WebConfig


@dataclass
class Paths:
    root: Path

    @property
    def data(self) -> Path:
        return self.root / "data"

    @property
    def model(self) -> Path:
        return self.data / "aletheia.pt"

    @property
    def telemetry(self) -> Path:
        return self.data / "experience.jsonl"

    @property
    def memory(self) -> Path:
        return self.data / "memory.sqlite3"

    @property
    def tokenizer(self) -> Path:
        return self.data / "tokenizer.json"


def _build(cls, payload: dict, *, ignored_keys: set[str] | None = None):
    if not isinstance(payload, dict):
        raise TypeError(f"{cls.__name__} configuration must be an object")
    allowed = {field.name for field in fields(cls)}
    ignored = set(ignored_keys or set())
    unknown = sorted(set(payload) - allowed - ignored)
    if unknown:
        raise ValueError(f"Unknown {cls.__name__} configuration keys: {', '.join(unknown)}")
    return cls(**{key: value for key, value in payload.items() if key in allowed})


def load_config(path: Path | None = None) -> AgentConfig:
    path = path or Path("config.json")
    payload = json.loads(path.read_text(encoding="utf-8"))
    capabilities = {
        name: _build(CapabilityConfig, settings, ignored_keys={"enabled"})
        for name, settings in payload["capabilities"].items()
        if settings.get("enabled", True)
    }
    return AgentConfig(
        capabilities=capabilities,
        model=_build(ModelConfig, payload["model"]),
        runtime=_build(RuntimeConfig, payload["runtime"]),
        web=WebConfig(
            search_providers=[_build(WebSearchProvider, item) for item in payload["web"].get("search_providers", []) if item.get("enabled", True)],
            bootstrap_discovery=[_build(DiscoverySource, item) for item in payload["web"].get("bootstrap_discovery", []) if item.get("enabled", True)],
            allowed_schemes=list(payload["web"].get("allowed_schemes", ["http", "https"])),
        ),
    )
