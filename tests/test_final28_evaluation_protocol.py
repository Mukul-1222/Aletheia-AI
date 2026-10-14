from __future__ import annotations

import hashlib

import pytest

from aletheia.evaluation_protocol import summarize_bpb_rows, validate_source_separated_manifest
from aletheia.deterministic_env import DeterministicEnvironment


def _record(item_id, source, split, text, *, replay=False):
    return {"item_id": item_id, "source_id": source, "split": split,
            "sha256": hashlib.sha256(text.encode()).hexdigest(),
            "memory_replay_allowed": replay, "tokenizer_training_allowed": split == "train"}


def test_manifest_rejects_source_leakage_and_heldout_replay():
    rows = [_record("a", "same-book", "train", "one"),
            _record("b", "same-book", "fixed-heldout", "two", replay=True)]
    errors = validate_source_separated_manifest(rows)
    assert any("crosses splits" in error for error in errors)
    assert any("memory_replay_allowed=false" in error for error in errors)


def test_manifest_accepts_source_separated_documents():
    rows = [_record("a", "book-a", "train", "one", replay=True),
            _record("b", "book-b", "fixed-heldout", "two")]
    assert validate_source_separated_manifest(rows) == []


def test_bpb_aggregation_is_byte_weighted_and_chunk_invariant():
    rows_a = [{"source_id": "doc1", "domain": "prose", "nll_bits": 10, "byte_count": 10},
              {"source_id": "doc2", "domain": "tech", "nll_bits": 90, "byte_count": 90}]
    rows_b = [{"source_id": "doc1", "domain": "prose", "nll_bits": 4, "byte_count": 4},
              {"source_id": "doc1", "domain": "prose", "nll_bits": 6, "byte_count": 6},
              rows_a[1]]
    first = summarize_bpb_rows(rows_a, bootstrap_samples=100, seed=4)
    second = summarize_bpb_rows(rows_b, bootstrap_samples=100, seed=4)
    assert first["bpb"] == pytest.approx(1.0)
    assert second["bpb"] == pytest.approx(first["bpb"])
    assert first["by_domain"]["tech"]["bpb"] == pytest.approx(1.0)
    assert first["ci"] == second["ci"]


def test_bpb_rejects_inconsistent_domain_for_same_document():
    with pytest.raises(ValueError):
        summarize_bpb_rows([
            {"source_id": "doc", "domain": "a", "nll_bits": 2, "byte_count": 2},
            {"source_id": "doc", "domain": "b", "nll_bits": 2, "byte_count": 2},
        ])


def test_deterministic_environment_separates_prediction_and_external_outcome():
    env = DeterministicEnvironment(initial_state=0, transitions={(0, "advance"): 1, (1, "finish"): 2},
                                   goal_verifier=lambda state: state == 2, max_steps=3)
    first = env.step("advance", predicted_next_state=9)
    assert first.prediction_match is False
    assert first.externally_verified is False
    assert env.evaluator_result()["status"] == "failure"
    second = env.step("finish", predicted_next_state=2)
    assert second.prediction_match is True
    assert second.externally_verified is True
    assert env.evaluator_result()["status"] == "success"
    assert "goal" not in vars(second.next_observation)


def test_deterministic_environment_reports_missing_feedback_as_unknown():
    env = DeterministicEnvironment(initial_state="x", transitions={("x", "stay"): "x"},
                                   goal_verifier=None, max_steps=1)
    env.step("stay")
    assert env.evaluator_result() == {"status": "unknown", "reason": "feedback_missing"}


def test_deterministic_environment_rejects_invalid_and_terminal_actions():
    env = DeterministicEnvironment(initial_state=0, transitions={(0, "noop"): 0},
                                   goal_verifier=lambda state: False, max_steps=1)
    with pytest.raises(ValueError, match="invalid_action"):
        env.step("missing")
    transition = env.step("noop")
    assert transition.verifier_reason == "timeout"
    with pytest.raises(RuntimeError, match="terminal"):
        env.step("noop")


def test_dialogue_serializer_keeps_training_and_inference_turn_boundaries():
    from aletheia.dialogue import format_dialogue_example, format_turn_prompt, parse_dialogue_example

    inference = format_turn_prompt("hello 🌍")
    assert inference == "Turn A:\nhello 🌍\nTurn B:\n"
    example = format_dialogue_example("hello 🌍", "hi there")
    assert example == inference + "hi there\n"
    assert parse_dialogue_example(example) == ("hello 🌍", "hi there")
    # Existing saved dialogue-feedback rows use the same markers with no newline after colon.
    assert parse_dialogue_example("Turn A: hello\nTurn B: hi") == ("hello", "hi")


def test_dialogue_serializer_does_not_nest_observed_context():
    from aletheia.dialogue import format_dialogue_example

    example = format_dialogue_example("Turn A:\nquestion\nTurn B:\nprior answer", "new answer")
    assert example.count("Turn A:") == 1
    assert example.endswith("\nTurn B:\nnew answer\n")


def test_capability_report_never_confuses_dependency_presence_with_tested():
    from aletheia.capability_report import build_capability_report

    report = build_capability_report()
    allowed = {"tested", "not tested", "dependency missing", "failed"}
    assert report["release"] == "FINAL34"
    assert report["capabilities"]
    assert all(item["status"] in allowed for item in report["capabilities"])
    assert all(item["status"] != "tested" for item in report["capabilities"])


def test_browser_websocket_guard_closes_socket():
    from aletheia.browser import BrowserEnvironment

    class SocketRoute:
        closed = None
        def close(self, code=1000, reason=""):
            self.closed = (code, reason)

    route = SocketRoute()
    BrowserEnvironment._guard_websocket(route)
    assert route.closed is not None
    assert route.closed[0] == 1008
    assert "disabled" in route.closed[1].lower()


def test_browser_request_guard_refreshes_dns_cache_and_rejects_nonpublic_hosts():
    from types import SimpleNamespace
    from aletheia.browser import BrowserEnvironment

    class Route:
        def __init__(self, url, method="GET"):
            self.request = SimpleNamespace(url=url, method=method)
            self.aborted = False
            self.continued = False
        def abort(self):
            self.aborted = True
        def continue_(self):
            self.continued = True

    cache = {"example.org": (9999999999, True, "93.184.216.34")}
    calls = []
    browser = object.__new__(BrowserEnvironment)
    browser.allow_form_submit = False
    browser.allow_authentication = False
    browser.allow_domains = {"example.org"}
    browser.egress_proxy_url = "http://127.0.0.1:3128"
    browser.egress_proxy_policy_confirmed = True
    browser.web = SimpleNamespace(_public_host_cache=cache, _host_is_public=lambda host: calls.append(host) or False)
    route = Route("https://example.org/page")
    browser._guard_request(route)
    assert route.aborted and not route.continued
    assert calls == ["example.org"]
    assert "example.org" not in cache


def test_source_validation_rejects_path_traversal_before_materialization(tmp_path):
    from aletheia.evolution import EvolutionLab

    lab = EvolutionLab(tmp_path)
    candidate = lab.stage_source_patch("aletheia/learning_adapter.py", "value = 2\n")
    with pytest.raises(ValueError):
        lab.validate_source_patch(candidate, "../outside.py", run_tests=False)
    assert not (tmp_path.parent / "outside.py").exists()


def test_source_commit_rejects_candidate_modified_after_validation(tmp_path):
    from aletheia.evolution import EvolutionLab

    lab = EvolutionLab(tmp_path)
    target = tmp_path / "aletheia" / "learning_adapter.py"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("value = 1\n", encoding="utf-8")
    candidate = lab.stage_source_patch("aletheia/learning_adapter.py", "value = 2\n")
    validation = lab.validate_source_patch(candidate, "aletheia/learning_adapter.py", run_tests=False)
    assert validation["validated"]
    candidate.write_text("value = 999\n", encoding="utf-8")
    with pytest.raises(ValueError, match="changed after validation"):
        lab.commit_source_patch(candidate, "aletheia/learning_adapter.py", validation)
    assert target.read_text(encoding="utf-8") == "value = 1\n"


def test_source_commit_rejects_target_changed_after_validation(tmp_path):
    from aletheia.evolution import EvolutionLab

    lab = EvolutionLab(tmp_path)
    target = tmp_path / "aletheia" / "learning_adapter.py"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("value = 1\n", encoding="utf-8")
    candidate = lab.stage_source_patch("aletheia/learning_adapter.py", "value = 2\n")
    validation = lab.validate_source_patch(candidate, "aletheia/learning_adapter.py", run_tests=False)
    assert validation["validated"]
    target.write_text("value = 77\n", encoding="utf-8")
    with pytest.raises(ValueError, match="active source changed"):
        lab.commit_source_patch(candidate, "aletheia/learning_adapter.py", validation)
    assert target.read_text(encoding="utf-8") == "value = 77\n"


def test_grouped_dataset_split_never_leaks_groups_and_is_repeatable():
    from aletheia.data_splits import split_examples

    items = [
        {"id": f"g{group}-{idx}", "group": f"session-{group}"}
        for group in range(8) for idx in range(3)
    ]
    key = lambda item: item["group"]
    first = split_examples(items, holdout_fraction=0.25, seed=19, group_key=key)
    second = split_examples(items, holdout_fraction=0.25, seed=19, group_key=key)
    train_groups = {item["group"] for item in first.train}
    holdout_groups = {item["group"] for item in first.holdout}
    assert first.strategy == "grouped_metadata"
    assert train_groups.isdisjoint(holdout_groups)
    assert [item["id"] for item in first.train] == [item["id"] for item in second.train]
    assert [item["id"] for item in first.holdout] == [item["id"] for item in second.holdout]
    assert first.train and first.holdout


def test_grouped_dataset_split_reports_weaker_fallback_without_group_metadata():
    from aletheia.data_splits import split_examples

    split = split_examples([1, 2, 3, 4, 5], holdout_fraction=0.4, seed=7, group_key=lambda _: "")
    assert split.strategy == "example_level_no_usable_group_metadata"
    assert split.train and split.holdout


def test_evaluation_builder_writes_source_separated_fixed_holdout_manifest(monkeypatch, tmp_path):
    import json
    import aletheia.eval_corpus as ec
    from aletheia.evaluation_protocol import validate_source_separated_manifest

    paragraphs = []
    for i in range(200):
        paragraphs.append(
            f"Record {i} covers a distinct classroom, river, measurement, conversation, and journey. "
            + (f"The observations for record {i} remain tied to their source and are independently described. " * 3)
        )
    raw_text = "\n\n".join(paragraphs)
    monkeypatch.setattr(ec, "_fetch", lambda _url: raw_text)
    result = ec.build_reference(
        tmp_path / "evaluation" / "heldout.txt", target_bytes=50000,
        sources=[("first", "https://example.org/one"), ("second", "https://example.org/two")],
    )
    manifest_path = tmp_path / "evaluation" / "heldout.paragraphs.json"
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    records = payload["paragraphs"]
    assert payload["schema_version"] == 3
    assert payload["dataset_split"] == "fixed-heldout"
    assert records and all(row["text"] and row["source_id"] for row in records)
    assert all(row["split"] == "fixed-heldout" for row in records)
    assert all(row["memory_replay_allowed"] is False for row in records)
    assert all(row["tokenizer_training_allowed"] is False for row in records)
    assert validate_source_separated_manifest(records) == []
    assert result["bytes"] == len((tmp_path / "evaluation" / "heldout.txt").read_bytes())
    source_manifest = json.loads((tmp_path / "evaluation" / "heldout.sources.json").read_text(encoding="utf-8"))
    assert source_manifest["schema_version"] == 3
    assert len(source_manifest["sources"]) == 2
    assert all(row["selected_content_sha256"] and row["retrieved_at_utc"] for row in source_manifest["sources"])
    assert all(row["memory_replay_allowed"] is False and row["tokenizer_training_allowed"] is False for row in source_manifest["sources"])


def test_source_metrics_use_only_the_scored_tail_and_report_document_uncertainty(tmp_path):
    import json
    from types import SimpleNamespace
    import torch
    from aletheia.agent import AletheiaAgent

    parts = [
        {"item_id": "1", "source_id": "book-a", "title": "Book A", "text": "A" * 100},
        {"item_id": "2", "source_id": "book-b", "title": "Book B", "text": "B" * 100},
    ]
    raw = (parts[0]["text"] + "\n\n" + parts[1]["text"]).encode()
    evaluation = tmp_path / "evaluation"
    evaluation.mkdir()
    (evaluation / "heldout.txt").write_bytes(raw)
    (evaluation / "heldout.paragraphs.json").write_text(
        json.dumps({"schema_version": 3, "paragraphs": parts}), encoding="utf-8"
    )

    class FakeModel:
        cfg = SimpleNamespace(context=256)
        def eval(self):
            return self
        def language_bpb_totals(self, tokens):
            count = max(0, int(tokens.shape[1]) - 2)
            return float(count), float(count)

    agent = object.__new__(AletheiaAgent)
    agent.root = tmp_path
    agent.config = SimpleNamespace(runtime=SimpleNamespace(
        language_eval_path="evaluation/heldout.txt", evaluation_bootstrap_samples=50, evaluation_seed=4
    ))
    agent.model = FakeModel()
    agent._encode_byte_chunks = lambda data, max_chunks=0, context=None: [
        torch.tensor([[257, *data, 258]], dtype=torch.long)
    ]

    metrics = agent._fixed_reference_source_metrics(raw, 20)
    assert metrics["status"] == "scored"
    assert metrics["documents"] == 2
    assert metrics["bytes"] < len(raw)
    assert metrics["ci"] is not None
    assert metrics["interpretation"].startswith("next-byte prediction only")


def _make_small_agent(tmp_path):
    from pathlib import Path
    from aletheia.agent import AletheiaAgent
    from aletheia.config import load_config

    cfg = load_config(Path(__file__).parents[1] / "config.json")
    cfg.runtime.device = "cpu"
    cfg.model.dim = 32
    cfg.model.layers = 1
    cfg.model.heads = 4
    cfg.model.ffn = 64
    cfg.model.recurrent_dim = 48
    cfg.model.context = 96
    cfg.model.world_model_hidden = 64
    cfg.model.world_attention_heads = 4
    cfg.model.gradient_checkpointing = False
    cfg.runtime.browser_enabled = False
    return AletheiaAgent(tmp_path, cfg)


def test_replay_filters_legacy_memory_that_overlaps_fixed_evaluation(tmp_path):
    eval_dir = tmp_path / "evaluation"
    eval_dir.mkdir()
    heldout = (
        "Fixed heldout paragraph phrase unique gatekeeper alpha about rainfall, travel, "
        "learning, and measurements; it must never leak into training or replay. " * 4
    )
    (eval_dir / "heldout.txt").write_text(heldout, encoding="utf-8")
    safe_text = (
        "Independent safe learning material explains how a laboratory records observations, "
        "compares repeated measurements, and checks uncertainty before drawing a conclusion. " * 4
    )
    agent = _make_small_agent(tmp_path)
    try:
        train, holdout = agent._chunks_for_memory_row({
            "text": heldout + "\n\n" + safe_text,
            "train_context": agent.model.cfg.context,
            "holdout_indices": "[]",
        })
        decoded = "\n".join(agent.tokenizer.decode(chunk.flatten().tolist()) for chunk in (train + holdout))
        assert "gatekeeper alpha" not in decoded
        assert "Independent safe learning material" in decoded
        only_eval_train, only_eval_holdout = agent._chunks_for_memory_row({"text": heldout, "train_context": 96})
        assert only_eval_train == [] and only_eval_holdout == []
    finally:
        agent.close()


def test_tokenizer_training_excludes_evaluation_assets_and_near_copies(tmp_path, monkeypatch):
    from aletheia.tokenizer import LearnedBPETokenizer

    evaluation = tmp_path / "evaluation"
    evaluation.mkdir()
    heldout = (
        "Fixed heldout tokenizer phrase unique gatekeeper beta about rainfall, travel, "
        "learning, and measurements; tokenizer merges must not see these bytes. " * 4
    )
    (evaluation / "heldout.txt").write_text(heldout, encoding="utf-8")
    inputs = tmp_path / "incoming"
    inputs.mkdir()
    (inputs / "combined.md").write_text(
        heldout + "\n\n" + ("Independent tokenizer training material discusses plants, mathematics, and astronomy. " * 8),
        encoding="utf-8",
    )
    agent = _make_small_agent(tmp_path)
    captured = {}

    class FakeTokenizer:
        name = "fake-bpe"
        vocab_size = 260
        def state(self):
            return {"name": self.name, "vocab_size": self.vocab_size}
        def save(self, _path):
            return None

    def fake_train(texts, **kwargs):
        captured["texts"] = list(texts)
        return FakeTokenizer(), {"trained_examples": len(texts)}

    monkeypatch.setattr(LearnedBPETokenizer, "train", staticmethod(fake_train))
    monkeypatch.setattr(agent, "_replace_language_tokenizer", lambda _tok: {"kind": "fake-bpe", "vocab_size": 260, "fingerprint": "fake"})
    monkeypatch.setattr(agent, "save", lambda *args, **kwargs: None)
    try:
        result = agent.train_tokenizer(inputs)
        combined = "\n".join(captured["texts"])
        assert "gatekeeper beta" not in combined
        assert "Independent tokenizer training material" in combined
        with pytest.raises(ValueError, match="fixed evaluation data"):
            agent.train_tokenizer(evaluation / "heldout.txt")
    finally:
        agent.close()


def test_memory_retrieval_filters_legacy_evaluation_text(tmp_path, monkeypatch):
    import torch

    evaluation = tmp_path / "evaluation"
    evaluation.mkdir()
    heldout = (
        "Fixed heldout retrieval phrase unique gatekeeper gamma about rivers, classrooms, "
        "weather, and measurement, which must be excluded from evidence. " * 4
    )
    (evaluation / "heldout.txt").write_text(heldout, encoding="utf-8")
    safe_text = "Independent safe source discusses galaxies, telescopes, and repeated measurements. " * 5
    agent = _make_small_agent(tmp_path)
    try:
        monkeypatch.setattr(agent, "_informative_terms", lambda _message: ["independent", "safe"])
        monkeypatch.setattr(agent, "_text_representation", lambda _message: torch.ones(4))
        monkeypatch.setattr(agent.memory, "search_chunks", lambda *_args, **_kwargs: [
            {"id": 42, "kind": "lesson", "title": "legacy mixed row", "url": "", "source": "old", "text": heldout + "\n\n" + safe_text}
        ])
        results = agent._retrieve_experience("independent safe", limit=1)
        assert len(results) == 1
        assert "gatekeeper gamma" not in results[0]["text"]
        assert "Independent safe source" in results[0]["text"]
        assert results[0].get("evaluation_overlap_filtered") is True
    finally:
        agent.close()


def test_candidate_language_bpb_uses_active_tokenizer_and_same_heldout_tail(tmp_path):
    from aletheia.evolution import EvolutionLab

    script = EvolutionLab(tmp_path)._candidate_language_bpb_script()
    assert "from aletheia.tokenizer import PAD, load_tokenizer" in script
    assert "tokenizer.encode_bytes(eval_bytes, bos=False, eos=False)" in script
    assert "eval_bytes = raw[split - 1:]" in script
    assert "model.language_bpb_totals(tokens)" in script
    assert "nll_bits / byte_count" in script
    assert "sum(values) / max(1, len(values))" not in script
    assert "cfg.runtime.language_eval_max_chars" not in script


def test_candidate_tree_copies_only_configured_tokenizer_and_normalizes_path(tmp_path):
    import json
    from aletheia.evolution import EvolutionLab

    root = tmp_path / "project"
    root.mkdir()
    (root / "config.json").write_text(json.dumps({"runtime": {"tokenizer_path": "data/tokenizer.json"}}), encoding="utf-8")
    (root / "data").mkdir()
    tokenizer = root / "data" / "tokenizer.json"
    tokenizer.write_text(json.dumps({"schema_version": 2, "kind": "byte"}), encoding="utf-8")
    (root / "data" / "private-memory.json").write_text("secret", encoding="utf-8")
    (root / "aletheia").mkdir()
    (root / "aletheia" / "model.py").write_text("x = 1\n", encoding="utf-8")
    lab = EvolutionLab(root)
    candidate_root = lab.stage_root / "candidate-test"
    lab._copy_tree(candidate_root)
    assert (candidate_root / "data" / "tokenizer.json").read_bytes() == tokenizer.read_bytes()
    assert not (candidate_root / "data" / "private-memory.json").exists()
    candidate_cfg = json.loads((candidate_root / "config.json").read_text(encoding="utf-8"))
    assert candidate_cfg["runtime"]["tokenizer_path"] == "data/tokenizer.json"


@pytest.mark.parametrize(
    "config_text",
    [None, "not-json", '{"runtime": {}}', '{"runtime": {"evolution_self_edit_allowlist": []}}',
     '{"runtime": {"evolution_self_edit_allowlist": ["../outside.py"]}}'],
)
def test_source_edit_allowlist_fails_closed_when_missing_or_malformed(tmp_path, config_text):
    from aletheia.evolution import EvolutionLab

    if config_text is not None:
        (tmp_path / "config.json").write_text(config_text, encoding="utf-8")
    lab = EvolutionLab(tmp_path)
    assert lab.self_edit_allowlist == set()
    with pytest.raises(ValueError, match="outside the self-edit allowlist"):
        lab._editable_path("aletheia/learning_adapter.py", autonomous=True)


def test_source_edit_allowlist_can_only_narrow_built_in_safe_scope(tmp_path):
    import json
    from aletheia.evolution import EvolutionLab

    (tmp_path / "config.json").write_text(json.dumps({"runtime": {
        "evolution_self_edit_allowlist": ["aletheia/experimental.py", "aletheia/evolution.py"]
    }}), encoding="utf-8")
    lab = EvolutionLab(tmp_path)
    assert lab.self_edit_allowlist == {__import__("pathlib").Path("aletheia/experimental.py")}


def test_configured_evaluation_path_rejects_escape_from_project_root(tmp_path):
    from pathlib import Path
    import pytest
    from aletheia.agent import AletheiaAgent
    from aletheia.config import load_config

    cfg = load_config(Path(__file__).parents[1] / "config.json")
    cfg.runtime.language_eval_path = "../outside-heldout.txt"
    with pytest.raises(ValueError, match="remain inside"):
        AletheiaAgent(tmp_path, cfg)


def test_teach_directory_excludes_evaluation_corpus_and_sidecars_by_source_path(tmp_path, monkeypatch):
    from pathlib import Path
    from aletheia.agent import AletheiaAgent
    from aletheia.config import load_config

    input_root = tmp_path / "input"
    evaluation_dir = input_root / "evaluation"
    evaluation_dir.mkdir(parents=True)
    (evaluation_dir / "heldout.txt").write_text(
        "UNIQUE_HELDOUT_SECRET: content that must never enter training memory or weights.\n" * 4,
        encoding="utf-8",
    )
    (evaluation_dir / "heldout.sources.json").write_text(
        '{"source":"UNIQUE_HELDOUT_SECRET", "heldout": true}', encoding="utf-8"
    )
    (evaluation_dir / "heldout.paragraphs.json").write_text(
        '{"text":"UNIQUE_HELDOUT_SECRET: content that must never enter training memory or weights."}',
        encoding="utf-8",
    )
    (input_root / "lesson.txt").write_text(
        "SAFE_TRAINING_TEXT: a separate ordinary lesson for the learner.\n" * 4,
        encoding="utf-8",
    )

    cfg = load_config(Path(__file__).parents[1] / "config.json")
    cfg.runtime.device = "cpu"
    cfg.runtime.language_eval_path = "input/evaluation/heldout.txt"
    cfg.model.dim = 32
    cfg.model.layers = 1
    cfg.model.heads = 4
    cfg.model.ffn = 64
    cfg.model.recurrent_dim = 48
    cfg.model.context = 96
    cfg.model.world_model_hidden = 64
    cfg.model.world_attention_heads = 4
    cfg.model.gradient_checkpointing = False
    cfg.runtime.browser_enabled = False
    agent = AletheiaAgent(tmp_path, cfg)

    captured = []
    monkeypatch.setattr(agent, "teach_text", lambda lesson, **kwargs: captured.append(lesson.text) or {"ok": True})
    results = agent.teach_file(input_root, epochs=1)

    taught = "\n".join(captured)
    assert results
    assert "SAFE_TRAINING_TEXT" in taught
    assert "UNIQUE_HELDOUT_SECRET" not in taught
    assert all("heldout" not in row.lower() for row in captured)
    pretrain_shards, _ = agent._load_pretrain_shards(input_root)
    pretrain_text = "\n".join(shard.text for shard in pretrain_shards)
    assert "SAFE_TRAINING_TEXT" in pretrain_text
    assert "UNIQUE_HELDOUT_SECRET" not in pretrain_text


def test_root_level_evaluation_path_does_not_exclude_all_tokenizer_training_sources(tmp_path):
    from pathlib import Path
    from aletheia.agent import AletheiaAgent
    from aletheia.config import load_config

    cfg = load_config(Path(__file__).parents[1] / "config.json")
    cfg.runtime.device = "cpu"
    cfg.runtime.language_eval_path = "heldout.txt"
    cfg.model.dim = 32
    cfg.model.layers = 1
    cfg.model.heads = 4
    cfg.model.ffn = 64
    cfg.model.recurrent_dim = 48
    cfg.model.context = 96
    cfg.model.world_model_hidden = 64
    cfg.model.world_attention_heads = 4
    cfg.model.gradient_checkpointing = False
    cfg.runtime.browser_enabled = False
    (tmp_path / "heldout.txt").write_text("fixed evaluation data that should not train tokenizer", encoding="utf-8")
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    (source_dir / "lesson.txt").write_text("ordinary independent lesson text for tokenizer training", encoding="utf-8")
    agent = AletheiaAgent(tmp_path, cfg)

    # The root can still be a source when the fixed evaluation file is configured at root.
    # We only exercise discovery/filtering, not the expensive model/tokenizer replacement.
    evaluation_path = agent._evaluation_path()
    excluded = {evaluation_path, evaluation_path.with_suffix(".sources.json"), evaluation_path.with_suffix(".paragraphs.json")}
    assert (tmp_path / "lesson.txt").resolve() not in excluded
    assert agent._is_fixed_evaluation_source(tmp_path / "heldout.txt")
    assert not agent._is_fixed_evaluation_source(source_dir / "lesson.txt")
    result = agent.train_tokenizer(source=source_dir, vocab_size=260, min_frequency=1, max_training_bytes=1000)
    assert result["training_texts"] == 1


def test_browser_external_navigation_fails_closed_without_policy_proxy(tmp_path):
    from aletheia.browser import BrowserEnvironment, BrowserEnvironmentError

    class FakeWeb:
        class Config:
            allowed_schemes = ["http", "https"]
        config = Config()
        normalize_url = staticmethod(lambda url, base_url=None, allowed_schemes=None: url if str(url).startswith(("http://", "https://")) else None)
        _host_is_public = staticmethod(lambda host: True)
        _robots = staticmethod(lambda url: True)

    browser = BrowserEnvironment(tmp_path, FakeWeb(), allow_domains=["example.org"])
    with pytest.raises(BrowserEnvironmentError, match="policy-enforcing egress proxy"):
        browser._validate_url("https://example.org/page")


def test_browser_empty_domain_allowlist_denies_every_remote_host(tmp_path):
    from aletheia.browser import BrowserEnvironment

    class FakeWeb:
        class Config:
            allowed_schemes = ["http", "https"]
        config = Config()
        normalize_url = staticmethod(lambda url, base_url=None, allowed_schemes=None: url if str(url).startswith(("http://", "https://")) else None)
        _host_is_public = staticmethod(lambda host: True)

    browser = BrowserEnvironment(tmp_path, FakeWeb())
    assert browser._domain_allowed("example.org") is False


def test_capability_report_marks_enabled_browser_without_proxy_as_failed(tmp_path):
    import json
    from aletheia.capability_report import build_capability_report

    (tmp_path / "config.json").write_text(json.dumps({"runtime": {
        "browser_enabled": True,
        "browser_allow_domains": ["example.org"],
        "browser_egress_proxy_url": "",
        "browser_egress_proxy_policy_confirmed": False,
    }}), encoding="utf-8")
    report = build_capability_report(tmp_path)
    browser = next(item for item in report["capabilities"] if item["name"] == "browser")
    assert browser["status"] == "failed"
    assert browser["browser_enabled_in_config"] is True
    assert browser["egress_proxy_configured"] is False
    assert "fail closed" in browser["details"]
