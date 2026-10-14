from __future__ import annotations

from pathlib import Path

import torch

from aletheia.agent import AletheiaAgent
from aletheia.config import load_config
from aletheia.model import AletheiaModel
from aletheia.outcomes import OpenWorldOutcomeEngine, OpenWorldSignals
from aletheia.objective_value import ObjectiveOutcomeLearner
from aletheia.tokenizer import BASE_VOCAB_SIZE, LearnedBPETokenizer


def _small_config():
    cfg = load_config(Path("config.json"))
    cfg.runtime.device = "cpu"
    cfg.model.dim = 32
    cfg.model.layers = 1
    cfg.model.heads = 4
    cfg.model.ffn = 64
    cfg.model.recurrent_dim = 48
    cfg.model.context = 64
    cfg.model.gradient_checkpointing = False
    cfg.runtime.teacher_holdout_fraction = 0.25
    cfg.runtime.minimum_bootstrap_memories = 0
    cfg.runtime.objective_value_min_examples = 2
    cfg.runtime.objective_value_batch_size = 2
    cfg.runtime.objective_value_training_steps = 1
    cfg.runtime.objective_value_buffer_size = 16
    return cfg


def test_learned_bpe_roundtrip_and_compression():
    text = ("communication requires context, repetition, and meaningful structure. " * 120)
    tokenizer, stats = LearnedBPETokenizer.train([text], vocab_size=400, min_frequency=2, max_training_bytes=100000)
    encoded = tokenizer.encode(text, bos=False, eos=False)
    assert tokenizer.vocab_size >= BASE_VOCAB_SIZE
    assert tokenizer.decode(encoded) == text
    assert len(encoded) < len(text.encode("utf-8"))
    assert stats["merges"] == tokenizer.vocab_size - BASE_VOCAB_SIZE


def test_model_checkpoint_migrates_byte_vocab_to_learned_tokenizer(tmp_path):
    cfg = _small_config()
    action_names = ["learn", "act"]
    old = AletheiaModel(cfg.model, action_names)
    optimizer = torch.optim.AdamW(old.parameters(), lr=1e-3)
    path = tmp_path / "model.pt"
    old.save(path, optimizer)
    tok, _ = LearnedBPETokenizer.train(["this is a repeated learned token corpus " * 40], vocab_size=300, min_frequency=2)
    cfg.model.vocab_size = tok.vocab_size
    optimizer2 = torch.optim.AdamW(AletheiaModel(cfg.model, action_names, tokenizer=tok).parameters(), lr=1e-3)
    migrated = AletheiaModel.load(path, torch.device("cpu"), cfg.model, action_names, optimizer2, tokenizer=tok)
    assert migrated.cfg.vocab_size == tok.vocab_size
    assert migrated.tokenizer.vocab_size == tok.vocab_size
    assert torch.isfinite(migrated.token_embedding.weight).all()


def test_open_world_signal_aggregator_is_parameter_free():
    signals = OpenWorldSignals(0.2, 0.8, None, 0.5)
    assert signals.aggregate() == 0.5
    assert OpenWorldOutcomeEngine.similarity(torch.ones(4), torch.ones(4)) == 1.0


def test_multiline_teaching_preserves_internal_newlines(tmp_path, monkeypatch):
    cfg = _small_config()
    agent = AletheiaAgent(tmp_path, cfg)
    try:
        seen = {}
        def fake_teach(lesson, **kwargs):
            seen["text"] = lesson.text
            return {"title": lesson.title, "chars": len(lesson.text), "chunks": 1, "loss_before": 0.0, "loss_after": 0.0, "learning_progress": 0.0, "holdout_bpb_after": 0.0, "fixed_eval_gain": 0.0, "lesson_accepted": True, "aggregate_improvement": 0.0, "replay_sequences_used": 0, "replay_lessons_used": 0, "dialogue_examples": 0}
        monkeypatch.setattr(agent, "teach_text", fake_teach)
        pasted = "first line\n\nsecond line with unicode: नमस्ते\nthird line"
        result = agent.teach_text_exact(pasted)
        assert result["chars"] == len(pasted)
        assert seen["text"] == pasted
    finally:
        agent.close()


def test_tokenizer_training_installs_and_persists_learned_vocab(tmp_path):
    cfg = _small_config()
    cfg.runtime.tokenizer_vocab_size = 320
    cfg.runtime.tokenizer_train_max_bytes = 50000
    source = tmp_path / "corpus.txt"
    source.write_text(("a learner sees repeated patterns and learns structure. " * 100), encoding="utf-8")
    agent = AletheiaAgent(tmp_path, cfg)
    try:
        result = agent.train_tokenizer(source)
        assert result["vocab_size"] > BASE_VOCAB_SIZE
        assert agent.tokenizer.name == "bpe"
        assert agent.paths.tokenizer.exists()
        assert agent.model.cfg.vocab_size == agent.tokenizer.vocab_size
        assert agent.total_tokens == 0
    finally:
        agent.close()


def test_agent_open_world_outcome_updates_from_capability_result(tmp_path):
    cfg = _small_config()
    agent = AletheiaAgent(tmp_path, cfg)
    try:
        agent.set_objective("an arbitrary open world objective")
        agent._last_action_state = agent.state.detach().clone()
        agent._last_action_state_after = agent.state.detach().clone()
        agent._last_action_index = 0
        agent._last_action_selection_mode = "planned"
        agent.objectives.current().progress = 0.0
        called = {"n": 0}
        original = agent.record_objective_outcome
        def wrapped(value, **kwargs):
            called["n"] += 1
            return original(value, **kwargs)
        agent.record_objective_outcome = wrapped
        agent._choose_action = lambda: (0, "planned", 0.0, torch.ones(len(agent.registry.items()), dtype=torch.bool))
        from dataclasses import replace
        existing = agent.registry.get(0)
        agent.registry._items[0] = replace(existing, run=lambda: (0.73, {"kind": "test", "valid_transition": True, "objective_outcome": 0.73}))
        event = agent.autonomous_cycle()
        assert called["n"] == 1
        assert event["signal"] == 0.73
        assert event["outcome_available"] is False
        assert event["unverified_outcome_candidate_available"] is True
    finally:
        agent.close()


def test_bpe_language_metric_is_bits_per_utf8_byte(tmp_path):
    cfg = _small_config()
    text = "token repetition makes BPE efficient: communication communication communication. " * 8
    tok, _ = LearnedBPETokenizer.train([text], vocab_size=320, min_frequency=2, max_training_bytes=50000)
    cfg.model.vocab_size = tok.vocab_size
    model = AletheiaModel(cfg.model, ["observe"], tokenizer=tok)
    token_ids = tok.encode(text, bos=False, eos=False)
    chunks = []
    payload = cfg.model.context - 2
    for start in range(0, len(token_ids), payload):
        chunks.append(torch.tensor([[257, *token_ids[start:start + payload], 258]], dtype=torch.long))
    metric = sum(model.language_bpb(chunk) for chunk in chunks) / len(chunks)
    assert 0.0 < metric < 12.0
    represented_bytes = sum(tok.token_byte_lengths(token_ids))
    assert represented_bytes == len(text.encode("utf-8"))


def test_objective_outcome_can_skip_per_example_disk_flush(tmp_path):
    learner = ObjectiveOutcomeLearner(tmp_path, torch.device("cpu"), state_dim=4, action_dim=3, hidden=12, min_examples=10, batch_size=2, buffer_size=8)
    state = torch.randn(4); goal = torch.randn(4); action = torch.randn(3)
    learner.record(state, state, goal, action, 0.5, objective_id="obj", persist=False)
    assert learner.stats.examples == 1
    assert not learner.path.exists()
