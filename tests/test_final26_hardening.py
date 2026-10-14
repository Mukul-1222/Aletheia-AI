from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from aletheia.evolution import EvolutionLab
from aletheia.objective_verifiers import ObjectiveVerifier
from aletheia.objectives import ObjectiveManager
from aletheia.config import load_config
from aletheia.model import AletheiaModel
from aletheia.tokenizer import ByteFallbackTokenizer


def test_patch_parser_rejects_multi_file_and_traversal(tmp_path):
    lab = EvolutionLab(tmp_path)
    assert not lab._validate_patch_text(
        """--- a/aletheia/learning_adapter.py\n+++ b/aletheia/learning_adapter.py\n@@\n-x\n+y\n--- a/aletheia/experimental.py\n+++ b/aletheia/experimental.py\n@@\n-x\n+y\n""",
        "aletheia/learning_adapter.py",
    )[0]
    assert not lab._validate_patch_text(
        """--- a/aletheia/learning_adapter.py\n+++ b/../config.json\n@@\n-x\n+y\n""",
        "aletheia/learning_adapter.py",
    )[0]


def test_objective_verifiers_support_real_contracts():
    t = ObjectiveVerifier.evaluate("threshold", 0.97, threshold=0.95)
    assert t.verified and t.complete
    a = ObjectiveVerifier.evaluate("answer", 0.0, evidence={"actual": "Paris", "expected": "Paris"})
    assert a.verified and a.complete and a.score == 1.0
    u = ObjectiveVerifier.evaluate("unit_test", 0.0, evidence={"passed": True})
    assert u.verified and u.complete


def test_objective_consecutive_verification_is_really_consecutive(tmp_path):
    manager = ObjectiveManager(tmp_path / "objectives.json")
    objective = manager.add("do task", verifier_type="threshold", completion_threshold=0.8, required_consecutive=2)
    first = manager.verify(objective.id, outcome=0.9)
    assert first["verified"] and not first["complete"]
    failed = manager.verify(objective.id, outcome=0.1)
    assert not failed["verified"] and failed["verification_count"] == 1
    second = manager.verify(objective.id, outcome=0.95)
    assert second["verified"] and not second["complete"]
    final = manager.verify(objective.id, outcome=0.99)
    assert final["verified"] and final["complete"]


def test_self_edit_allowlist_fails_closed(tmp_path):
    (tmp_path / "config.json").write_text(json.dumps({"runtime": {"evolution_self_edit_allowlist": ["aletheia/learning_adapter.py", "aletheia/../web.py"]}}), encoding="utf-8")
    lab = EvolutionLab(tmp_path)
    assert lab.self_edit_allowlist == {Path("aletheia/learning_adapter.py")}
    with pytest.raises(ValueError):
        lab._editable_path("aletheia/web.py", autonomous=True)


def test_focused_source_context_is_patchable_source_not_numbered_text():
    source = "def keep():\n    return 1\n\ndef target():\n    return 2\n"
    excerpt = EvolutionLab._focused_source_context(source, "change target", 200)
    assert "0001:" not in excerpt
    assert "def target():" in excerpt


def test_config_rejects_unknown_runtime_keys(tmp_path):
    payload = json.loads(Path("config.json").read_text(encoding="utf-8"))
    payload["runtime"]["this_key_does_not_exist"] = 1
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError):
        load_config(path)


def test_browser_hardening_defaults_are_safe(tmp_path):
    cfg = load_config(Path("config.json"))
    assert cfg.runtime.browser_persistent_profile is False
    assert cfg.runtime.browser_allow_form_submit is False
    assert cfg.runtime.browser_allow_downloads is False
    assert cfg.runtime.browser_allow_file_uploads is False
    assert cfg.runtime.browser_allow_authentication is False
    assert cfg.runtime.evolution_require_hard_sandbox is True


def test_cross_modal_alignment_loss_is_real_and_trainable():
    cfg = load_config(Path("config.json")).model
    cfg.dim = 32; cfg.layers = 1; cfg.heads = 4; cfg.ffn = 64; cfg.recurrent_dim = 48
    cfg.context = 64; cfg.perception_dim = 16; cfg.vocab_size = ByteFallbackTokenizer().vocab_size
    model = AletheiaModel(cfg, ["observe"], tokenizer=ByteFallbackTokenizer())
    text = torch.randint(4, cfg.vocab_size, (4, 12))
    vision = torch.randn(4, cfg.perception_dim)
    audio = torch.randn(4, cfg.perception_dim)
    loss = model.cross_modal_alignment_loss(text, vision_embeddings=vision, audio_embeddings=audio)
    assert torch.isfinite(loss)
    loss.backward()
    assert model.text_to_world.weight.grad is not None


def test_anchor_is_reset_when_evaluation_fingerprint_changes(tmp_path):
    from aletheia.agent import AletheiaAgent
    agent = object.__new__(AletheiaAgent)
    agent.root = tmp_path
    agent.evaluate_fixed_reference = lambda: {
        "available": True, "fingerprint": "new", "model_bpb": 4.0, "baseline_bpb": 3.0
    }
    anchor = tmp_path / "data" / "web_learning_anchor.json"
    anchor.parent.mkdir(parents=True, exist_ok=True)
    anchor.write_text(json.dumps({"reference_fingerprint": "old", "anchor_model_bpb": 99.0, "anchor_baseline_bpb": 3.0}), encoding="utf-8")
    payload = agent._web_learning_anchor()
    assert payload["reference_fingerprint"] == "new"
    assert payload["anchor_model_bpb"] == 4.0


def test_autonomy_kill_switch_and_budget_count_only_started_entries(tmp_path):
    from aletheia.agent import AletheiaAgent
    runtime = SimpleNamespace(
        autonomy_kill_switch_path="data/STOP",
        autonomy_max_external_actions_per_hour=2,
        autonomy_action_ledger_path="data/actions.jsonl",
        autonomy_external_action_budget_enabled=True,
    )
    agent = object.__new__(AletheiaAgent)
    agent.root = tmp_path
    agent.config = SimpleNamespace(runtime=runtime)
    ok, reason, count = agent._autonomy_guard()
    assert ok and reason == "ok" and count == 0
    agent._ledger_external_action(capability="search", status="started", cycle=1)
    agent._ledger_external_action(capability="search", status="finished", cycle=1)
    agent._ledger_external_action(capability="search", status="started", cycle=2)
    ok, reason, count = agent._autonomy_guard()
    assert not ok and reason == "external_action_hourly_budget_exhausted" and count == 2
    (tmp_path / "data" / "STOP").write_text("stop", encoding="utf-8")
    ok, reason, _ = agent._autonomy_guard()
    assert not ok and reason == "kill_switch_present"


def test_learned_bpe_merge_training_is_deterministic():
    from aletheia.tokenizer import LearnedBPETokenizer
    corpus = ["abababab cababab", "abab cdcdcd", "日本語日本語", "ไทยไทยไทย"]
    one, stats_one = LearnedBPETokenizer.train(corpus, vocab_size=300, min_frequency=2, max_training_bytes=10000)
    two, stats_two = LearnedBPETokenizer.train(corpus, vocab_size=300, min_frequency=2, max_training_bytes=10000)
    assert one.merges == two.merges
    assert stats_one == stats_two



def test_hard_sandbox_never_exposes_host_root(monkeypatch, tmp_path):
    from aletheia.evolution import EvolutionLab
    monkeypatch.setattr("aletheia.evolution.shutil.which", lambda name: "/usr/bin/bwrap" if name == "bwrap" else None)
    prefix = EvolutionLab._hard_sandbox_prefix(tmp_path)
    assert prefix is not None
    joined = " ".join(prefix)
    assert "--ro-bind / /" not in joined
    assert "--unshare-net" in prefix
    assert "/work" in prefix
