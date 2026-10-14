from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from aletheia.agent import AletheiaAgent
from aletheia.config import ModelConfig, load_config
from aletheia.evolution import EvolutionLab
from aletheia.model import AletheiaModel
from aletheia.objectives import ObjectiveManager
from aletheia.perception import AudioEncoder, PerceptualIdentityMemory, VisionEncoder, PerceptionSystem


def small_config():
    cfg = load_config(Path("config.json"))
    cfg.runtime.device = "cpu"
    cfg.runtime.memory_limit = 40
    cfg.runtime.max_frontier = 40
    cfg.runtime.language_ready_min_tokens = 0
    cfg.runtime.language_ready_min_lessons = 0
    cfg.runtime.language_ready_min_bpb_gain = 0
    cfg.runtime.language_ready_min_eval_bytes = 0
    cfg.model.dim = 32
    cfg.model.layers = 1
    cfg.model.heads = 4
    cfg.model.ffn = 64
    cfg.model.recurrent_dim = 48
    cfg.model.context = 96
    cfg.model.dropout = 0.0
    cfg.model.perception_dim = 256
    return cfg


def test_objective_persistence_and_gap_inference(tmp_path):
    manager = ObjectiveManager(tmp_path / "objectives.json")
    obj = manager.add("learn human emotion and communication")
    gaps = manager.infer_gaps(obj, "communication is learned through interaction", limit=8)
    assert obj.id
    assert "emotion" in gaps
    manager2 = ObjectiveManager(tmp_path / "objectives.json")
    assert manager2.current() is not None
    assert manager2.current().text == obj.text


def test_visual_encoder_is_random_init_and_identity_clusters(tmp_path):
    encoder = VisionEncoder()
    assert sum(p.numel() for p in encoder.parameters()) > 0
    image = np.zeros((128, 128, 3), dtype=np.uint8)
    with torch.no_grad():
        embedding = encoder(encoder.prepare_image(image))[0]
    memory = PerceptualIdentityMemory(tmp_path / "identity.json", threshold=0.90)
    first = memory.observe(embedding, alias="observer")
    second = memory.observe(embedding, alias="observer")
    assert first["new"] is True
    assert second["new"] is False
    assert second["entity_id"] == first["entity_id"]
    assert "observer" in second["alias"]


def test_agent_objective_and_sensory_state(tmp_path):
    cfg = small_config()
    agent = AletheiaAgent(tmp_path, cfg)
    try:
        result = agent.set_objective("learn visual identity")
        assert result["text"] == "learn visual identity"
        before = agent.state.clone()
        image = np.zeros((96, 96, 3), dtype=np.uint8)
        obs = agent.perception.observe_image(image, alias="person")
        tensor = agent.perception.vision.prepare_image(image).to(agent.device)
        with torch.no_grad():
            embedding = agent.perception.vision(tensor)[0]
            agent.state = agent.model.integrate_perception(embedding, agent.state, "vision").detach()
        assert obs["entity_id"]
        assert not torch.equal(before, agent.state)
    finally:
        agent.close()


def test_checkpoint_can_expand_action_head(tmp_path):
    cfg = small_config()
    agent = AletheiaAgent(tmp_path, cfg)
    try:
        agent.save()
        model_path = agent.paths.model
        original_actions = list(agent.model.action_names)
    finally:
        agent.close()
    cfg2 = small_config()
    cfg2.capabilities["extra_action"] = type(next(iter(cfg2.capabilities.values())))(description="test")
    # The real registry rejects unknown implementations, so exercise the migration
    # primitive directly rather than constructing the full agent with an invalid handler.
    from aletheia.model import AletheiaModel
    optimizer = torch.optim.AdamW(AletheiaModel(cfg2.model, original_actions).parameters(), lr=1e-3)
    loaded = AletheiaModel.load(model_path, torch.device("cpu"), cfg2.model, original_actions + ["extra"], optimizer)
    assert loaded.action_names[-1] == "extra"

def test_evolution_lab_stays_project_local(tmp_path):
    lab = EvolutionLab(tmp_path)
    candidate = lab.stage_source_patch("aletheia/demo.py", "x = 1\n", rationale="test")
    result = lab.validate_source_patch(candidate, "aletheia/demo.py", run_tests=False)
    assert result["validated"] is True
    committed = lab.commit_source_patch(candidate, "aletheia/demo.py")
    assert committed["committed"] is True
    assert (tmp_path / "aletheia" / "demo.py").read_text(encoding="utf-8") == "x = 1\n"
    try:
        lab.stage_source_patch("../outside.py", "x=1")
    except ValueError:
        pass
    else:
        raise AssertionError("path escape was accepted")


def test_visual_encoder_can_learn_without_pretrained_weights(tmp_path):
    system = PerceptionSystem(tmp_path, torch.device("cpu"), embedding_dim=32)
    image = np.zeros((128, 128, 3), dtype=np.uint8)
    image[24:104, 32:96] = 180
    before = {k: v.detach().clone() for k, v in system.vision.state_dict().items()}
    result = system.train_visual_observations([image, image], steps=2)
    assert result["trained"] is False
    assert result["reason"] == "insufficient_distinct_images"
    distinct = []
    for i in range(32):
        item = image.copy()
        item[:, :, 0] = i * 7
        distinct.append(item)
    result = system.train_visual_observations(distinct, steps=2)
    assert result["trained"] is True
    assert system.vision_steps == 2
    assert any(not torch.equal(before[k], v) for k, v in system.vision.state_dict().items())


def test_objective_feedback_updates_state_and_policy_memory(tmp_path):
    cfg = small_config()
    agent = AletheiaAgent(tmp_path, cfg)
    try:
        agent.set_objective("learn visual identity")
        event = agent.autonomous_cycle()
        before = agent.objectives.current().progress
        feedback = agent.objective_feedback(True, "that helped")
        assert feedback["updated"] is True
        assert agent.objectives.current().progress > before
    finally:
        agent.close()


def test_self_edit_requires_full_validation_before_commit(tmp_path):
    lab = EvolutionLab(tmp_path)
    candidate = lab.stage_source_patch("aletheia/demo.py", "def broken(:\n    pass\n", rationale="invalid")
    validation = lab.validate_source_patch(candidate, "aletheia/demo.py", run_tests=False)
    assert validation["validated"] is False
    try:
        lab.commit_source_patch(candidate, "aletheia/demo.py", validation)
    except ValueError:
        pass
    else:
        raise AssertionError("invalid source was committed")


def test_self_edit_commit_keeps_backup(tmp_path):
    lab = EvolutionLab(tmp_path)
    target = tmp_path / "aletheia" / "demo.py"
    target.parent.mkdir(parents=True)
    target.write_text("x = 1\n", encoding="utf-8")
    candidate = lab.stage_source_patch("aletheia/demo.py", "x = 2\n", rationale="test")
    validation = lab.validate_source_patch(candidate, "aletheia/demo.py", run_tests=False)
    assert validation["validated"] is True
    result = lab.commit_source_patch(candidate, "aletheia/demo.py", validation)
    assert target.read_text(encoding="utf-8") == "x = 2\n"
    assert result["restart_required"] is True
    assert Path(result["backup"]).read_text(encoding="utf-8") == "x = 1\n"


def test_source_validation_discovers_current_test_files(tmp_path):
    lab = EvolutionLab(tmp_path)
    (tmp_path / "aletheia").mkdir(parents=True, exist_ok=True)
    (tmp_path / "aletheia" / "demo.py").write_text("value = 2\n", encoding="utf-8")
    (tmp_path / "tests").mkdir(parents=True, exist_ok=True)
    (tmp_path / "tests" / "test_final12_current.py").write_text("def test_current_tree_is_tested(): assert True\n", encoding="utf-8")
    candidate = lab.stage_source_patch("aletheia/demo.py", "value = 3\n")
    result = lab.validate_source_patch(candidate, "aletheia/demo.py", run_tests=True, timeout=30)
    assert result["tests_passed"] is True
    assert "tests/test_final12_current.py" in result["test_files"]


def test_source_evolution_protects_validator_and_recovery_modules(tmp_path):
    lab = EvolutionLab(tmp_path)
    for path in ("aletheia/evolution.py", "aletheia/__main__.py", "aletheia/evaluation.py", "aletheia/eval_corpus.py"):
        try:
            lab.stage_source_patch(path, "x = 1\n")
            # staging is harmless; the actual edit/commit entry points must reject it
            lab.commit_source_patch(lab.stage_root / path, path, {"validated": True})
        except ValueError:
            pass
        else:
            raise AssertionError(f"protected path was accepted: {path}")


def test_source_validation_rejects_fixed_reference_regression(tmp_path, monkeypatch):
    lab = EvolutionLab(tmp_path)
    (tmp_path / "aletheia").mkdir(parents=True, exist_ok=True)
    target = tmp_path / "aletheia" / "demo.py"
    target.write_text("value = 1\n", encoding="utf-8")
    candidate = lab.stage_source_patch("aletheia/demo.py", "value = 2\n")
    monkeypatch.setattr(lab, "_candidate_language_bpb", lambda *args, **kwargs: (1.25, "candidate bpb"))
    result = lab.validate_source_patch(candidate, "aletheia/demo.py", run_tests=False, active_bpb=1.0)
    assert result["validated"] is False
    assert result["bpb_gate"] == "rejected"


def test_model_refuses_unsafe_architecture_migration(tmp_path):
    old_cfg = ModelConfig(vocab_size=259, dim=16, layers=2, heads=4, ffn=32, recurrent_dim=24, context=32, perception_dim=8, dropout=0.0, gradient_checkpointing=False)
    new_cfg = ModelConfig(vocab_size=259, dim=24, layers=3, heads=4, ffn=48, recurrent_dim=32, context=48, perception_dim=8, dropout=0.0, gradient_checkpointing=False)
    actions = ["discover", "search", "evolve"]
    old = AletheiaModel(old_cfg, actions)
    old_opt = torch.optim.AdamW(old.parameters(), lr=1e-3)
    path = tmp_path / "model.pt"
    old.save(path, old_opt)
    new_opt = torch.optim.AdamW(AletheiaModel(new_cfg, actions).parameters(), lr=1e-3)
    try:
        AletheiaModel.load(path, torch.device("cpu"), new_cfg, actions, new_opt)
    except ValueError as exc:
        assert "architecture differs" in str(exc)
    else:
        raise AssertionError("unsafe architecture migration was accepted")


def test_agent_preserves_incompatible_checkpoint(tmp_path):
    old_cfg = small_config()
    old = AletheiaModel(old_cfg.model, ["discover", "search", "evolve"])
    old_opt = torch.optim.AdamW(old.parameters(), lr=1e-3)
    checkpoint = tmp_path / "data" / "aletheia.pt"
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    old.save(checkpoint, old_opt)
    cfg = small_config()
    cfg.model.dim = 40
    agent = AletheiaAgent(tmp_path, cfg)
    try:
        archived = list((tmp_path / "data").glob("incompatible_checkpoint_*.pt"))
        assert archived
        assert archived[0].exists()
    finally:
        agent.close()


def test_visual_ssl_keeps_cross_sample_variance(tmp_path):
    system = PerceptionSystem(tmp_path, torch.device("cpu"), embedding_dim=16, batch_size=32, buffer_size=64)
    images = []
    for i in range(32):
        image = np.zeros((128, 128, 3), dtype=np.uint8)
        image[:, :, 0] = (i * 7) % 256
        image[16 + (i % 8):96, 16:96, 1] = 80 + (i % 8) * 15
        image[(i % 32):96, (i * 3) % 64:((i * 3) % 64) + 32, 2] = 180
        images.append(image)
    result = system.train_visual_observations(images, steps=3)
    assert result["batch_size"] >= 32
    assert result["embedding_std_mean"] > 1e-3
    embeddings = []
    for image in images[:8]:
        tensor = VisionEncoder.prepare_image(image).to(system.device)
        with torch.no_grad():
            embeddings.append(system.vision(tensor)[0].cpu())
    matrix = torch.stack(embeddings)
    assert float(matrix.std(dim=0).mean()) > 1e-4


def test_wav_loader_accepts_multiple_pcm_widths(tmp_path):
    import wave
    for width in (1, 2, 3, 4):
        path = tmp_path / f"sample_{width}.wav"
        with wave.open(str(path), "wb") as handle:
            handle.setnchannels(1)
            handle.setsampwidth(width)
            handle.setframerate(16000)
            if width == 1:
                payload = bytes([128, 140, 116, 128])
            elif width == 2:
                payload = np.array([0, 2048, -2048, 0], dtype=np.int16).tobytes()
            elif width == 3:
                samples = [0, 1000, -1000, 0]
                raw = bytearray()
                for sample in samples:
                    if sample < 0:
                        sample += 1 << 24
                    raw.extend((sample & 0xFFFFFF).to_bytes(3, "little"))
                payload = bytes(raw)
            else:
                payload = np.array([0, 1_000_000, -1_000_000, 0], dtype=np.int32).tobytes()
            handle.writeframes(payload)
        waveform, rate = AudioEncoder.load_wav(path)
        assert rate == 16000
        assert waveform.numel() == 4


def test_chat_feedback_targets_latest_chat_response(tmp_path):
    cfg = small_config()
    agent = AletheiaAgent(tmp_path, cfg)
    try:
        agent.set_objective("learn communication")
        agent.communicate("hello there")
        feedback = agent.objective_feedback(True, "useful")
        assert feedback["target_kind"] == "chat"
        rows = agent.memory.search("dialogue feedback", 20)
        # The memory kind is the authoritative marker; search may not match the title text.
        assert any(row.get("kind") == "dialogue_feedback" for row in agent.memory.recent(50))
    finally:
        agent.close()


def test_feedback_works_without_objective_and_strips_response_tag(tmp_path):
    cfg = small_config()
    agent = AletheiaAgent(tmp_path, cfg)
    try:
        agent.communicate("hello")
        result = agent.objective_feedback(True, "good response")
        assert result["updated"] is True
        assert result["target_kind"] == "chat"
        rows = [row for row in agent.memory.recent(100) if row.get("kind") == "dialogue_feedback"]
        assert rows
        assert not rows[-1]["text"].split("Turn B: ",1)[-1].lstrip().startswith("[")
    finally:
        agent.close()


def test_visual_buffer_persists_and_does_not_train_from_single_image(tmp_path):
    system = PerceptionSystem(tmp_path, torch.device("cpu"), embedding_dim=16, batch_size=32, buffer_size=64)
    image = np.zeros((128, 128, 3), dtype=np.uint8)
    image[:, :, 0] = 100
    first = system.train_visual_observations([image], steps=1)
    assert first["trained"] is False
    assert first["distinct_images"] == 1
    for i in range(32):
        item = image.copy()
        item[:, :, 1] = (i + 1) * 7
        system.train_visual_observations([item], steps=1)
    system.flush()
    assert (tmp_path / "data" / "perception_buffers.pt").exists()
    restored = PerceptionSystem(tmp_path, torch.device("cpu"), embedding_dim=16, batch_size=32, buffer_size=64)
    assert len(restored._vision_buffer) >= 32


def test_source_validation_cleans_candidate_tree_and_stripped_checkpoint(tmp_path):
    import torch
    from aletheia.evolution import EvolutionLab
    lab = EvolutionLab(tmp_path)
    (tmp_path / "aletheia").mkdir(parents=True, exist_ok=True)
    (tmp_path / "tests").mkdir(parents=True, exist_ok=True)
    (tmp_path / "tests" / "test_ok.py").write_text("def test_ok(): assert True\n", encoding="utf-8")
    checkpoint = tmp_path / "data" / "aletheia.pt"
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"config": {"dim": 1}, "action_names": [], "state_dict": {}, "optimizer": {"state": {1: 2}, "param_groups": []}}, checkpoint)
    candidate = lab.stage_source_patch("aletheia/demo.py", "x = 2\n")
    validation = lab.validate_source_patch(candidate, "aletheia/demo.py", timeout=30)
    assert validation["validated"] is True
    candidate_roots = list((tmp_path / "data" / "evolution" / "candidates").glob("run_*"))
    assert candidate_roots == []


def test_source_validation_treats_timeout_as_rejection(tmp_path, monkeypatch):
    import subprocess
    lab = EvolutionLab(tmp_path)
    candidate = lab.stage_source_patch("aletheia/demo.py", "x = 2\n")
    original = subprocess.run
    def timeout_run(*args, **kwargs):
        # The safe runner may wrap the interpreter with prlimit on POSIX; detect the
        # interpreter anywhere in the argv vector rather than assuming argv[0].
        if args and isinstance(args[0], list) and __import__("sys").executable in args[0]:
            raise subprocess.TimeoutExpired(args[0], kwargs.get("timeout", 1))
        return original(*args, **kwargs)
    monkeypatch.setattr(subprocess, "run", timeout_run)
    result = lab.validate_source_patch(candidate, "aletheia/demo.py", timeout=1)
    assert result["validated"] is False
    assert "timeout" in result["test_output"]
