import json
from pathlib import Path

import torch

from aletheia.config import ModelConfig, load_config
from aletheia.memory import ExperienceMemory
from aletheia.model import AletheiaModel, CHECKPOINT_SCHEMA_VERSION
from aletheia.tokenizer import encode


def test_multimodal_fusion_returns_state_and_attention():
    cfg = ModelConfig(dim=48, layers=1, heads=4, ffn=96, recurrent_dim=64, context=64, world_attention_heads=4, world_model_hidden=96)
    model = AletheiaModel(cfg, ["observe", "act"])
    state = model.initial_state(1, torch.device("cpu"))
    text = torch.randn(1, cfg.dim)
    vision = torch.randn(1, cfg.perception_dim)
    audio = torch.randn(1, cfg.perception_dim)
    fused, attention = model.integrate_multimodal(
        state, text_summary=text, vision_embedding=vision, audio_embedding=audio
    )
    assert fused.shape == state.shape
    assert attention.shape == (1, 3)
    assert torch.isfinite(fused).all()
    assert torch.isfinite(attention).all()
    assert torch.allclose(attention.sum(dim=-1), torch.ones(1), atol=1e-5)


def test_world_prediction_path_backpropagates():
    cfg = ModelConfig(dim=48, layers=1, heads=4, ffn=96, recurrent_dim=64, context=64, world_attention_heads=4, world_model_hidden=96)
    model = AletheiaModel(cfg, ["observe", "act"])
    tokens = [
        torch.tensor([encode("the first developmental event")]),
        torch.tensor([encode("the second developmental event follows")]),
        torch.tensor([encode("the third developmental event changes")]),
    ]
    loss, _ = model.trajectory_loss(tokens, [None, 0, 1], [0.0, 0.1, 0.2])
    assert torch.isfinite(loss)
    loss.backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.world_transition.parameters())
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.world_observation_head.parameters())


def test_world_prediction_batched_path_is_finite():
    cfg = ModelConfig(dim=48, layers=1, heads=4, ffn=96, recurrent_dim=64, context=64, world_attention_heads=4, world_model_hidden=96)
    model = AletheiaModel(cfg, ["observe", "act"])
    sequences = []
    for prefix in ("alpha", "beta"):
        chunks = [
            torch.tensor([encode(f"{prefix} observation one")]),
            torch.tensor([encode(f"{prefix} observation two")]),
        ]
        sequences.append((chunks, [None, 1], [0.0, 0.2]))
    loss, state = model.trajectory_loss_batched(sequences)
    assert torch.isfinite(loss)
    assert state.shape == (2, cfg.recurrent_dim)
    loss.backward()


def test_final16_checkpoint_migrates_cognitive_modules(tmp_path):
    cfg = ModelConfig(dim=32, layers=1, heads=4, ffn=64, recurrent_dim=48, context=48, world_attention_heads=4, world_model_hidden=64)
    model = AletheiaModel(cfg, ["observe", "act"])
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    payload = {
        "schema_version": 7,
        "config": {
            "vocab_size": cfg.vocab_size,
            "dim": cfg.dim,
            "layers": cfg.layers,
            "heads": cfg.heads,
            "ffn": cfg.ffn,
            "recurrent_dim": cfg.recurrent_dim,
            "context": cfg.context,
            "perception_dim": cfg.perception_dim,
        },
        "action_names": list(model.action_names),
        "state_dict": {
            key: value for key, value in model.state_dict().items()
            if not key.startswith(("text_to_world", "modality_embedding", "world_attention", "world_query", "world_fusion_gate", "world_observation_head", "world_transition", "world_uncertainty_head"))
        },
        "optimizer": optimizer.state_dict(),
    }
    path = tmp_path / "final16.pt"
    torch.save(payload, path)
    restored = AletheiaModel.load(path, torch.device("cpu"), cfg, list(model.action_names), torch.optim.AdamW(model.parameters(), lr=1e-3))
    assert CHECKPOINT_SCHEMA_VERSION == 10
    assert restored.world_transition is not None


def test_cognitive_memory_events_are_compact_and_persistent(tmp_path):
    memory = ExperienceMemory(tmp_path / "memory.sqlite3", limit=50, frontier_limit=20)
    event_id = memory.add_cognitive_event(
        event_kind="visual_observation",
        modality="vision",
        source="camera",
        entity_id="entity_1",
        salience=0.8,
        novelty=0.4,
        prediction_error=0.2,
        detail={"attention": [0.2, 0.8]},
        limit=3,
    )
    assert event_id > 0
    rows = memory.recent_cognitive_events(4)
    assert len(rows) == 1
    assert rows[0]["entity_id"] == "entity_1"
    assert rows[0]["detail"]["attention"] == [0.2, 0.8]
    summary = memory.cognitive_summary()
    assert summary["events"] == 1
    memory.close()


def test_runtime_config_exposes_final17_cognitive_controls():
    cfg = load_config()
    assert cfg.runtime.cognitive_event_limit > 0
    assert cfg.runtime.cognitive_recent_events > 0
    assert 0.0 <= cfg.runtime.cognitive_attention_threshold <= 1.0
