import torch

from aletheia.config import ModelConfig
from aletheia.model import AletheiaModel
from aletheia.tokenizer import encode


def test_model_forward_and_recurrent_backprop():
    cfg = ModelConfig(dim=64, layers=2, heads=4, ffn=128, recurrent_dim=96, context=64)
    model = AletheiaModel(cfg, ["discover", "follow", "reflect"])
    state = model.initial_state(1, torch.device("cpu"))
    a = torch.tensor([encode("first developmental observation")], dtype=torch.long)
    b = torch.tensor([encode("second developmental observation")], dtype=torch.long)
    loss, final_state = model.trajectory_loss([a, b], [None, 0], [0.0, 0.1])
    assert torch.isfinite(loss)
    assert final_state.shape == (1, 96)
    loss.backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.recurrent.parameters())


def test_policy_optimizer_uses_configured_learning_rate():
    from aletheia.config import load_config
    cfg = load_config()
    from aletheia.agent import AletheiaAgent
    import tempfile
    from pathlib import Path
    with tempfile.TemporaryDirectory() as tmp:
        agent = AletheiaAgent(Path(tmp), cfg)
        group_lrs = {id(group["params"][0]): group["lr"] for group in agent.optimizer.param_groups if group["params"]}
        action_ids = {id(parameter) for parameter in agent.model.action_head.parameters()}
        action_lrs = {lr for group in agent.optimizer.param_groups for lr in [group["lr"]] if any(id(p) in action_ids for p in group["params"])}
        predictive_lrs = {lr for group in agent.optimizer.param_groups for lr in [group["lr"]] if not any(id(p) in action_ids for p in group["params"])}
        assert cfg.runtime.policy_learning_rate in action_lrs
        assert cfg.runtime.learning_rate in predictive_lrs
        agent.close()


def test_full_configuration_single_update():
    from aletheia.config import load_config
    cfg = load_config()
    model = AletheiaModel(cfg.model, list(cfg.capabilities))
    assert sum(p.numel() for p in model.parameters()) > 6_000_000
    a = torch.tensor([encode("one real developmental observation")], dtype=torch.long)
    b = torch.tensor([encode("another real developmental observation")], dtype=torch.long)
    loss, _ = model.trajectory_loss([a, b], [0, 1], [0.0, 0.1])
    assert torch.isfinite(loss)
    loss.backward()


def test_legacy_checkpoint_without_action_names_is_migratable(tmp_path):
    import torch
    from aletheia.config import ModelConfig
    from aletheia.model import AletheiaModel

    cfg = ModelConfig(vocab_size=259, dim=32, layers=1, heads=4, ffn=64, recurrent_dim=16, context=32, dropout=0.0)
    actions = ["discover", "search"]
    model = AletheiaModel(cfg, actions)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    path = tmp_path / "legacy.pt"
    payload = {"config": cfg.__dict__, "state_dict": model.state_dict(), "optimizer": optimizer.state_dict()}
    torch.save(payload, path)
    restored_optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    restored = AletheiaModel.load(path, torch.device("cpu"), cfg, actions, restored_optimizer)
    assert restored.action_names == actions


def test_random_initialization_has_sane_language_scale():
    cfg = ModelConfig(dim=32, layers=1, heads=4, ffn=64, recurrent_dim=48, context=64)
    model = AletheiaModel(cfg, ["discover", "follow", "reflect"])
    tokens = torch.tensor([encode("initialization sanity check")], dtype=torch.long)
    bpb = model.language_bpb(tokens)
    assert 0.0 < bpb < 12.0




def test_batched_trajectory_casts_recurrent_state_back_to_master_dtype():
    """Regression test for mixed-precision GRU output assigned into persistent state."""
    import torch.nn as nn

    cfg = ModelConfig(
        dim=32, layers=1, heads=4, ffn=64, recurrent_dim=48, context=128, dropout=0.0
    )
    model = AletheiaModel(cfg, ["teacher"])

    class HalfOutputRecurrent(nn.Module):
        def __init__(self, base: nn.Module):
            super().__init__()
            self.base = base

        def forward(self, inputs: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
            return self.base(inputs, state).half()

    model.recurrent = HalfOutputRecurrent(model.recurrent)
    sequences = []
    for text in ("first developmental observation", "second developmental observation"):
        tokens = torch.tensor([encode(text)], dtype=torch.long)
        sequences.append(([tokens], [None], [0.0]))

    loss, state = model.trajectory_loss_batched(sequences)
    assert torch.isfinite(loss)
    assert state.dtype == model.token_embedding.weight.dtype
    loss.backward()


def test_batched_trajectory_state_survives_cuda_autocast_if_available():
    """Regression test for FP16 recurrent output assigned into FP32 state."""
    import pytest

    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")

    cfg = ModelConfig(
        dim=64, layers=2, heads=4, ffn=128, recurrent_dim=96, context=64, dropout=0.0
    )
    model = AletheiaModel(cfg, ["teacher"]).cuda()
    sequences = []
    for text in ("first developmental observation", "second developmental observation"):
        tokens = torch.tensor([encode(text)], dtype=torch.long, device="cuda")
        sequences.append(([tokens], [None], [0.0]))

    with torch.autocast(device_type="cuda", dtype=torch.float16):
        loss, state = model.trajectory_loss_batched(sequences)

    assert torch.isfinite(loss)
    assert state.dtype == torch.float32
    loss.backward()
