
import json

import torch

from aletheia.config import ModelConfig, load_config
from aletheia.model import AletheiaModel
from aletheia.planner import ObjectiveHierarchy, WorldModelPlanner
from aletheia.tokenizer import encode


def _small_model():
    cfg = ModelConfig(
        dim=32,
        layers=1,
        heads=4,
        ffn=64,
        recurrent_dim=48,
        context=64,
        dropout=0.0,
        world_model_hidden=64,
        world_attention_heads=4,
    )
    return AletheiaModel(cfg, ["a", "b", "c"])


def test_planner_rolls_out_learned_actions_without_world_labels(tmp_path):
    torch.manual_seed(4)
    model = _small_model()
    planner = WorldModelPlanner(
        tmp_path,
        torch.device("cpu"),
        model,
        ["a", "b", "c"],
        horizon=2,
        beam_width=3,
        branch_factor=2,
        hidden=32,
    )
    state = model.initial_state(1, torch.device("cpu"))
    mask = torch.tensor([True, False, True])
    result = planner.plan(
        state,
        "reach a novel developmental outcome",
        valid_mask=mask,
    )
    assert result["enabled"] is True
    assert result["action_index"] in {0, 2}
    assert len(result["actions"]) >= 1
    assert all(name in {"a", "c"} for name in result["action_names"])
    assert torch.isfinite(torch.tensor(result["score"]))
    assert torch.isfinite(torch.tensor(result["uncertainty"]))


def test_planner_feedback_changes_value_model_and_persists(tmp_path):
    model = _small_model()
    planner = WorldModelPlanner(tmp_path, torch.device("cpu"), model, ["a", "b", "c"], hidden=32)
    state = model.initial_state(1, torch.device("cpu"))
    before = {
        key: value.detach().clone()
        for key, value in planner.value_head.state_dict().items()
    }
    result = planner.learn_feedback(
        state,
        "learn a useful developmental objective",
        1,
        target=1.0,
        steps=2,
    )
    assert result["accepted"]
    assert planner.training_steps == 2
    changed = any(not torch.equal(before[k], planner.value_head.state_dict()[k]) for k in before)
    assert changed

    restored = WorldModelPlanner(tmp_path, torch.device("cpu"), model, ["a", "b", "c"], hidden=32)
    assert restored.training_steps == 2


def test_objective_hierarchy_is_domain_agnostic_and_persistent(tmp_path):
    calls = []

    def fake_generator(prompt):
        calls.append(prompt)
        return "observe the current situation\nlearn what is missing\nchoose a next experiment"

    hierarchy = ObjectiveHierarchy(tmp_path / "objective_plans.json")
    steps = hierarchy.ensure("objective-1", "improve the outcome", generator=fake_generator, max_steps=6)
    assert [step.text for step in steps] == [
        "observe the current situation",
        "learn what is missing",
        "choose a next experiment",
    ]
    assert calls
    assert hierarchy.active("objective-1").text == "observe the current situation"
    feedback = hierarchy.feedback("objective-1", True)
    assert feedback["completed"] is True
    assert feedback["next_step"] == "learn what is missing"

    hierarchy2 = ObjectiveHierarchy(tmp_path / "objective_plans.json")
    assert hierarchy2.active("objective-1").text == "learn what is missing"
    payload = json.loads((tmp_path / "objective_plans.json").read_text(encoding="utf-8"))
    assert "password" not in json.dumps(payload).lower()
    assert len(payload["plans"]["objective-1"]["steps"]) == 3


def test_final20_runtime_config_present():
    cfg = load_config()
    assert cfg.runtime.planning_enabled is True
    assert cfg.runtime.planning_horizon >= 1
    assert cfg.runtime.planning_beam_width >= 1
    assert cfg.runtime.planning_branch_factor >= 1
    assert 0.0 <= cfg.runtime.planning_discount <= 1.0


def test_agent_uses_model_based_rollout_for_active_objective(tmp_path, monkeypatch):
    from aletheia.agent import AletheiaAgent
    from tests.test_agent import FakeWeb

    cfg = load_config()
    cfg.model.dim = 32
    cfg.model.layers = 1
    cfg.model.heads = 4
    cfg.model.ffn = 64
    cfg.model.recurrent_dim = 48
    cfg.model.context = 64
    cfg.model.world_model_hidden = 64
    cfg.model.world_attention_heads = 4
    cfg.runtime.device = "cpu"
    cfg.runtime.memory_limit = 20
    cfg.runtime.minimum_bootstrap_memories = 1
    cfg.runtime.planning_horizon = 2
    cfg.runtime.planning_beam_width = 2
    cfg.runtime.planning_branch_factor = 2
    cfg.runtime.planning_subgoals = 3
    agent = AletheiaAgent(tmp_path, cfg)
    agent.web.close()
    agent.web = FakeWeb()
    try:
        objective = agent.set_objective("improve the current developmental outcome")
        assert objective["planning_steps"] >= 1
        event = agent.autonomous_cycle()
        assert event["selection_mode"] == "planned"
        assert event["plan"].get("action_name")
        assert event["plan"].get("actions")
    finally:
        agent.close()
