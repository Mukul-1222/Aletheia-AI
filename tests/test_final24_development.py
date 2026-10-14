from __future__ import annotations

from pathlib import Path

import torch

from aletheia.agent import AletheiaAgent
from aletheia.config import load_config
from aletheia.objective_value import ObjectiveOutcomeLearner


def _small_config(tmp_path):
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
    cfg.runtime.objective_value_min_examples = 2
    cfg.runtime.objective_value_training_steps = 2
    cfg.runtime.objective_value_batch_size = 2
    cfg.runtime.objective_value_buffer_size = 32
    cfg.runtime.minimum_bootstrap_memories = 0
    return cfg


def test_holdout_is_contiguous_tail(tmp_path):
    agent = AletheiaAgent(tmp_path, _small_config(tmp_path))
    try:
        train, holdout = agent._split_train_holdout_indices(12, 0.25)
        assert train == list(range(9))
        assert holdout == list(range(9, 12))
        assert max(train) + 1 == min(holdout)
    finally:
        agent.close()


def test_objective_value_model_learns_feedback_transition(tmp_path):
    device = torch.device("cpu")
    learner = ObjectiveOutcomeLearner(
        tmp_path,
        device,
        state_dim=8,
        action_dim=4,
        hidden=16,
        batch_size=2,
        buffer_size=8,
        min_examples=2,
    )
    state_before = torch.randn(8)
    state_after = torch.randn(8)
    goal = torch.randn(8)
    action = torch.randn(4)
    learner.record(state_before, state_after, goal, action, 1.0, objective_id="o1")
    learner.record(state_before + 0.1, state_after + 0.1, goal, action, 0.0, objective_id="o2")
    result = learner.fit(steps=4)
    assert result["trained"] is True
    assert learner.stats.examples == 2
    assert learner.stats.training_steps > 0
    prediction = learner.predict(state_before, state_after, goal, action)
    assert prediction is not None
    assert 0.0 <= prediction <= 1.0
    assert learner.path.exists()


def test_objective_progress_comes_from_observed_feedback_not_fixed_delta(tmp_path):
    cfg = _small_config(tmp_path)
    agent = AletheiaAgent(tmp_path, cfg)
    try:
        agent.set_objective("learn visual identity")
        agent._cap_search = lambda query_override=None: (0.9, {"kind": "search", "learning_progress": 0.9})
        event = agent.autonomous_cycle()
        objective = agent.objectives.current()
        assert objective.progress == 0.0
        feedback = agent.objective_feedback(True, "observed success")
        assert feedback["updated"] is True
        assert feedback["progress"] > 0.0
        assert feedback["objective_progress_source"] in {"feedback_mean", "learned"}
        assert feedback["objective_outcome_count"] >= 1
    finally:
        agent.close()


def test_planned_actions_train_policy_head(tmp_path):
    cfg = _small_config(tmp_path)
    agent = AletheiaAgent(tmp_path, cfg)
    try:
        mask = torch.ones(len(agent.registry.items()), dtype=torch.bool)
        state = agent.state.detach().clone()
        before = [p.detach().clone() for p in agent.model.action_head.parameters()]
        updated = agent._learn_policy(state, 0, 1.0, True, "planned", mask)
        after = [p.detach().clone() for p in agent.model.action_head.parameters()]
        assert updated is True
        assert any(not torch.equal(a, b) for a, b in zip(before, after))
    finally:
        agent.close()


def test_objective_value_persistence_preserves_examples_and_stats(tmp_path):
    device = torch.device("cpu")
    first = ObjectiveOutcomeLearner(tmp_path, device, state_dim=4, action_dim=3, hidden=12, min_examples=2, batch_size=2, buffer_size=8)
    state = torch.randn(4)
    goal = torch.randn(4)
    action = torch.randn(3)
    first.record(state, state + 0.2, goal, action, 1.0, objective_id="obj")
    first.record(state, state + 0.3, goal, action, 0.0, objective_id="obj")
    restored = ObjectiveOutcomeLearner(tmp_path, device, state_dim=4, action_dim=3, hidden=12, min_examples=2, batch_size=2, buffer_size=8)
    stats = restored.outcome_stats("obj")
    assert stats["count"] == 2
    assert stats["mean"] == 0.5


def test_objective_progress_uses_per_objective_feedback_not_global_average(tmp_path):
    device = torch.device("cpu")
    learner = ObjectiveOutcomeLearner(tmp_path, device, state_dim=4, action_dim=3, hidden=12, min_examples=10, batch_size=2, buffer_size=16)
    state = torch.randn(4); goal = torch.randn(4); action = torch.randn(3)
    for _ in range(4):
        learner.record(state, state + 0.1, goal, action, 1.0, objective_id="other")
    learner.record(state, state + 0.1, goal, action, 0.0, objective_id="target")
    local = learner.outcome_stats("target")
    assert local["count"] == 1 and local["mean"] == 0.0


def test_evolution_acceptance_does_not_invent_objective_progress():
    agent = object.__new__(AletheiaAgent)
    from types import SimpleNamespace
    agent.objectives = SimpleNamespace(current=lambda: SimpleNamespace(id="o1", progress=0.42), update=lambda *_a, **_k: None)
    agent.config = SimpleNamespace(runtime=SimpleNamespace(evolution_runtime_mutations=["learning_rate"], learning_rate=0.001))
    agent.cycles = 0
    from aletheia.evolution import EvolutionResult
    agent.evolution = SimpleNamespace(
        evolve_runtime_parameter=lambda _agent, _parameter: EvolutionResult(
            changed=True, parameter="learning_rate", before=0.001, after=0.0005,
            score_before=1.0, score_after=0.9, accepted=True, reason="test"
        )
    )
    signal, result = agent._cap_evolve()
    assert signal == 1.0
    assert result["valid_transition"] is True


def test_failed_pretrain_does_not_commit_unique_tokens(tmp_path):
    cfg = _small_config(tmp_path)
    cfg.runtime.pretrain_epochs = 1
    cfg.runtime.pretrain_sequence_length = 1
    cfg.runtime.train_batch_size = 1
    agent = AletheiaAgent(tmp_path, cfg)
    try:
        def fail_training(*_args, **_kwargs):
            raise RuntimeError("training failure")
        agent._train_sequence_collection = fail_training
        source = tmp_path / "lesson.txt"
        source.write_text("A sufficiently long foundation lesson for the failure accounting test. " * 4, encoding="utf-8")
        try:
            agent.pretrain(source, epochs=1, seed=1)
        except RuntimeError:
            pass
        assert agent.total_tokens == 0
    finally:
        agent.close()


def test_objective_value_estimate_uses_per_objective_cold_start(tmp_path):
    learner = ObjectiveOutcomeLearner(tmp_path, torch.device("cpu"), state_dim=4, action_dim=3, hidden=12, min_examples=10, batch_size=2, buffer_size=16)
    state = torch.randn(4); goal = torch.randn(4); action = torch.randn(3)
    for _ in range(4):
        learner.record(state, state + 0.1, goal, action, 1.0, objective_id="other")
    learner.record(state, state + 0.2, goal, action, 0.0, objective_id="target")
    estimate, source = learner.estimate(state, state + 0.2, goal, action, fallback=0.0)
    assert estimate == 0.0
    assert source == "feedback_mean"


def test_record_objective_outcome_accepts_continuous_environment_score(tmp_path):
    cfg = _small_config(tmp_path)
    agent = AletheiaAgent(tmp_path, cfg)
    try:
        agent.set_objective("an arbitrary measured task")
        agent._last_action_state = torch.randn(cfg.model.recurrent_dim)
        agent._last_action_state_after = torch.randn(cfg.model.recurrent_dim)
        agent._last_action_index = 0
        agent._last_feedback_target = {"kind": "autonomous", "action_name": "test_action"}
        result = agent.record_objective_outcome(
            0.7, source="environment", verified=True,
            evidence={"verifier": "test_environment", "measurement": "independently observed"},
        )
        assert result["updated"] is True
        assert result["outcome"] == 0.7
        assert result["source"] == "environment"
        assert 0.0 <= result["progress"] <= 0.99
    finally:
        agent.close()
