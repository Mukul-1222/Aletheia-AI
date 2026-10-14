"""Small deterministic environment interface for outcome-verification experiments.

The environment mechanics are generic and configured with explicit transitions. The
agent receives only observations and legal actions. The evaluator's target condition is
kept behind a separate method and should be called by the experiment runner, not fed
back as a hidden feature during action selection.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Hashable, Mapping, Sequence

State = Hashable
Action = Hashable


@dataclass(frozen=True)
class EnvironmentObservation:
    state: State
    available_actions: tuple[Action, ...]
    step_index: int
    terminal: bool


@dataclass(frozen=True)
class EnvironmentTransition:
    previous: EnvironmentObservation
    action: Action
    next_observation: EnvironmentObservation
    prediction_match: bool | None
    externally_verified: bool | None
    verifier_reason: str


class DeterministicEnvironment:
    """A transition-table environment with an externally supplied verifier."""

    def __init__(
        self,
        *,
        initial_state: State,
        transitions: Mapping[tuple[State, Action], State],
        goal_verifier: Callable[[State], bool] | None,
        max_steps: int = 32,
    ) -> None:
        if max_steps < 1:
            raise ValueError("max_steps must be at least 1")
        self._initial = initial_state
        self._transitions = dict(transitions)
        self._verifier = goal_verifier
        self._max_steps = int(max_steps)
        self.reset()

    def reset(self) -> EnvironmentObservation:
        self._state = self._initial
        self._step = 0
        self._done = False
        self._last_transition: EnvironmentTransition | None = None
        return self.observe()

    def observe(self) -> EnvironmentObservation:
        legal = tuple(action for (state, action), _ in self._transitions.items() if state == self._state)
        return EnvironmentObservation(self._state, legal, self._step, self._done)

    def step(self, action: Action, *, predicted_next_state: State | None = None) -> EnvironmentTransition:
        if self._done:
            raise RuntimeError("environment is terminal; call reset() before another action")
        previous = self.observe()
        key = (self._state, action)
        if key not in self._transitions:
            raise ValueError("invalid_action")
        self._state = self._transitions[key]
        self._step += 1
        reached_goal = self._verifier(self._state) if self._verifier is not None else None
        timed_out = self._step >= self._max_steps and reached_goal is not True
        self._done = reached_goal is True or timed_out
        next_observation = self.observe()
        match = None if predicted_next_state is None else predicted_next_state == self._state
        if reached_goal is None:
            reason = "feedback_missing"
        elif reached_goal:
            reason = "goal_verified"
        elif timed_out:
            reason = "timeout"
        else:
            reason = "goal_not_met"
        transition = EnvironmentTransition(
            previous=previous, action=action, next_observation=next_observation,
            prediction_match=match, externally_verified=reached_goal, verifier_reason=reason,
        )
        self._last_transition = transition
        return transition

    def evaluator_result(self) -> dict:
        """Return evaluator-only information; do not pass this into policy inputs."""
        if self._last_transition is None:
            return {"status": "unknown", "reason": "no_transition"}
        verified = self._last_transition.externally_verified
        if verified is None:
            return {"status": "unknown", "reason": "feedback_missing"}
        return {"status": "success" if verified else ("timeout" if self._last_transition.verifier_reason == "timeout" else "failure"),
                "reason": self._last_transition.verifier_reason}
