from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn
import torch.nn.functional as F


class ObjectiveValueHead(nn.Module):
    """Learned transition-value estimator with no domain-specific semantics."""

    def __init__(self, state_dim: int, action_dim: int, hidden: int = 192):
        super().__init__()
        state_dim = int(state_dim)
        action_dim = int(action_dim)
        hidden = max(32, int(hidden))
        input_dim = state_dim * 3 + action_dim
        self.net = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, 1),
        )
        self._initialize()

    def _initialize(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(
        self,
        state_before: torch.Tensor,
        state_after: torch.Tensor,
        goal: torch.Tensor,
        action: torch.Tensor,
    ) -> torch.Tensor:
        tensors = []
        for value in (state_before, state_after, goal, action):
            if value.ndim == 1:
                value = value.unsqueeze(0)
            tensors.append(value)
        return self.net(torch.cat(tensors, dim=-1)).squeeze(-1)


@dataclass
class ObjectiveOutcomeStats:
    examples: int = 0
    positives: int = 0
    mean_outcome: float = 0.0
    training_steps: int = 0
    last_loss: float | None = None


class ObjectiveOutcomeLearner:
    """Small from-scratch value learner trained only from observed outcomes."""

    def __init__(
        self,
        root: Path,
        device: torch.device,
        *,
        state_dim: int,
        action_dim: int,
        hidden: int = 192,
        learning_rate: float = 2e-4,
        batch_size: int = 32,
        buffer_size: int = 4096,
        min_examples: int = 32,
    ):
        self.root = Path(root).resolve()
        self.path = self.root / "data" / "objective_value.pt"
        self.device = device
        self.batch_size = max(1, int(batch_size))
        self.buffer_size = max(32, int(buffer_size))
        self.min_examples = max(1, int(min_examples))
        self.model = ObjectiveValueHead(state_dim, action_dim, hidden).to(device)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=float(learning_rate))
        self.stats = ObjectiveOutcomeStats()
        self._examples: list[dict[str, torch.Tensor | float | str]] = []
        self.load()

    @property
    def ready(self) -> bool:
        return self.stats.examples >= self.min_examples and self.stats.training_steps > 0

    def _normalize_outcome(self, outcome: float) -> float:
        value = float(outcome)
        if not math.isfinite(value):
            raise ValueError("objective outcome must be finite")
        return max(0.0, min(1.0, value))

    @staticmethod
    def _cpu_half(value: torch.Tensor) -> torch.Tensor:
        return value.detach().float().cpu().to(dtype=torch.float16)

    def record(
        self,
        state_before: torch.Tensor,
        state_after: torch.Tensor,
        goal: torch.Tensor,
        action: torch.Tensor,
        outcome: float,
        *,
        objective_id: str = "",
        source: str = "outcome",
        persist: bool = True,
    ) -> None:
        item = {
            "state_before": self._cpu_half(state_before.squeeze(0)),
            "state_after": self._cpu_half(state_after.squeeze(0)),
            "goal": self._cpu_half(goal.squeeze(0)),
            "action": self._cpu_half(action.squeeze(0)),
            "outcome": float(self._normalize_outcome(outcome)),
            "objective_id": str(objective_id),
            "source": str(source),
        }
        self._examples.append(item)
        if len(self._examples) > self.buffer_size:
            self._examples = self._examples[-self.buffer_size:]
        outcomes = [float(item["outcome"]) for item in self._examples]
        self.stats.examples = len(self._examples)
        self.stats.positives = sum(value >= 0.5 for value in outcomes)
        self.stats.mean_outcome = float(sum(outcomes) / max(1, len(outcomes)))
        if persist:
            self.save()

    def _batch(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        count = min(self.batch_size, len(self._examples))
        if count <= 0:
            raise ValueError("no objective outcomes recorded")
        indices = torch.randperm(len(self._examples))[:count].tolist()
        chosen = [self._examples[index] for index in indices]
        states_before = torch.stack([item["state_before"] for item in chosen]).float().to(self.device)
        states_after = torch.stack([item["state_after"] for item in chosen]).float().to(self.device)
        goals = torch.stack([item["goal"] for item in chosen]).float().to(self.device)
        actions = torch.stack([item["action"] for item in chosen]).float().to(self.device)
        targets = torch.tensor([float(item["outcome"]) for item in chosen], device=self.device)
        return states_before, states_after, goals, actions, targets

    def outcome_stats(self, objective_id: str) -> dict[str, float | int]:
        outcomes = [
            float(item.get("outcome", 0.0))
            for item in self._examples
            if str(item.get("objective_id", "")) == str(objective_id)
        ]
        return {
            "count": len(outcomes),
            "mean": float(sum(outcomes) / len(outcomes)) if outcomes else 0.0,
        }

    def fit(self, steps: int = 1) -> dict[str, Any]:
        if not self._examples:
            return {"trained": False, "reason": "no_examples", "examples": 0}
        self.model.train()
        losses: list[float] = []
        for _ in range(max(1, int(steps))):
            state_before, state_after, goal, action, target = self._batch()
            logits = self.model(state_before, state_after, goal, action)
            loss = F.binary_cross_entropy_with_logits(logits, target)
            if not torch.isfinite(loss):
                continue
            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            losses.append(float(loss.detach().cpu()))
            self.stats.training_steps += 1
        self.stats.last_loss = losses[-1] if losses else self.stats.last_loss
        self.model.eval()
        self.save()
        return {
            "trained": bool(losses),
            "steps": len(losses),
            "loss": self.stats.last_loss,
            "examples": self.stats.examples,
            "ready": self.ready,
        }

    @torch.no_grad()
    def predict(
        self,
        state_before: torch.Tensor,
        state_after: torch.Tensor,
        goal: torch.Tensor,
        action: torch.Tensor,
    ) -> float | None:
        if not self._examples:
            return None
        self.model.eval()
        logits = self.model(
            state_before.to(self.device),
            state_after.to(self.device),
            goal.to(self.device),
            action.to(self.device),
        )
        return float(torch.sigmoid(logits).mean().detach().cpu())

    def estimate(
        self,
        state_before: torch.Tensor | None,
        state_after: torch.Tensor | None,
        goal: torch.Tensor | None,
        action: torch.Tensor | None,
        *,
        fallback: float | None = None,
    ) -> tuple[float | None, str]:
        if self.ready and state_before is not None and state_after is not None and goal is not None and action is not None:
            predicted = self.predict(state_before, state_after, goal, action)
            if predicted is not None and math.isfinite(predicted):
                return predicted, "learned"
        if fallback is not None and math.isfinite(float(fallback)):
            return max(0.0, min(1.0, float(fallback))), "feedback_mean"
        if self.stats.examples:
            return self.stats.mean_outcome, "feedback_mean_global"
        return fallback, "unobserved"

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "schema_version": 1,
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "stats": self.stats.__dict__,
            "examples": self._examples,
        }
        tmp = self.path.with_suffix(".tmp")
        torch.save(payload, tmp)
        tmp.replace(self.path)

    def load(self) -> None:
        if not self.path.exists():
            self.model.eval()
            return
        try:
            payload = torch.load(self.path, map_location=self.device, weights_only=True)
            if not isinstance(payload, dict):
                return
            state = payload.get("model")
            if isinstance(state, dict):
                self.model.load_state_dict(state, strict=False)
            optimizer = payload.get("optimizer")
            if isinstance(optimizer, dict):
                try:
                    self.optimizer.load_state_dict(optimizer)
                except Exception:
                    pass
            raw_stats = payload.get("stats", {})
            if isinstance(raw_stats, dict):
                self.stats = ObjectiveOutcomeStats(
                    examples=int(raw_stats.get("examples", 0)),
                    positives=int(raw_stats.get("positives", 0)),
                    mean_outcome=float(raw_stats.get("mean_outcome", 0.0)),
                    training_steps=int(raw_stats.get("training_steps", 0)),
                    last_loss=raw_stats.get("last_loss"),
                )
            examples = payload.get("examples", [])
            if isinstance(examples, list):
                self._examples = examples[-self.buffer_size:]
                self.stats.examples = len(self._examples)
                if self._examples:
                    outcomes = [float(item.get("outcome", 0.0)) for item in self._examples]
                    self.stats.positives = sum(value >= 0.5 for value in outcomes)
                    self.stats.mean_outcome = float(sum(outcomes) / len(outcomes))
        except Exception:
            self._examples = []
            self.stats = ObjectiveOutcomeStats()
        self.model.eval()
