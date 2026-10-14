
from __future__ import annotations

import json
import math
import re
import time
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F



@dataclass
class PlanStep:
    id: str
    objective_id: str
    text: str
    order: int
    status: str = "pending"
    attempts: int = 0
    created: float = 0.0
    completed: float = 0.0


class ObjectiveHierarchy:
    """Persistent, domain-agnostic hierarchy of learned/generated sub-objectives.

    No world facts live here. The only durable content is what came from the user's
    objective or from Aletheia's own generated decomposition.
    """

    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._plans: dict[str, dict] = {}
        self._load()

    def _load(self) -> None:
        if not self.path.exists():
            return
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
            raw = payload.get("plans", {}) if isinstance(payload, dict) else {}
            if isinstance(raw, dict):
                self._plans = raw
        except (OSError, ValueError, TypeError):
            self._plans = {}

    def _save(self) -> None:
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(
            json.dumps({"plans": self._plans}, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        tmp.replace(self.path)

    @staticmethod
    def _clean_lines(text: str, *, max_steps: int) -> list[str]:
        lines: list[str] = []
        seen: set[str] = set()
        for raw in str(text).splitlines():
            value = re.sub(r"^\s*(?:[-*•]|\d+[.)])\s*", "", raw).strip()
            value = re.sub(r"\s+", " ", value)
            if len(value) < 6 or len(value) > 800:
                continue
            key = value.casefold()
            if key in seen:
                continue
            seen.add(key)
            lines.append(value)
            if len(lines) >= max(1, int(max_steps)):
                break
        return lines

    def get(self, objective_id: str) -> list[PlanStep]:
        record = self._plans.get(str(objective_id), {})
        raw = record.get("steps", []) if isinstance(record, dict) else []
        result: list[PlanStep] = []
        for item in raw:
            try:
                result.append(PlanStep(**item))
            except (TypeError, ValueError):
                continue
        return sorted(result, key=lambda item: item.order)

    def ensure(
        self,
        objective_id: str,
        root_text: str,
        *,
        generator: Callable[[str], str] | None = None,
        max_steps: int = 6,
    ) -> list[PlanStep]:
        existing = self.get(objective_id)
        if existing:
            return existing

        candidates: list[str] = []
        if generator is not None:
            prompt = (
                "Decompose the following objective into a short ordered sequence of "
                "concrete sub-objectives for an autonomous learning system. "
                "Use only information present or implied by the objective. "
                "One sub-objective per line. Do not explain the decomposition.\n\n"
                f"OBJECTIVE:\n{root_text}\n\nSUB-OBJECTIVES:\n"
            )
            try:
                candidates = self._clean_lines(generator(prompt), max_steps=max_steps)
            except Exception:
                candidates = []

        # Root-only is an explicit fallback, not semantic world knowledge.
        if not candidates:
            candidates = [str(root_text).strip()]

        now = time.time()
        steps = [
            PlanStep(
                id=uuid.uuid4().hex[:12],
                objective_id=str(objective_id),
                text=item,
                order=index,
                status="pending",
                created=now,
            )
            for index, item in enumerate(candidates)
        ]
        self._plans[str(objective_id)] = {
            "objective_id": str(objective_id),
            "root_text": str(root_text),
            "created": now,
            "updated": now,
            "steps": [asdict(item) for item in steps],
        }
        self._save()
        return steps

    def active(self, objective_id: str) -> PlanStep | None:
        for step in self.get(objective_id):
            if step.status == "pending":
                return step
        return None

    def text(self, objective_id: str, root_text: str) -> str:
        step = self.active(objective_id)
        return step.text if step is not None else root_text

    def feedback(self, objective_id: str, positive: bool) -> dict:
        steps = self.get(objective_id)
        if not steps:
            return {"updated": False, "reason": "no_plan"}
        active = next((item for item in steps if item.status == "pending"), None)
        if active is None:
            return {"updated": False, "reason": "complete"}
        active.attempts += 1
        if positive:
            active.status = "completed"
            active.completed = time.time()
        payload = self._plans.get(str(objective_id), {})
        payload["steps"] = [asdict(item) for item in steps]
        payload["updated"] = time.time()
        self._plans[str(objective_id)] = payload
        self._save()
        next_step = next((item for item in steps if item.status == "pending"), None)
        return {
            "updated": True,
            "step_id": active.id,
            "completed": positive,
            "next_step": next_step.text if next_step else None,
            "remaining": sum(item.status == "pending" for item in steps),
        }

    def status(self, objective_id: str) -> list[dict]:
        return [asdict(item) for item in self.get(objective_id)]


class GoalActionValue(nn.Module):
    """From-scratch action-value head conditioned on state + objective + action."""

    def __init__(self, state_dim: int, action_dim: int, hidden: int = 256):
        super().__init__()
        hidden = max(32, int(hidden))
        self.net = nn.Sequential(
            nn.LayerNorm(state_dim * 2 + action_dim),
            nn.Linear(state_dim * 2 + action_dim, hidden),
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

    def forward(self, state: torch.Tensor, goal: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        if state.ndim == 1:
            state = state.unsqueeze(0)
        if goal.ndim == 1:
            goal = goal.unsqueeze(0)
        if action.ndim == 1:
            action = action.unsqueeze(0)
        return self.net(torch.cat([state, goal, action], dim=-1)).squeeze(-1)


@dataclass
class PlannedPath:
    actions: list[int]
    score: float
    policy_score: float
    goal_score: float
    uncertainty: float
    states: list[torch.Tensor]


class WorldModelPlanner:
    """Model-based planner over Aletheia's learned latent action environment.

    The planner contains no semantic action dictionary and no hard-coded world model.
    It only rolls forward the learned transition model and scores predicted states
    against a learned objective representation plus an experience-trained value head.
    """

    def __init__(
        self,
        root: Path,
        device: torch.device,
        model,
        action_names: Sequence[str],
        *,
        horizon: int = 3,
        beam_width: int = 6,
        branch_factor: int = 4,
        goal_weight: float = 0.65,
        alignment_weight: float = 0.20,
        policy_weight: float = 0.15,
        uncertainty_weight: float = 0.08,
        discount: float = 0.90,
        learning_rate: float = 2e-4,
        hidden: int = 256,
        enabled: bool = True,
    ):
        self.root = Path(root)
        self.data_path = self.root / "data" / "planning.pt"
        self.device = device
        self.model = model
        self.action_names = list(action_names)
        self.horizon = max(1, int(horizon))
        self.beam_width = max(1, int(beam_width))
        self.branch_factor = max(1, int(branch_factor))
        self.goal_weight = float(goal_weight)
        self.alignment_weight = float(alignment_weight)
        self.policy_weight = float(policy_weight)
        self.uncertainty_weight = float(uncertainty_weight)
        self.discount = float(discount)
        self.enabled = bool(enabled)
        self.training_steps = 0
        self.last_plan: dict = {}
        self.value_head = GoalActionValue(
            state_dim=int(model.cfg.recurrent_dim),
            action_dim=int(model.cfg.dim),
            hidden=hidden,
        ).to(device)
        self.optimizer = torch.optim.AdamW(self.value_head.parameters(), lr=float(learning_rate))
        self.hierarchy = ObjectiveHierarchy(self.root / "data" / "objective_plans.json")
        self.load()

    def _goal_vector(self, text: str) -> torch.Tensor:
        tokenizer = getattr(self.model, "tokenizer", None)
        encoded = tokenizer.encode(str(text), bos=True, eos=False) if tokenizer is not None else [ord(ch) % 256 for ch in str(text)]
        tokens = torch.tensor([encoded], dtype=torch.long, device=self.device)
        with torch.no_grad():
            rep = self.model.text_representation(tokens)
            world = self.model.text_to_world(rep)
            return F.normalize(world, dim=-1)

    @staticmethod
    def _masked_policy(logits: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
        logits = logits.float()
        mask = valid_mask.to(device=logits.device, dtype=torch.bool)
        masked = logits.masked_fill(~mask, float("-inf"))
        if not bool(torch.isfinite(masked).any()):
            return torch.zeros_like(logits)
        return F.softmax(masked, dim=-1)

    def _candidate_actions(self, state: torch.Tensor, valid_mask: torch.Tensor) -> tuple[list[int], torch.Tensor]:
        with torch.no_grad():
            logits = self.model.action_logits(state).squeeze(0)
            probs = self._masked_policy(logits, valid_mask)
            eligible = valid_mask.nonzero(as_tuple=False).flatten()
            if eligible.numel() == 0:
                return [], probs
            count = min(self.branch_factor, int(eligible.numel()))
            values, indices = torch.topk(probs, count)
            return [int(item) for item in indices.tolist() if bool(valid_mask[item])], probs

    @staticmethod
    def _alignment(state: torch.Tensor, goal: torch.Tensor) -> torch.Tensor:
        state_vec = F.normalize(state, dim=-1)
        return ((state_vec * goal).sum(dim=-1) + 1.0) * 0.5

    def _value(
        self,
        state: torch.Tensor,
        goal: torch.Tensor,
        action_index: int,
    ) -> torch.Tensor:
        action = self.model.action_embedding(
            torch.tensor([int(action_index)], dtype=torch.long, device=self.device)
        )
        with torch.no_grad():
            return torch.sigmoid(self.value_head(state, goal, action))

    @torch.no_grad()
    def plan(
        self,
        state: torch.Tensor,
        objective_text: str,
        *,
        prediction_error: float = 0.0,
        valid_mask: torch.Tensor | None = None,
        previous_action: int | None = None,
    ) -> dict:
        was_training = bool(self.model.training)
        self.model.eval()
        self.value_head.eval()
        if state.ndim == 1:
            state = state.unsqueeze(0)
        if valid_mask is None:
            valid_mask = torch.ones(len(self.action_names), dtype=torch.bool, device=self.device)
        if valid_mask.numel() != len(self.action_names):
            raise ValueError("valid_mask size must match action count")
        goal = self._goal_vector(objective_text)
        root_candidates, root_probs = self._candidate_actions(state, valid_mask)
        if not root_candidates:
            result = {
                "enabled": False,
                "reason": "no_valid_actions",
                "action_index": None,
                "action_name": None,
                "path": [],
                "entropy": 0.0,
            }
            if was_training:
                self.model.train()
            return result

        entropy = float(
            (-(root_probs[valid_mask] * root_probs[valid_mask].clamp_min(1e-8).log()).sum())
            .detach().cpu()
        )
        beams = [{
            "state": state.detach(),
            "actions": [],
            "score": 0.0,
            "policy_score": 0.0,
            "goal_score": 0.0,
            "uncertainty": 0.0,
            "states": [state.detach()],
            "error": float(prediction_error),
        }]

        for depth in range(self.horizon):
            expanded: list[dict] = []
            for beam in beams:
                candidates, probs = self._candidate_actions(beam["state"], valid_mask)
                if depth == 0 and root_candidates:
                    candidates = root_candidates
                for action_index in candidates:
                    probability = float(probs[action_index].clamp_min(1e-8).item())
                    next_state = self.model.predict_next_state(
                        beam["state"],
                        action_index,
                        beam["error"],
                    )
                    uncertainty = float(
                        self.model.transition_uncertainty(next_state).mean().detach().cpu()
                    )
                    alignment = float(self._alignment(next_state, goal).mean().detach().cpu())
                    value = float(
                        self._value(next_state, goal, action_index).mean().detach().cpu()
                    )
                    policy_term = math.log(max(probability, 1e-8))
                    local_score = (
                        self.discount ** depth
                    ) * (
                        self.goal_weight * value
                        + self.alignment_weight * alignment
                        + self.policy_weight * policy_term
                        - self.uncertainty_weight * uncertainty
                    )
                    expanded.append({
                        "state": next_state.detach(),
                        "actions": beam["actions"] + [int(action_index)],
                        "score": beam["score"] + local_score,
                        "policy_score": beam["policy_score"] + policy_term,
                        "goal_score": beam["goal_score"] + self.goal_weight * value + self.alignment_weight * alignment,
                        "uncertainty": beam["uncertainty"] + uncertainty,
                        "states": beam["states"] + [next_state.detach()],
                        "error": uncertainty,
                    })
            if not expanded:
                break
            expanded.sort(key=lambda item: item["score"], reverse=True)
            beams = expanded[: self.beam_width]

        best = max(beams, key=lambda item: item["score"])
        first_action = int(best["actions"][0])
        action_name = self.action_names[first_action] if 0 <= first_action < len(self.action_names) else str(first_action)
        self.last_plan = {
            "objective": str(objective_text),
            "action_index": first_action,
            "action_name": action_name,
            "actions": list(best["actions"]),
            "action_names": [self.action_names[item] if 0 <= item < len(self.action_names) else str(item) for item in best["actions"]],
            "score": float(best["score"]),
            "policy_score": float(best["policy_score"]),
            "goal_score": float(best["goal_score"]),
            "uncertainty": float(best["uncertainty"]),
            "horizon": int(self.horizon),
            "beam_width": int(self.beam_width),
            "branch_factor": int(self.branch_factor),
            "entropy": entropy,
            "created": time.time(),
        }
        result = {"enabled": True, **self.last_plan}
        if was_training:
            self.model.train()
        return result

    def _goal_and_action(
        self,
        state: torch.Tensor,
        objective_text: str,
        action_index: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        goal = self._goal_vector(objective_text).detach()
        action = self.model.action_embedding(
            torch.tensor([int(action_index)], dtype=torch.long, device=self.device)
        ).detach()
        return goal, action

    def learn_feedback(
        self,
        state: torch.Tensor,
        objective_text: str,
        action_index: int,
        *,
        target: float,
        steps: int = 1,
    ) -> dict:
        """Train the planner only from observed feedback; no semantic labels are embedded."""
        state = state.detach().to(self.device)
        goal, action = self._goal_and_action(state, objective_text, action_index)
        self.value_head.train()
        losses = []
        target_value = torch.tensor([float(max(0.0, min(1.0, target)))], device=self.device)
        for _ in range(max(1, int(steps))):
            prediction = torch.sigmoid(self.value_head(state, goal, action))
            loss = F.binary_cross_entropy(prediction, target_value)
            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.value_head.parameters(), 1.0)
            self.optimizer.step()
            losses.append(float(loss.detach().cpu()))
            self.training_steps += 1
        self.value_head.eval()
        self.save()
        return {
            "accepted": True,
            "target": float(target_value.item()),
            "loss_before_after": [losses[0], losses[-1]],
            "training_steps": self.training_steps,
        }

    def save(self) -> None:
        self.data_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "schema_version": 1,
            "action_names": list(self.action_names),
            "value_head": self.value_head.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "training_steps": self.training_steps,
        }
        tmp = self.data_path.with_suffix(".tmp")
        torch.save(payload, tmp)
        tmp.replace(self.data_path)

    def load(self) -> None:
        if not self.data_path.exists():
            self.value_head.eval()
            return
        try:
            payload = torch.load(self.data_path, map_location=self.device, weights_only=True)
            if not isinstance(payload, dict):
                return
            saved_actions = payload.get("action_names", [])
            saved_state = payload.get("value_head")
            if isinstance(saved_state, dict):
                if list(saved_actions) == list(self.action_names):
                    self.value_head.load_state_dict(saved_state, strict=False)
                else:
                    # Only reusable action rows are copied when the action interface changed.
                    current = self.value_head.state_dict()
                    compatible = {
                        key: value for key, value in saved_state.items()
                        if key in current and tuple(current[key].shape) == tuple(value.shape)
                    }
                    if compatible:
                        self.value_head.load_state_dict(compatible, strict=False)
            optimizer_state = payload.get("optimizer")
            if isinstance(optimizer_state, dict):
                try:
                    self.optimizer.load_state_dict(optimizer_state)
                except Exception:
                    pass
            self.training_steps = int(payload.get("training_steps", 0))
        except Exception:
            self.value_head = self.value_head.to(self.device)
        self.value_head.eval()
