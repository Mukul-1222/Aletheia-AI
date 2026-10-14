from __future__ import annotations

import json
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path

from .memory import lexical_terms


@dataclass
class Objective:
    id: str
    text: str
    created: float
    updated: float
    status: str = "active"
    priority: float = 1.0
    progress: float = 0.0
    attempts: int = 0
    last_action: str = ""
    knowledge_gaps: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    outcome_count: int = 0
    outcome_mean: float = 0.0
    progress_source: str = "unobserved"
    verifier_type: str = "manual"
    completion_threshold: float = 0.95
    required_consecutive: int = 1
    verification_count: int = 0
    verification_streak: int = 0
    last_verification_reason: str = ""


class ObjectiveManager:
    """Persistent objective state without encoding domain-specific meanings."""

    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._objectives: list[Objective] = []
        self._load()

    def _load(self) -> None:
        if not self.path.exists():
            return
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
            self._objectives = [Objective(**item) for item in raw.get("objectives", [])]
        except (OSError, ValueError, TypeError):
            self._objectives = []

    def _save(self) -> None:
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(
            json.dumps({"objectives": [asdict(item) for item in self._objectives]}, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        tmp.replace(self.path)

    def add(self, text: str, *, priority: float = 1.0, verifier_type: str = "manual", completion_threshold: float = 0.95, required_consecutive: int = 1) -> Objective:
        text = " ".join(str(text).split()).strip()
        if not text:
            raise ValueError("objective text cannot be empty")
        now = time.time()
        objective = Objective(
            id=uuid.uuid4().hex[:12],
            text=text,
            created=now,
            updated=now,
            priority=float(priority),
            verifier_type=str(verifier_type or "manual"),
            completion_threshold=max(0.0, min(1.0, float(completion_threshold))),
            required_consecutive=max(1, int(required_consecutive)),
        )
        self._objectives.append(objective)
        self._save()
        return objective

    def current(self) -> Objective | None:
        active = [item for item in self._objectives if item.status == "active"]
        if not active:
            return None
        return max(active, key=lambda item: (item.priority, item.updated))

    def all(self) -> list[Objective]:
        return list(self._objectives)

    def update(self, objective_id: str, **changes) -> Objective:
        for item in self._objectives:
            if item.id != objective_id:
                continue
            for key, value in changes.items():
                if hasattr(item, key):
                    setattr(item, key, value)
            item.updated = time.time()
            self._save()
            return item
        raise KeyError(objective_id)

    def complete(self, objective_id: str, *, reason: str = "verified") -> Objective:
        objective = self.update(objective_id, status="completed", progress=1.0, progress_source="verified", last_verification_reason=str(reason))
        return objective

    def verify(self, objective_id: str, *, outcome: float, human_confirmed: bool = False, evidence: dict | None = None) -> dict:
        from .objective_verifiers import ObjectiveVerifier

        objective = next((item for item in self._objectives if item.id == objective_id), None)
        if objective is None:
            raise KeyError(objective_id)
        result = ObjectiveVerifier.evaluate(
            objective.verifier_type,
            float(outcome),
            threshold=float(objective.completion_threshold),
            human_confirmed=human_confirmed is True,
            evidence=evidence,
        )
        count = int(objective.verification_count) + (1 if result.verified else 0)
        streak = int(objective.verification_streak) + 1 if result.verified else 0
        required = max(1, int(objective.required_consecutive))
        progress = max(float(objective.progress), result.score if result.verified else float(objective.progress))
        self.update(
            objective_id,
            verification_count=count,
            verification_streak=streak,
            progress=min(0.99, progress),
            progress_source=objective.progress_source,
            last_verification_reason=result.reason,
        )
        if result.complete and streak >= required:
            objective = self.complete(objective_id, reason=result.reason)
        else:
            objective = next(item for item in self._objectives if item.id == objective_id)
        return {
            "verified": result.verified,
            "complete": objective.status == "completed",
            "score": result.score,
            "reason": result.reason,
            "evidence": result.evidence,
            "verification_count": int(objective.verification_count),
            "verification_streak": int(objective.verification_streak),
            "status": objective.status,
        }

    def pause(self, objective_id: str) -> Objective:
        return self.update(objective_id, status="paused")

    def infer_gaps(self, objective: Objective, known_text: str, *, limit: int = 16) -> list[str]:
        known = set(lexical_terms(known_text))
        candidates = []
        for term in lexical_terms(objective.text):
            if term not in known:
                candidates.append(term)
        unique = []
        seen = set()
        for term in candidates:
            if term not in seen:
                seen.add(term)
                unique.append(term)
        objective.knowledge_gaps = unique[: max(0, int(limit))]
        objective.updated = time.time()
        self._save()
        return objective.knowledge_gaps
