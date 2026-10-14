"""Generic open-world verification signals.

These metrics do not encode what any task, object, person, or website means. They only
compare what Aletheia predicted or remembered with what she later observed.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, asdict
from typing import Sequence

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class OpenWorldSignals:
    prediction_match: float | None = None
    cross_source_consistency: float | None = None
    reconstruction_match: float | None = None
    self_qa_match: float | None = None

    def finite_values(self) -> list[float]:
        values: list[float] = []
        for value in asdict(self).values():
            if value is not None and math.isfinite(float(value)):
                values.append(max(0.0, min(1.0, float(value))))
        return values

    def aggregate(self) -> float | None:
        values = self.finite_values()
        if not values:
            return None
        # Median is a parameter-free robust aggregator: no semantic weights are hard-coded.
        ordered = sorted(values)
        middle = len(ordered) // 2
        if len(ordered) % 2:
            return ordered[middle]
        return 0.5 * (ordered[middle - 1] + ordered[middle])

    def as_dict(self) -> dict[str, float | None]:
        return {key: (None if value is None else float(value)) for key, value in asdict(self).items()}


class OpenWorldOutcomeEngine:
    """Compute generic verification signals from observations and learned representations."""

    @staticmethod
    def similarity(left: torch.Tensor | None, right: torch.Tensor | None) -> float | None:
        """Return cosine similarity mapped to [0, 1], or None if undefined.

        Missing, empty, mismatched, zero-norm, and non-finite embeddings are not
        negative evidence. Treating them as a numeric score can contaminate a
        learned outcome signal with an artifact of invalid tensor input.
        """
        if not isinstance(left, torch.Tensor) or not isinstance(right, torch.Tensor):
            return None
        if left.numel() == 0 or right.numel() == 0 or left.numel() != right.numel():
            return None
        a = left.detach().float().reshape(-1)
        b = right.detach().float().reshape(-1)
        if not bool(torch.isfinite(a).all()) or not bool(torch.isfinite(b).all()):
            return None
        norm_a = torch.linalg.vector_norm(a)
        norm_b = torch.linalg.vector_norm(b)
        if float(norm_a) <= torch.finfo(a.dtype).eps or float(norm_b) <= torch.finfo(b.dtype).eps:
            return None
        cosine = float(F.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0), dim=-1).item())
        if not math.isfinite(cosine):
            return None
        return max(0.0, min(1.0, 0.5 * (cosine + 1.0)))

    @staticmethod
    def mean_consistency(reference: torch.Tensor | None, candidates: Sequence[torch.Tensor]) -> float | None:
        values = [
            score for item in candidates
            if (score := OpenWorldOutcomeEngine.similarity(reference, item)) is not None
        ]
        if not values:
            return None
        return float(sum(values) / len(values))
