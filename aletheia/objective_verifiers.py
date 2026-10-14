from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Any, Mapping


@dataclass(frozen=True)
class VerificationResult:
    verified: bool
    score: float
    complete: bool = False
    reason: str = ""
    evidence: dict[str, Any] = field(default_factory=dict)


class ObjectiveVerifier:
    """Generic, persisted objective verification contracts.

    The verifier has no domain knowledge. Domain-specific code can supply its own
    observed score/evidence through the same interface. Built-ins are deliberately
    generic so that the core learner does not encode task semantics.
    """

    @staticmethod
    def evaluate(verifier_type: str, outcome: float, *, threshold: float = 0.95, human_confirmed: bool = False, evidence: Mapping[str, Any] | None = None) -> VerificationResult:
        kind = str(verifier_type or "manual").strip().lower()
        raw_score = float(outcome)
        if not math.isfinite(raw_score):
            raise ValueError("objective outcome must be finite")
        score = max(0.0, min(1.0, raw_score))
        payload = dict(evidence or {})
        # Confirmation and test evidence are control inputs, not text to coerce.
        # In particular, strings such as "false" must never count as approval.
        confirmed = human_confirmed is True
        if kind == "threshold":
            limit = float(threshold)
            if not math.isfinite(limit) or not 0.0 <= limit <= 1.0:
                raise ValueError("verification threshold must be finite and between 0 and 1")
            ok = score >= limit
            return VerificationResult(ok, score, complete=ok, reason="threshold_met" if ok else "below_threshold", evidence=payload)
        if kind == "human":
            return VerificationResult(confirmed, score, complete=confirmed, reason="human_confirmed" if confirmed else "awaiting_human_confirmation", evidence=payload)
        if kind == "answer":
            actual = " ".join(str(payload.get("actual", "")).split()).casefold()
            expected = " ".join(str(payload.get("expected", "")).split()).casefold()
            ok = bool(expected) and actual == expected
            return VerificationResult(ok, 1.0 if ok else 0.0, complete=ok, reason="answer_match" if ok else "answer_mismatch", evidence=payload)
        if kind == "unit_test":
            # Require a real boolean. When execution metadata is supplied, reject
            # contradictory evidence rather than trusting one positive field.
            ok = payload.get("passed") is True
            exit_code = payload.get("exit_code")
            if exit_code is not None and (isinstance(exit_code, bool) or not isinstance(exit_code, int) or exit_code != 0):
                ok = False
            for key in ("failed", "errors", "xpassed"):
                count = payload.get(key)
                if count is not None and (isinstance(count, bool) or not isinstance(count, int) or count != 0):
                    ok = False
            return VerificationResult(ok, 1.0 if ok else 0.0, complete=ok, reason="tests_passed" if ok else "tests_failed_or_untrusted_evidence", evidence=payload)
        if kind == "manual":
            # Manual verification records evidence and may update progress, but it
            # does not silently close the objective. Use verifier_type="human" or
            # "threshold" for explicit automatic completion.
            return VerificationResult(confirmed, score, complete=False, reason="human_feedback_recorded" if confirmed else "manual_verification_required", evidence=payload)
        raise ValueError(f"unsupported objective verifier type: {verifier_type}")
