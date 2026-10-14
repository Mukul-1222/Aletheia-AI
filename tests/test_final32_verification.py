from __future__ import annotations

import math
from pathlib import Path

import pytest
import torch

from aletheia import __version__
from aletheia.evaluation_protocol import validate_source_separated_manifest
from aletheia.objective_verifiers import ObjectiveVerifier
from aletheia.objectives import ObjectiveManager
from aletheia.outcomes import OpenWorldOutcomeEngine


def test_release_version_and_project_metadata_match():
    assert __version__ == "0.34.0"
    project = (Path(__file__).resolve().parents[1] / "pyproject.toml").read_text(encoding="utf-8")
    assert 'version = "0.34.0"' in project


def test_objective_verification_rejects_non_finite_scores_and_thresholds():
    for value in (math.nan, math.inf, -math.inf):
        with pytest.raises(ValueError, match="finite"):
            ObjectiveVerifier.evaluate("threshold", value)
    for threshold in (math.nan, math.inf, -0.01, 1.01):
        with pytest.raises(ValueError, match="threshold"):
            ObjectiveVerifier.evaluate("threshold", 0.5, threshold=threshold)


def test_objective_verification_does_not_coerce_untrusted_confirmation_or_test_evidence():
    assert not ObjectiveVerifier.evaluate("human", 1.0, human_confirmed="false").verified
    assert not ObjectiveVerifier.evaluate("unit_test", 1.0, evidence={"passed": "false"}).verified
    assert not ObjectiveVerifier.evaluate("unit_test", 1.0, evidence={"passed": True, "exit_code": 1}).verified
    assert not ObjectiveVerifier.evaluate("unit_test", 1.0, evidence={"passed": True, "failed": 1}).verified
    assert ObjectiveVerifier.evaluate("unit_test", 1.0, evidence={"passed": True, "exit_code": 0, "failed": 0}).complete


def test_objective_manager_does_not_treat_truthy_text_as_human_confirmation(tmp_path):
    manager = ObjectiveManager(tmp_path / "objectives.json")
    objective = manager.add("confirm result", verifier_type="human")
    result = manager.verify(objective.id, outcome=1.0, human_confirmed="false")
    assert not result["verified"]
    assert not result["complete"]


def test_manifest_reports_non_mapping_rows_instead_of_crashing():
    errors = validate_source_separated_manifest([None, 7, "bad"])
    assert len(errors) == 3
    assert all("record must be a mapping" in item for item in errors)


def test_open_world_similarity_treats_invalid_embeddings_as_unknown():
    assert OpenWorldOutcomeEngine.similarity(None, torch.ones(4)) is None
    assert OpenWorldOutcomeEngine.similarity(torch.zeros(4), torch.ones(4)) is None
    assert OpenWorldOutcomeEngine.similarity(torch.ones(3), torch.ones(4)) is None
    assert OpenWorldOutcomeEngine.similarity(torch.tensor([math.nan]), torch.ones(1)) is None
    assert OpenWorldOutcomeEngine.mean_consistency(torch.zeros(4), [torch.ones(4)]) is None


def test_open_world_consistency_skips_invalid_candidates_without_promoting_them():
    result = OpenWorldOutcomeEngine.mean_consistency(
        torch.ones(4), [torch.ones(3), torch.ones(4), torch.zeros(4)]
    )
    assert result == 1.0
