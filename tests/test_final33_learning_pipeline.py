from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from aletheia.learning_pipeline import LearningPipeline, _cap_text, split_source_text


def _paragraphs(prefix: str = "lesson") -> str:
    bodies = [
        f"{prefix} paragraph {index} explains a distinct observation about rivers, weather, and changing landscapes. "
        f"The details in section {index} are written separately for a clean source-level evaluation split."
        for index in range(1, 8)
    ]
    return "\n\n".join(bodies)


class _Memory:
    def __init__(self):
        self.status = []
        self.metrics = []

    def update_learning_status(self, lesson_id, *, accepted, replay_eligible):
        self.status.append((lesson_id, accepted, replay_eligible))

    def update_learning_metrics(self, lesson_id, **kwargs):
        self.metrics.append((lesson_id, kwargs))


class _FakeAgent:
    def __init__(self, bpb_scores: list[float], *, internal_accepted: bool = True):
        self.model = torch.nn.Linear(1, 1, bias=False)
        torch.nn.init.constant_(self.model.weight, 1.0)
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=0.1, momentum=0.1)
        self.state = torch.tensor([0.2])
        self.memory = _Memory()
        self.scores = list(bpb_scores)
        self.internal_accepted = internal_accepted
        self.taught_texts = []
        self.saved = 0
        self.total_tokens = 3
        self.processed_tokens = 4
        self.training_steps = 5
        self.last_language_loss = 1.1
        self.last_teaching_loss = 1.2
        self.last_lesson_id = 6
        self.last_lesson_accepted = True
        self.last_lesson_fixed_gain_delta = 0.1
        self._cognitive_step = 8
        self._unique_training_fingerprints = {"prior"}

    def evaluate_text_bpb(self, text: str) -> float:
        assert text.strip()
        return self.scores.pop(0)

    def evaluate_fixed_reference(self):
        return {"available": True, "model_bpb": 3.0, "gain": 0.1}

    def teach_text(self, lesson, **kwargs):
        self.taught_texts.append(lesson.text)
        with torch.no_grad():
            self.model.weight.add_(0.25)
        self.total_tokens += 100
        self.processed_tokens += 120
        self.training_steps += 2
        self._unique_training_fingerprints.add("new")
        return {
            "lesson_id": 17,
            "lesson_accepted": self.internal_accepted,
            "loss_before": 3.0,
            "loss_after": 2.0,
            "learning_progress": 1.0,
            "unique_tokens_added": 100,
        }

    def save(self):
        self.saved += 1


def test_source_split_is_deterministic_nonempty_and_disjoint():
    source = _paragraphs("unique split")
    first = split_source_text(source, source_hash="abc123")
    second = split_source_text(source, source_hash="abc123")
    assert first == second
    assert first["train_text"] and first["evaluation_text"]
    assert first["strategy"] == "paragraph_grouped"
    assert first["train_sha256"] != first["evaluation_sha256"]
    assert not (set(first["train_text"].split("\n\n")) & set(first["evaluation_text"].split("\n\n")))


def test_training_and_evaluation_text_caps_are_deterministic_and_respect_budget():
    source = "\n\n".join(
        f"Cap test unit {index} contains distinct scientific language about river flow, soil absorption, and measurement uncertainty. "
        f"The separate unit adds observations number {index} without depending on another paragraph."
        for index in range(30)
    )
    first = _cap_text(source, 900, seed="cap-seed")
    second = _cap_text(source, 900, seed="cap-seed")
    assert first == second
    capped_text, was_capped, unit_count = first
    assert was_capped
    assert len(capped_text) <= 900
    assert unit_count >= 1
    source_units = set(source.split("\n\n"))
    assert set(capped_text.split("\n\n")).issubset(source_units)


def test_single_oversized_unit_is_word_bounded_without_exceeding_budget():
    source = " ".join(f"word{index}" for index in range(500))
    capped_text, was_capped, unit_count = _cap_text(source, 500, seed="single-unit")
    assert was_capped
    assert len(capped_text) <= 500
    assert unit_count == 1
    assert capped_text.split() == source.split()[:len(capped_text.split())]


def test_unpunctuated_text_uses_explicitly_weaker_fallback():
    text = " ".join(f"word{index}" for index in range(150))
    split = split_source_text(text, source_hash="unpunctuated")
    assert split["strategy"] == "contiguous_word_fallback_weaker_isolation"
    assert set(split["train_text"].split()).isdisjoint(set(split["evaluation_text"].split()))


def test_unsupported_extensions_are_rejected(tmp_path):
    bad = tmp_path / "payload.exe"
    bad.write_bytes(b"not a supported learning material")
    with LearningPipeline(tmp_path / "project") as pipeline:
        with pytest.raises(ValueError, match="unsupported file type"):
            pipeline.enqueue_file(bad)


def test_duplicate_upload_returns_existing_job(tmp_path):
    source = tmp_path / "notes.txt"
    source.write_text(_paragraphs("duplicate payload"), encoding="utf-8")
    with LearningPipeline(tmp_path / "project") as pipeline:
        first = pipeline.enqueue_file(source)
        second = pipeline.enqueue_file(source)
        assert not first["duplicate"]
        assert second["duplicate"]
        assert second["id"] == first["id"]
        assert len(pipeline.list_jobs()) == 1


def test_queue_survives_restart_and_interrupted_early_stage_recovers(tmp_path):
    root = tmp_path / "project"
    pipeline = LearningPipeline(root)
    item = pipeline.enqueue_text(_paragraphs("restart"))
    pipeline._update(item["id"], state="indexing", progress=28, status_message="simulated interruption")
    pipeline.close()
    with LearningPipeline(root) as restarted:
        current = restarted.get_job(item["id"])
        assert current["state"] == "queued"
        assert "Recovered" in current["status_message"]


def test_interrupted_training_is_quarantined_not_silently_accepted(tmp_path):
    root = tmp_path / "project"
    with LearningPipeline(root) as pipeline:
        item = pipeline.enqueue_text(_paragraphs("interrupted training"))
        pipeline._update(item["id"], state="training", progress=40, status_message="simulated interruption")
    with LearningPipeline(root) as restarted:
        current = restarted.get_job(item["id"])
        assert current["state"] == "failed"
        assert "final job-level acceptance" in current["error"]


def test_opening_queue_does_not_recover_a_job_owned_by_a_live_worker(tmp_path):
    import os

    root = tmp_path / "project"
    with LearningPipeline(root) as first:
        item = first.enqueue_text(_paragraphs("live worker"))
        first._update(
            item["id"], state="training", progress=40, status_message="worker is active", worker_pid=os.getpid()
        )
        with LearningPipeline(root) as second:
            current = second.get_job(item["id"])
            assert current["state"] == "training"
            with pytest.raises(ValueError, match="only queued jobs can run"):
                second.run_job(item["id"], _FakeAgent([2.0, 1.9]))
            assert second.get_job(item["id"])["state"] == "training"


def test_pipeline_trains_only_training_split_and_writes_report(tmp_path):
    root = tmp_path / "project"
    with LearningPipeline(root) as pipeline:
        item = pipeline.enqueue_text(_paragraphs("accepted job"), title="Geography notes", license_name="CC BY 4.0")
        queued = pipeline.get_job(item["id"])
        original = pipeline._row(item["id"])["source_path"]
        assert Path(original).is_file()
        agent = _FakeAgent([2.0, 1.6])
        before_weight = agent.model.weight.detach().clone()
        result = pipeline.run_job(item["id"], agent)
        assert result["state"] == "accepted"
        assert agent.taught_texts == [pipeline._row(item["id"])["train_text"]]
        assert agent.taught_texts[0] != pipeline._row(item["id"])["evaluation_text"]
        assert result["metrics"]["source_holdout_delta_bpb"] == pytest.approx(-0.4)
        assert result["metrics"]["provenance"]["license"] == "CC BY 4.0"
        assert result["report_path"] and Path(result["report_path"]).is_file()
        assert not Path(original).exists()
        assert torch.equal(before_weight, agent.model.weight.detach()) is False
        report = json.loads(Path(result["report_path"]).read_text(encoding="utf-8"))
        assert report["metrics"]["decision"] == "accepted"
        # The report may describe a held-out hash and metrics but never stores the holdout text.
        report_text = Path(result["report_path"]).read_text(encoding="utf-8")
        assert pipeline._row(item["id"])["evaluation_text"] not in report_text


def test_external_holdout_regression_restores_model_and_replay_status(tmp_path):
    root = tmp_path / "project"
    with LearningPipeline(root) as pipeline:
        item = pipeline.enqueue_text(_paragraphs("regression job"))
        agent = _FakeAgent([2.0, 2.4])
        before = {key: value.detach().clone() for key, value in agent.model.state_dict().items()}
        result = pipeline.run_job(item["id"], agent, max_regression_bpb=0.08)
        assert result["state"] == "rejected"
        assert "exceeded tolerance" in result["error"]
        for key, value in agent.model.state_dict().items():
            assert torch.equal(value, before[key])
        assert agent.memory.status[-1] == (17, False, False)
        assert agent.total_tokens == 3
        assert agent.processed_tokens == 4
        assert agent.training_steps == 5
        assert agent._unique_training_fingerprints == {"prior"}
        assert agent.saved >= 1


def test_internal_rejection_cannot_be_overridden_by_external_improvement(tmp_path):
    root = tmp_path / "project"
    with LearningPipeline(root) as pipeline:
        item = pipeline.enqueue_text(_paragraphs("internal rejection"))
        agent = _FakeAgent([2.0, 1.0], internal_accepted=False)
        before = {key: value.detach().clone() for key, value in agent.model.state_dict().items()}
        result = pipeline.run_job(item["id"], agent)
        assert result["state"] == "rejected"
        assert result["metrics"]["external_holdout_accepted"] is True
        assert result["metrics"]["internal_lesson_accepted"] is False
        for key, value in agent.model.state_dict().items():
            assert torch.equal(value, before[key])


def test_corrupt_document_fails_with_report_without_training(tmp_path):
    root = tmp_path / "project"
    corrupt = tmp_path / "corrupt.pdf"
    corrupt.write_bytes(b"%PDF-1.4\nnot a valid document")
    with LearningPipeline(root) as pipeline:
        item = pipeline.enqueue_file(corrupt)
        agent = _FakeAgent([2.0, 1.0])
        result = pipeline.run_job(item["id"], agent)
        assert result["state"] == "failed"
        assert "unreadable PDF" in result["error"] or "no extractable text" in result["error"]
        assert not agent.taught_texts
        assert result["report_path"] and Path(result["report_path"]).is_file()


def test_extracted_content_duplicate_is_rejected_before_model_update(tmp_path):
    root = tmp_path / "project"
    body = _paragraphs("identical normalized content")
    with LearningPipeline(root) as pipeline:
        first = pipeline.enqueue_text(body, title="first.txt")
        first_agent = _FakeAgent([2.0, 1.8])
        first_result = pipeline.run_job(first["id"], first_agent)
        assert first_result["state"] == "accepted"
        second = pipeline.enqueue_text("\n\n" + body + "\n", title="second.txt")
        second_agent = _FakeAgent([2.0, 1.8])
        second_result = pipeline.run_job(second["id"], second_agent)
        assert second_result["state"] == "rejected"
        assert "duplicate of job" in second_result["error"]
        assert not second_agent.taught_texts


def test_learning_report_contains_provenance_and_never_claims_pretrained_model(tmp_path):
    with LearningPipeline(tmp_path / "project") as pipeline:
        item = pipeline.enqueue_text(_paragraphs("provenance"), license_name="user-owned material", source_uri="https://example.invalid/material")
        data = pipeline.get_job(item["id"])
        assert data["provenance"]["license"] == "user-owned material"
        assert data["provenance"]["source_uri"] == "https://example.invalid/material"
        assert "train_text" not in data and "evaluation_text" not in data


def test_repeated_paragraph_units_are_deduplicated_before_split():
    first = "A unique repeated-unit paragraph gives enough detail about measurement, evidence, and observations."
    second = "Another independent paragraph explains a separate process for recording different information clearly."
    text = "\n\n".join([first, second, first])
    split = split_source_text(text, source_hash="repeat-test")
    all_units = split["train_text"].split("\n\n") + split["evaluation_text"].split("\n\n")
    assert len(all_units) == 2
    assert len(set(all_units)) == 2


def test_capped_views_are_rechecked_for_train_holdout_overlap(tmp_path, monkeypatch):
    import aletheia.learning_pipeline as module

    shared = " ".join(f"sharedterm{index}" for index in range(100))

    def fake_cap(text, maximum, *, seed):
        tail = " training-tail distinctwords" if seed.endswith(":train") else " evaluation-tail differentwords"
        return shared + tail, True, 1

    monkeypatch.setattr(module, "_cap_text", fake_cap)
    with LearningPipeline(tmp_path / "project") as pipeline:
        item = pipeline.enqueue_text(_paragraphs("cap overlap guard"))
        agent = _FakeAgent([2.0, 1.8])
        result = pipeline.run_job(item["id"], agent)
        assert result["state"] == "failed"
        assert "substantially overlap" in result["error"]
        assert not agent.taught_texts
        assert agent.scores == [2.0, 1.8]


def test_holdout_already_present_in_accepted_memory_is_rejected(tmp_path):
    import sqlite3

    class MemoryWithDB(_Memory):
        def __init__(self, text):
            super().__init__()
            self.db = sqlite3.connect(":memory:")
            self.db.execute("CREATE TABLE memories(id INTEGER,source TEXT,title TEXT,text TEXT,kind TEXT,accepted INTEGER,created REAL)")
            self.db.execute("INSERT INTO memories VALUES(1,'old-source','old lesson',?,'lesson',1,1.0)", (text,))

    source = _paragraphs("already trained source")
    with LearningPipeline(tmp_path / "project") as pipeline:
        item = pipeline.enqueue_text(source)
        agent = _FakeAgent([2.0, 1.8])
        agent.memory = MemoryWithDB(source)
        result = pipeline.run_job(item["id"], agent)
        assert result["state"] == "rejected"
        assert "holdout contamination" in result["error"]
        assert result["metrics"]["holdout_contamination"]["lesson_id"] == 1
        assert not agent.taught_texts


def test_upload_byte_budget_is_enforced_before_a_job_is_created(tmp_path):
    with LearningPipeline(tmp_path / "project", max_upload_bytes=64) as pipeline:
        with pytest.raises(ValueError, match="upload is .* configured limit"):
            pipeline.enqueue_text("x" * 200)
        assert pipeline.list_jobs() == []


def test_cli_can_queue_a_file_and_persist_job_without_starting_model(tmp_path, monkeypatch, capsys):
    from aletheia import cli

    source = tmp_path / "lecture.md"
    source.write_text(_paragraphs("CLI upload"), encoding="utf-8")
    root = tmp_path / "project"
    monkeypatch.setattr(
        "sys.argv",
        ["aletheia", "--root", str(root), "--learning-add", str(source), "--learning-license", "CC BY 4.0"],
    )
    cli.main()
    printed = json.loads(capsys.readouterr().out)
    assert printed["state"] == "queued"
    assert printed["provenance"]["license"] == "CC BY 4.0"
    with LearningPipeline(root) as pipeline:
        assert pipeline.get_job(printed["id"])["state"] == "queued"
