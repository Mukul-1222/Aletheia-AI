from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("aletheia_test_batch_runner", ROOT / "tools" / "run_test_batches.py")
assert SPEC is not None and SPEC.loader is not None
RUNNER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(RUNNER)


def test_runner_report_defaults_are_for_current_release():
    source = (ROOT / "tools" / "run_test_batches.py").read_text(encoding="utf-8")
    assert 'default="FINAL34_BATCH_TEST_RESULTS.json"' in source
    assert 'default="test-results/final34-batches"' in source


@pytest.mark.skipif(os.name != "posix", reason="process-group descendant termination is POSIX-specific")
def test_runner_can_record_logs_outside_project_root(tmp_path):
    external = tmp_path / "outside" / "batch.log"
    assert RUNNER.display_path(external) == str(external.resolve())
    internal = ROOT / "test-results" / "batch.log"
    assert RUNNER.display_path(internal) == "test-results/batch.log"


@pytest.mark.skipif(os.name != "posix", reason="process-group descendant termination is POSIX-specific")
def test_runner_timeout_kills_process_group_without_pipe_hang(tmp_path):
    command = [
        sys.executable,
        "-c",
        "import subprocess,sys,time; subprocess.Popen([sys.executable,'-c','import time; time.sleep(30)']); print('child-started',flush=True); time.sleep(30)",
    ]
    started = time.monotonic()
    with pytest.raises(subprocess.TimeoutExpired) as error:
        RUNNER.run_isolated(command, cwd=tmp_path, timeout=1)
    assert time.monotonic() - started < 8
    assert error.value.timeout == 1


def test_evolution_benchmark_has_no_fixed_semantic_task_answers():
    from aletheia.evolution import GenerationalEvolution

    benchmark = GenerationalEvolution._benchmark_script()
    compile(benchmark, "<evolution-benchmark>", "exec")
    assert "The capital of France" not in benchmark
    assert "task_success" not in benchmark
    assert "agent.evaluate_fixed_reference()" in benchmark
    assert "vision_ssl_loss" in benchmark
    assert "audio_ssl_loss" in benchmark


def test_evolution_score_ignores_unregistered_metrics():
    from types import SimpleNamespace
    from aletheia.evolution import GenerationalEvolution

    lab = GenerationalEvolution.__new__(GenerationalEvolution)
    runtime = SimpleNamespace(evolution_benchmark_weights={"task_success": 1.0, "language_bpb": 1.0})
    result = lab.score_candidate(
        {"task_success": 0.1, "language_bpb": 2.0},
        {"task_success": 0.9, "language_bpb": 1.0},
        runtime,
    )
    assert "task_success" not in result["ratios"]
    assert result["ratios"]["language_bpb"] > 1.0


def test_runner_coverage_excludes_timed_out_node_batches():
    result = {
        "batches": [
            {"node_count": 3, "status": "passed"},
            {"node_count": 2, "status": "timeout"},
            {"node_count": 1, "status": "failed"},
        ]
    }
    RUNNER.update_coverage_metrics(result, collection_count=6)
    assert result["selected_nodeids"] == 6
    assert result["completed_nodes"] == 4
    assert result["timed_out_nodes"] == 2
    assert result["uncovered_nodes"] == 2


def test_capability_report_defaults_to_current_release_without_claiming_tested():
    from aletheia.capability_report import build_capability_report

    report = build_capability_report(ROOT)
    assert report["release"] == "FINAL34"
    assert all(item["status"] != "tested" for item in report["capabilities"])


def test_timeout_partial_progress_never_counts_as_verified_passes():
    verified, observed = RUNNER.verified_counts("5 passed, 1 skipped in 18.1s", "timeout")
    assert verified["passed"] == 0
    assert verified["skipped"] == 0
    assert observed is not None and observed["passed"] == 5 and observed["skipped"] == 1
    completed, partial = RUNNER.verified_counts("5 passed, 1 skipped in 2.0s", "passed")
    assert completed["passed"] == 5 and completed["skipped"] == 1
    assert partial is None


@pytest.mark.skipif(os.name != "posix", reason="detached-child process-group behavior is POSIX-specific")
def test_runner_timeout_does_not_wait_for_detached_stdout_inheritor(tmp_path):
    pid_file = tmp_path / "detached-child.pid"
    child_code = (
        "import os,time; "
        f"open({str(pid_file)!r},'w').write(str(os.getpid())); "
        "print('detached-child', flush=True); time.sleep(30)"
    )
    parent_code = (
        "import subprocess,sys,time; "
        f"subprocess.Popen([sys.executable,'-c',{child_code!r}], start_new_session=True); "
        "print('parent-started',flush=True); time.sleep(30)"
    )
    started = time.monotonic()
    try:
        with pytest.raises(subprocess.TimeoutExpired) as error:
            RUNNER.run_isolated([sys.executable, "-c", parent_code], cwd=tmp_path, timeout=1)
        assert time.monotonic() - started < 8
        assert "parent-started" in (error.value.stdout or "")
    finally:
        if pid_file.exists():
            try:
                os.kill(int(pid_file.read_text(encoding="utf-8")), 9)
            except (ProcessLookupError, ValueError, OSError):
                pass
