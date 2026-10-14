#!/usr/bin/env python3
"""Run pytest nodes in isolated per-file batches and write a machine-readable report.

Aletheia's legacy regression suite contains long PyTorch training tests and has shown
process-level hangs when run as one very large pytest session in constrained runtimes.
This runner keeps test-file boundaries and splits unusually large files into smaller
processes. It still executes every collected test node and never turns a timeout or a
failed test into a pass.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import signal
import subprocess
import sys
import tempfile
import time
from collections import OrderedDict
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SUMMARY = re.compile(r"(?P<count>\d+)\s+(?P<kind>passed|failed|skipped|error|xfailed|xpassed)")


def run_isolated(command: list[str], *, cwd: Path, env: dict[str, str] | None = None,
                 timeout: int, text_output: bool = True) -> subprocess.CompletedProcess[str]:
    """Run a command in its own process group with timeout-safe output capture.

    Output is redirected to a temporary file instead of a pipe. This matters when a test
    spawns a detached descendant that inherits stdout: such a descendant can keep a pipe open
    after pytest itself exits, preventing ``communicate()`` from seeing EOF. A file-backed
    capture lets the runner observe the direct process exit independently and collect partial
    output on timeout without waiting for unrelated descendants.
    """
    with tempfile.TemporaryFile(mode="w+b") as capture:
        kwargs = {
            "cwd": cwd,
            "env": env,
            "stdout": capture,
            "stderr": subprocess.STDOUT,
        }
        if os.name == "posix":
            kwargs["start_new_session"] = True
        elif os.name == "nt":
            kwargs["creationflags"] = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
        process = subprocess.Popen(command, **kwargs)
        try:
            process.wait(timeout=max(1, int(timeout)))
        except subprocess.TimeoutExpired:
            if os.name == "posix":
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                except OSError:
                    process.kill()
            else:
                process.kill()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
            capture.flush()
            capture.seek(0)
            raw_output = capture.read()
            captured = raw_output.decode("utf-8", "replace") if text_output else raw_output
            raise subprocess.TimeoutExpired(command, timeout, output=captured) from None
        capture.flush()
        capture.seek(0)
        raw_output = capture.read()
        output = raw_output.decode("utf-8", "replace") if text_output else raw_output
        return subprocess.CompletedProcess(command, process.returncode, output, None)


def display_path(path: Path) -> str:
    """Prefer project-relative paths but allow logs to live outside the checkout."""
    resolved = Path(path).resolve()
    try:
        return resolved.relative_to(ROOT).as_posix()
    except ValueError:
        return str(resolved)


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    temporary.replace(path)


def collect_nodes(timeout: int) -> list[str]:
    completed = run_isolated(
        [sys.executable, "-m", "pytest", "--collect-only", "-q"],
        cwd=ROOT,
        timeout=timeout,
    )
    if completed.returncode != 0:
        raise RuntimeError(f"pytest collection failed (exit {completed.returncode}):\n{completed.stdout[-12000:]}")
    return [line.strip() for line in completed.stdout.splitlines() if "::" in line and not line.lstrip().startswith("<")]


def batch_nodes(nodes: list[str], batch_size: int) -> list[tuple[str, int, list[str]]]:
    files: OrderedDict[str, list[str]] = OrderedDict()
    for node in nodes:
        filename = node.split("::", 1)[0]
        files.setdefault(filename, []).append(node)
    batches: list[tuple[str, int, list[str]]] = []
    for filename, file_nodes in files.items():
        for offset in range(0, len(file_nodes), batch_size):
            chunk = file_nodes[offset:offset + batch_size]
            batches.append((filename, offset // batch_size + 1, chunk))
    return batches


def parse_counts(output: str) -> dict[str, int]:
    counts = {key: 0 for key in ("passed", "failed", "skipped", "error", "xfailed", "xpassed")}
    for match in SUMMARY.finditer(output):
        counts[match.group("kind")] += int(match.group("count"))
    return counts


def verified_counts(output: str, status: str) -> tuple[dict[str, int], dict[str, int] | None]:
    """Never attribute partial output from a timed-out batch to verified totals."""
    observed = parse_counts(output)
    if status == "timeout":
        empty = {key: 0 for key in observed}
        return empty, observed
    return observed, None


def update_coverage_metrics(result: dict, collection_count: int) -> None:
    """Count selected nodes separately from nodes in batches that actually finished."""
    batches = result.get("batches", [])
    result["selected_nodeids"] = sum(int(batch.get("node_count", 0)) for batch in batches)
    result["completed_nodes"] = sum(
        int(batch.get("node_count", 0)) for batch in batches
        if batch.get("status") in {"passed", "failed"}
    )
    result["timed_out_nodes"] = sum(
        int(batch.get("node_count", 0)) for batch in batches
        if batch.get("status") == "timeout"
    )
    result["uncovered_nodes"] = max(0, int(collection_count) - result["completed_nodes"])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-size", type=int, default=20, help="maximum node IDs per subprocess (default: 20)")
    parser.add_argument("--test-timeout", type=int, default=180, help="timeout in seconds per batch")
    parser.add_argument("--release", default="FINAL33", help="release label recorded in the report (default: FINAL33)")
    parser.add_argument("--report", default="FINAL34_BATCH_TEST_RESULTS.json", help="JSON report path under project root")
    parser.add_argument("--log-dir", default="test-results/final34-batches", help="batch log directory under project root")
    args = parser.parse_args()
    batch_size = max(1, min(50, int(args.batch_size)))
    report_path = (ROOT / args.report).resolve() if not Path(args.report).is_absolute() else Path(args.report)
    log_dir = (ROOT / args.log_dir).resolve() if not Path(args.log_dir).is_absolute() else Path(args.log_dir)
    started = time.time()

    nodes = collect_nodes(timeout=max(15, int(args.test_timeout)))
    batches = batch_nodes(nodes, batch_size)
    result = {
        "schema_version": 1,
        "release": str(args.release),
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "collection_count": len(nodes),
        "strategy": "isolated per-file pytest node batches; no test failures or timeouts are hidden",
        "single_process_full_suite_claim": False,
        "batch_size": batch_size,
        "batch_count": len(batches),
        "counts": {key: 0 for key in ("passed", "failed", "skipped", "error", "xfailed", "xpassed")},
        "batches": [],
    }
    atomic_json(report_path, result)
    log_dir.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    # Limit oversubscription in small CPU-only CI/VM environments; these settings do not
    # change model architecture, weights, random seeds, or test assertions.
    env.setdefault("OMP_NUM_THREADS", "1")
    env.setdefault("MKL_NUM_THREADS", "1")
    env.setdefault("OPENBLAS_NUM_THREADS", "1")

    for index, (filename, sub_index, chunk) in enumerate(batches, start=1):
        batch_start = time.time()
        command = [sys.executable, "-m", "pytest", "-q", *chunk]
        partial_counts = None
        try:
            completed = run_isolated(
                command, cwd=ROOT, env=env,
                timeout=max(10, int(args.test_timeout)),
            )
            output = completed.stdout
            counts = parse_counts(output)
            status = "passed" if completed.returncode == 0 and counts["failed"] == 0 and counts["error"] == 0 else "failed"
            exit_code = completed.returncode
        except subprocess.TimeoutExpired as exc:
            output = (exc.stdout.decode("utf-8", "replace") if isinstance(exc.stdout, bytes) else (exc.stdout or ""))
            output += f"\nBATCH TIMEOUT after {args.test_timeout} seconds.\n"
            counts, partial_counts = verified_counts(output, "timeout")
            status = "timeout"
            exit_code = None
        slug = re.sub(r"[^A-Za-z0-9_.-]+", "_", filename.replace("/", "_"))
        log_path = log_dir / f"{index:02d}_{slug}_part{sub_index}.log"
        log_path.write_text(output, encoding="utf-8", errors="replace")
        item = {
            "index": index,
            "file": filename,
            "part": sub_index,
            "node_count": len(chunk),
            "status": status,
            "exit_code": exit_code,
            "elapsed_seconds": round(time.time() - batch_start, 3),
            "counts": counts,
            "partial_observed_counts": partial_counts if status == "timeout" else None,
            "log": display_path(log_path),
            "nodeids": chunk,
        }
        result["batches"].append(item)
        for key, value in counts.items():
            result["counts"][key] += value
        update_coverage_metrics(result, len(nodes))
        result["elapsed_seconds"] = round(time.time() - started, 3)
        result["updated_at_utc"] = datetime.now(timezone.utc).isoformat()
        atomic_json(report_path, result)
        print(f"[{index}/{len(batches)}] {filename} part {sub_index}: {status}; {counts}; {item['elapsed_seconds']}s", flush=True)
        if output.strip():
            print(output[-3500:], flush=True)

    result["elapsed_seconds"] = round(time.time() - started, 3)
    result["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
    result["all_batches_completed"] = all(item["status"] != "timeout" for item in result["batches"])
    result["all_batches_passed"] = all(item["status"] == "passed" for item in result["batches"])
    update_coverage_metrics(result, len(nodes))
    result["all_tests_passed"] = result["all_batches_passed"] and result["uncovered_nodes"] == 0
    atomic_json(report_path, result)
    print(f"FINAL: completed {result['completed_nodes']}/{len(nodes)} collected tests; selected={result['selected_nodeids']}; timed_out={result['timed_out_nodes']}; counts={result['counts']}; all_tests_passed={result['all_tests_passed']}", flush=True)
    return 0 if result["all_tests_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
