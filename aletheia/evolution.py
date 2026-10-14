from __future__ import annotations

import copy
import json
import hashlib
import math
import os
import random
import shutil
import subprocess
import sys
import tempfile
import time
import ast
import inspect
import re
import shlex

import torch
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass
class EvolutionResult:
    changed: bool
    parameter: str
    before: float
    after: float
    score_before: float
    score_after: float
    accepted: bool
    reason: str


class EvolutionLab:
    """Development laboratory for parameter and source self-improvement.

    Every source change is evaluated in an isolated successor tree. The active
    installation is changed only after compilation and the project's tests pass.
    Accepted files are backed up before the atomic replacement so a bad successor
    can always be restored.
    """

    MUTATIONS = {
        "learning_rate": (0.5, 0.75, 1.25, 1.5),
        "policy_learning_rate": (0.5, 0.75, 1.25, 1.5),
        "replay_mix_fraction": (0.0, 0.15, 0.45, 0.6),
        "generation_temperature": (0.60, 0.75, 1.00, 1.15),
    }

    IGNORE_DIRS = {
        ".git", ".venv", "venv", "__pycache__", ".pytest_cache",
        "data", "dist", "build", "evolution_backups", "candidates",
    }
    # Self-editing is an explicit allowlist, not a denylist. Security, reward,
    # objective, browser, configuration, and orchestration code are not editable by
    # the model-generated source evolution path.
    SELF_EDITABLE_SOURCE_PATHS = {
        Path("aletheia/learning_adapter.py"),
        Path("aletheia/experimental.py"),
    }
    IMMUTABLE_SOURCE_PATHS = {
        Path("aletheia/evolution.py"),
        Path("aletheia/__main__.py"),
        Path("aletheia/evaluation.py"),
        Path("aletheia/eval_corpus.py"),
        Path("aletheia/web.py"),
        Path("aletheia/browser.py"),
        Path("aletheia/objective_value.py"),
        Path("aletheia/objectives.py"),
        Path("aletheia/outcomes.py"),
        Path("aletheia/config.py"),
        Path("aletheia/agent.py"),
    }

    @staticmethod
    def recover_pending_startup(root: Path) -> dict:
        """Restore a previous source version when the last accepted successor cannot import."""
        root = Path(root).resolve()
        pending = root / "data" / "evolution" / "pending_successor.json"
        if not pending.exists():
            return {"recovered": False, "status": "clean"}
        try:
            record = json.loads(pending.read_text(encoding="utf-8"))
            relative = Path(record["path"])
            backup = Path(record.get("backup", "")) if record.get("backup") else None
            target = root / relative
            if backup and backup.exists():
                shutil.copy2(backup, target)
                failed = pending.with_name(f"failed_successor_{int(time.time())}.json")
                pending.replace(failed)
                return {"recovered": True, "status": "restored", "path": relative.as_posix(), "backup": str(backup)}
            pending.unlink(missing_ok=True)
            return {"recovered": False, "status": "no_backup", "path": relative.as_posix()}
        except Exception as exc:
            return {"recovered": False, "status": "recovery_error", "error": repr(exc)}

    def __init__(self, root: Path):
        self.root = root.resolve()
        self.self_edit_allowlist = self._load_self_edit_allowlist()
        self.history_path = self.root / "data" / "evolution" / "history.jsonl"
        self.history_path.parent.mkdir(parents=True, exist_ok=True)
        self.stage_root = self.history_path.parent / "candidates"
        self.stage_root.mkdir(parents=True, exist_ok=True)
        self.backup_root = self.history_path.parent / "backups"
        self.backup_root.mkdir(parents=True, exist_ok=True)

    def _load_self_edit_allowlist(self) -> set[Path]:
        """Load the explicit self-edit allowlist; malformed/missing config disables edits."""
        default = set(self.SELF_EDITABLE_SOURCE_PATHS)
        path = self.root / "config.json"
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(payload, dict) or not isinstance(payload.get("runtime"), dict):
                return set()
            values = payload["runtime"].get("evolution_self_edit_allowlist")
            if not isinstance(values, list) or not values:
                return set()
            configured: set[Path] = set()
            for item in values:
                if not isinstance(item, str) or not item.strip():
                    continue
                rel = Path(item)
                if rel.is_absolute() or ".." in rel.parts or rel.suffix != ".py" or rel.parts[:1] != ("aletheia",):
                    continue
                configured.add(rel)
            # The configuration can narrow the source-code edit scope, never expand it.
            return configured & default
        except (OSError, json.JSONDecodeError, TypeError, ValueError):
            return set()

    def _record(self, payload: dict) -> None:
        with self.history_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps({"time": time.time(), **payload}, ensure_ascii=False) + "\n")

    def candidate_values(self, parameter: str, current: float) -> list[float]:
        values = []
        for item in self.MUTATIONS.get(parameter, ()):
            candidate = (
                float(current) * float(item)
                if item < 2.0 and parameter in {"learning_rate", "policy_learning_rate"}
                else float(item)
            )
            if candidate > 0:
                values.append(candidate)
        return sorted(set(values))

    def evolve_runtime_parameter(self, agent, parameter: str) -> EvolutionResult:
        """Compare runtime candidates on identical paired seeds and compute budgets.

        Every arm starts from the same model, optimizer, EMA and accounting state. A
        candidate is accepted only when it beats the control on every predeclared seed
        and clears the configured minimum mean BPB gain. Unavailable/incomplete scores
        are inconclusive and cannot win.
        """
        if not hasattr(agent.config.runtime, parameter):
            raise AttributeError(parameter)
        if parameter != "learning_rate":
            raise ValueError(
                "runtime parameter evolution currently measures only fixed-reference language BPB; "
                "only learning_rate is eligible. Policy, vision, audio, speech, TTS, and planning "
                "rates require their own task-specific held-out metric."
            )

        current = float(getattr(agent.config.runtime, parameter))
        candidates = [current] + [value for value in self.candidate_values(parameter, current) if value != current]
        original_runtime = copy.deepcopy(vars(agent.config.runtime))
        original_model = {key: value.detach().cpu().clone() for key, value in agent.model.state_dict().items()}
        original_optimizer = copy.deepcopy(agent.optimizer.state_dict())
        original_lrs = [group.get("lr") for group in agent.optimizer.param_groups]
        original_model_training = bool(agent.model.training)
        original_scaler = copy.deepcopy(agent.scaler.state_dict()) if hasattr(agent, "scaler") else None
        side_names = (
            "training_steps", "total_tokens", "processed_tokens", "last_language_loss",
            "last_replay_stats", "last_eval_bpb", "eval_baseline_bpb", "eval_reference_bytes",
            "state", "last_prediction_error", "last_learning_progress",
        )
        original_side_state = {}
        for name in side_names:
            if hasattr(agent, name):
                value = getattr(agent, name)
                if torch.is_tensor(value):
                    original_side_state[name] = value.detach().clone()
                else:
                    original_side_state[name] = copy.deepcopy(value)
        original_ema = None
        if getattr(agent, "_ema_shadow", None) is not None:
            original_ema = {key: value.detach().cpu().clone() for key, value in agent._ema_shadow.items()}

        raw_seeds = getattr(agent.config.runtime, "evolution_eval_seeds", [1337, 2027, 3037])
        try:
            seeds = list(dict.fromkeys(int(seed) for seed in raw_seeds))
        except (TypeError, ValueError):
            seeds = []
        if len(seeds) < 3:
            # Fewer than three paired trials is not enough evidence for promotion.
            seeds = [1337, 2027, 3037]
        min_gain = max(0.0, float(getattr(agent.config.runtime, "evolution_runtime_min_bpb_gain", 0.005)))
        per_candidate: dict[float, list[float]] = {}
        seed_report: dict[str, list[dict]] = {}
        baseline = math.inf
        best_score = math.inf
        best = current
        reason = "no_candidate_improved_on_all_paired_seeds"

        def restore_starting_state() -> None:
            vars(agent.config.runtime).clear()
            vars(agent.config.runtime).update(copy.deepcopy(original_runtime))
            agent.model.load_state_dict(original_model, strict=False)
            try:
                agent.optimizer.load_state_dict(copy.deepcopy(original_optimizer))
            except (ValueError, RuntimeError):
                pass
            for index, group in enumerate(agent.optimizer.param_groups):
                if index < len(original_lrs) and original_lrs[index] is not None:
                    group["lr"] = original_lrs[index]
            if original_scaler is not None and hasattr(agent, "scaler"):
                try:
                    agent.scaler.load_state_dict(copy.deepcopy(original_scaler))
                except (ValueError, RuntimeError):
                    pass
            if original_ema is not None:
                agent._ema_shadow = {key: value.detach().clone().to(agent.device) for key, value in original_ema.items()}
            for name, value in original_side_state.items():
                setattr(agent, name, value.detach().clone() if torch.is_tensor(value) else copy.deepcopy(value))
            agent.model.train(original_model_training)

        try:
            for value in candidates:
                scores: list[float] = []
                observations: list[dict] = []
                for seed in seeds:
                    restore_starting_state()
                    random.seed(seed)
                    torch.manual_seed(seed)
                    if torch.cuda.is_available():
                        torch.cuda.manual_seed_all(seed)
                    setattr(agent.config.runtime, parameter, value)
                    if parameter == "learning_rate" and agent.optimizer.param_groups:
                        agent.optimizer.param_groups[0]["lr"] = value
                    try:
                        if agent.memory.lesson_count() > 0:
                            agent._train_recent_trajectory(max(1, int(agent.config.runtime.train_steps_per_observation)))
                        evaluation = agent.evaluate_fixed_reference()
                        measured = evaluation.get("model_bpb") if isinstance(evaluation, dict) and evaluation.get("available", True) else None
                        score = float(measured) if measured is not None and math.isfinite(float(measured)) else math.inf
                    except Exception as exc:
                        score = math.inf
                        observations.append({"seed": seed, "score": None, "error": repr(exc)[:500]})
                    else:
                        observations.append({"seed": seed, "score": score if math.isfinite(score) else None})
                    scores.append(score)
                per_candidate[value] = scores
                seed_report[str(value)] = observations

            baseline_scores = per_candidate.get(current, [])
            baseline = sum(baseline_scores) / len(baseline_scores) if baseline_scores and all(math.isfinite(x) for x in baseline_scores) else math.inf
            best_score = baseline
            for value in candidates[1:]:
                scores = per_candidate.get(value, [])
                if not baseline_scores or len(scores) != len(baseline_scores):
                    continue
                if not all(math.isfinite(x) for x in scores + baseline_scores):
                    continue
                paired_gains = [control - candidate for control, candidate in zip(baseline_scores, scores)]
                mean_gain = sum(paired_gains) / len(paired_gains)
                # Require a positive paired result on every predeclared seed as well
                # as an actionable mean gain. This is deliberately conservative.
                if all(gain > 0.0 for gain in paired_gains) and mean_gain >= min_gain and (baseline - sum(scores) / len(scores)) > (baseline - best_score):
                    best = value
                    best_score = sum(scores) / len(scores)
                    reason = "candidate_improved_all_paired_seeds"
        finally:
            restore_starting_state()

        accepted = best != current and math.isfinite(baseline) and math.isfinite(best_score) and (baseline - best_score) >= min_gain
        if not accepted:
            best = current
            best_score = baseline
            reason = "no_candidate_improved_on_all_paired_seeds"
        setattr(agent.config.runtime, parameter, best)
        if parameter == "learning_rate" and agent.optimizer.param_groups:
            agent.optimizer.param_groups[0]["lr"] = best
        try:
            agent.evaluate_fixed_reference()
        except Exception:
            pass
        agent.model.train(original_model_training)

        result = EvolutionResult(
            changed=accepted,
            parameter=parameter,
            before=current,
            after=best,
            score_before=baseline,
            score_after=best_score,
            accepted=accepted,
            reason=reason,
        )
        self._record({
            "event": "runtime_evolution", **asdict(result),
            "paired_seed_results": seed_report,
            "evaluation_seeds": seeds,
            "minimum_mean_bpb_gain": min_gain,
            "acceptance_rule": "wins_on_every_paired_seed_and_minimum_mean_gain",
        })
        return result

    def _copy_tree(self, destination: Path) -> None:
        def ignore(current_root: str, names: list[str]) -> set[str]:
            current = Path(current_root)
            ignored = {name for name in names if name in self.IGNORE_DIRS}
            for name in names:
                path = current / name
                if path.is_file() and name.endswith(".pt"):
                    # The learned model checkpoint is useful for candidate bpb
                    # evaluation. Other runtime checkpoints/state are deliberately
                    # excluded from the isolated source tree.
                    if path.relative_to(self.root).as_posix() != "data/aletheia.pt":
                        ignored.add(name)
            return ignored

        shutil.copytree(self.root, destination, ignore=ignore)
        # Only the learned language checkpoint crosses into the successor tree;
        # persistent memory, telemetry, identity state, and other runtime data stay out.
        checkpoint = self.root / "data" / "aletheia.pt"
        if checkpoint.exists():
            target = destination / "data" / "aletheia.pt"
            target.parent.mkdir(parents=True, exist_ok=True)
            payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
            if not isinstance(payload, dict) or not isinstance(payload.get("state_dict"), dict):
                raise ValueError("active checkpoint is not a valid model checkpoint")
            torch.save({
                "schema_version": payload.get("schema_version"),
                "config": payload.get("config"),
                "action_names": payload.get("action_names"),
                "state_dict": payload.get("state_dict"),
            }, target)

        # The model checkpoint alone is insufficient when a learned BPE tokenizer is
        # active: candidate BPB must use the exact same token-to-byte mapping. Copy only
        # the configured tokenizer, and normalize its candidate config path so evaluation
        # cannot silently fall back to byte tokenization or read assets outside the tree.
        config_path = self.root / "config.json"
        if config_path.is_file():
            try:
                config_payload = json.loads(config_path.read_text(encoding="utf-8"))
                runtime = config_payload.get("runtime", {}) if isinstance(config_payload, dict) else {}
                configured = Path(str(runtime.get("tokenizer_path", "data/tokenizer.json")))
                if configured.is_absolute():
                    configured_resolved = configured.resolve()
                    try:
                        configured_rel = configured_resolved.relative_to(self.root)
                    except ValueError as exc:
                        raise ValueError("candidate evaluation tokenizer must be inside the project root") from exc
                else:
                    configured_rel = configured
                if configured_rel.is_absolute() or ".." in configured_rel.parts:
                    raise ValueError("candidate evaluation tokenizer path must be a safe project-relative path")
                source_tokenizer = (self.root / configured_rel).resolve()
                if self.root not in source_tokenizer.parents:
                    raise ValueError("candidate evaluation tokenizer path escapes the project root")
                if source_tokenizer.is_file():
                    target_tokenizer = destination / configured_rel
                    target_tokenizer.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(source_tokenizer, target_tokenizer)
                runtime["tokenizer_path"] = configured_rel.as_posix()
                config_path_candidate = destination / "config.json"
                config_payload["runtime"] = runtime
                config_path_candidate.write_text(json.dumps(config_payload, indent=2), encoding="utf-8")
            except (OSError, json.JSONDecodeError, TypeError, AttributeError) as exc:
                raise ValueError(f"candidate evaluation configuration is invalid: {exc}") from exc

    def _candidate_test_files(self, candidate_root: Path) -> list[str]:
        tests_dir = candidate_root / "tests"
        return sorted(
            path.relative_to(candidate_root).as_posix()
            for path in tests_dir.glob("test_*.py")
            if path.is_file()
        )

    def _candidate_language_bpb_script(self) -> str:
        """Evaluate candidate weights using the exact active-tokenizer held-out BPB protocol."""
        return r"""
import json
from pathlib import Path
import torch
from aletheia.config import load_config
from aletheia.evaluation import load_reference
from aletheia.model import AletheiaModel
from aletheia.tokenizer import PAD, load_tokenizer
root = Path('.').resolve()
cfg = load_config(root / 'config.json')
reference = load_reference(root, cfg.runtime.language_eval_path, 0)
raw = reference.encode('utf-8', errors='replace')
if len(raw) < 8:
    raise ValueError('fixed evaluation corpus is too small')
split = min(len(raw) - 1, max(1, int(len(raw) * 0.8)))
eval_bytes = raw[split - 1:]
tokenizer_path = Path(cfg.runtime.tokenizer_path)
if not tokenizer_path.is_absolute():
    tokenizer_path = root / tokenizer_path
tokenizer = load_tokenizer(tokenizer_path)
cfg.model.vocab_size = int(tokenizer.vocab_size)
checkpoint = torch.load(root / 'data' / 'aletheia.pt', map_location='cpu', weights_only=True)
if not isinstance(checkpoint, dict) or not isinstance(checkpoint.get('state_dict'), dict):
    raise ValueError('candidate checkpoint is invalid')
model = AletheiaModel(cfg.model, list(checkpoint.get('action_names') or cfg.capabilities.keys()), tokenizer=tokenizer)
model.load_state_dict(checkpoint['state_dict'], strict=True)
model.eval()
body = [int(value) for value in tokenizer.encode_bytes(eval_bytes, bos=False, eos=False) if int(value) not in (257, 258, PAD)]
width = max(1, int(model.cfg.context) - 2)
nll_bits = 0.0
byte_count = 0.0
with torch.inference_mode():
    for start in range(0, max(len(body), 1), width):
        piece = body[start:start + width]
        is_final = start + len(piece) >= len(body)
        ids = [257, *piece]
        if is_final:
            ids.append(258)
        tokens = torch.tensor([ids], dtype=torch.long)
        chunk_nll, chunk_bytes = model.language_bpb_totals(tokens)
        nll_bits += chunk_nll
        byte_count += chunk_bytes
if byte_count <= 0:
    raise ValueError('fixed evaluation produced no scorable bytes')
print(json.dumps({'model_bpb': float(nll_bits / byte_count), 'reference_bytes': len(raw), 'test_bytes': int(byte_count), 'tokenizer': getattr(tokenizer, 'name', 'unknown')}))
"""

    def _candidate_language_bpb(self, candidate_root: Path, timeout: int) -> tuple[float | None, str]:
        try:
            result = self._run_python(candidate_root, ["-c", self._candidate_language_bpb_script()], timeout=timeout, hard_sandbox=False)
        except subprocess.TimeoutExpired as exc:
            return None, f"language evaluation timeout after {timeout}s: {exc}"
        except Exception as exc:
            return None, repr(exc)
        output = (result.stdout + "\n" + result.stderr).strip()
        if result.returncode != 0:
            return None, output[-10000:]
        for line in reversed(result.stdout.splitlines()):
            try:
                payload = json.loads(line.strip())
            except json.JSONDecodeError:
                continue
            if payload.get("model_bpb") is not None:
                return float(payload["model_bpb"]), output[-10000:]
        return None, output[-10000:]

    def _active_language_bpb(self, agent, timeout: int) -> tuple[float | None, str]:
        agent.save()
        return self._candidate_language_bpb(self.root, timeout)

    def _editable_path(self, relative_path: str, *, autonomous: bool = False) -> Path:
        rel = Path(relative_path)
        if rel.is_absolute() or ".." in rel.parts or rel.suffix != ".py" or not rel.parts or rel.parts[0] != "aletheia":
            raise ValueError("candidate path must be a relative Python path under aletheia/")
        if rel in self.IMMUTABLE_SOURCE_PATHS:
            raise ValueError(f"source target is immutable: {rel.as_posix()}")
        if autonomous and rel not in self.self_edit_allowlist:
            raise ValueError(f"source target is outside the self-edit allowlist: {rel.as_posix()}")
        return rel

    @staticmethod
    def _sandbox_env(candidate_root: Path) -> dict[str, str]:
        keep = {"SYSTEMROOT", "WINDIR", "TEMP", "TMP", "HOME", "USERPROFILE", "LANG", "LC_ALL", "PYTHONNOUSERSITE"}
        env = {k: v for k, v in os.environ.items() if k in keep}
        env.update({
            "PYTHONPATH": str(candidate_root),
            "PYTHONNOUSERSITE": "1",
            "NO_PROXY": "*",
            "no_proxy": "*",
        })
        return env

    @staticmethod
    def _hard_sandbox_prefix(candidate_root: Path) -> list[str] | None:
        """Return a Linux namespace sandbox prefix when bubblewrap is available.

        A model-generated candidate is not considered safely executable unless a
        hard sandbox can disable network access and isolate its writable tree.
        """
        # bubblewrap relies on Linux namespaces and is not a Windows/macOS sandbox.
        # Do not mistake a similarly named executable for a verified isolation backend.
        if os.name != "posix" or not sys.platform.startswith("linux"):
            return None
        bwrap = shutil.which("bwrap")
        if not bwrap:
            return None
        # Never bind-mount the host root: that would make SSH keys, home directories
        # and other readable host files visible to model-generated code. Bind only the
        # system runtime locations needed by the interpreter and the candidate tree.
        required_roots = [Path("/usr"), Path("/bin"), Path("/lib"), Path("/lib64"), Path("/etc")]
        # The active interpreter is often inside a virtual environment under /opt or
        # a user-managed prefix, not under /usr. Bind only that interpreter environment
        # read-only; without it, the sandbox can start and then fail importing Python.
        for candidate in (Path(sys.prefix), Path(sys.base_prefix), Path(sys.executable).resolve().parent):
            if candidate.exists() and not any(candidate == existing or existing in candidate.parents for existing in required_roots):
                required_roots.append(candidate)
        prefix = [bwrap, "--die-with-parent", "--new-session", "--unshare-net"]
        for source in required_roots:
            if source.exists():
                prefix.extend(["--ro-bind", str(source), str(source)])
        prefix.extend([
            "--dev", "/dev", "--proc", "/proc", "--tmpfs", "/tmp",
            "--bind", str(candidate_root), "/work", "--chdir", "/work",
        ])
        return prefix

    @staticmethod
    def _run_python(candidate_root: Path, code_args: list[str], *, timeout: int, hard_sandbox: bool, env_extra: dict[str, str] | None = None) -> subprocess.CompletedProcess[str]:
        prefix = EvolutionLab._hard_sandbox_prefix(candidate_root) if hard_sandbox else None
        if hard_sandbox and prefix is None:
            raise RuntimeError("hard candidate sandbox unavailable; this build requires Linux bubblewrap (no Docker/Windows backend is implemented)")
        command = (prefix or []) + [sys.executable, *code_args]
        # Avoid subprocess ``preexec_fn``: it runs Python code after fork and can
        # deadlock if native libraries (e.g. PyTorch/OpenMP) already hold locks in
        # other threads. Apply POSIX limits through prlimit instead. The external
        # subprocess timeout remains active even on systems without prlimit.
        limits = []
        prlimit = shutil.which("prlimit") if os.name == "posix" else None
        if prlimit:
            limits = [
                prlimit,
                f"--cpu={max(2, int(timeout))}:{max(2, int(timeout) + 2)}",
                f"--as={3 * 1024**3}:{3 * 1024**3}",
                f"--fsize={256 * 1024**2}:{256 * 1024**2}",
                "--",
            ]
        command = (prefix or []) + limits + [sys.executable, *code_args]
        env = EvolutionLab._sandbox_env(candidate_root)
        if env_extra:
            env.update({str(k): str(v) for k, v in env_extra.items()})
        return subprocess.run(
            command,
            cwd=candidate_root,
            env=env,
            capture_output=True,
            text=True,
            timeout=timeout,
        )

    @staticmethod
    def _validate_patch_text(generated: str, relative_path: str) -> tuple[bool, str]:
        rel = Path(relative_path)
        if "\x00" in generated:
            return False, "patch contains NUL bytes"
        file_headers = re.findall(r"^diff --git a/(.*?) b/(.*?)$", generated, flags=re.MULTILINE)
        plus_headers = re.findall(r"^\+\+\+ b/(.*?)$", generated, flags=re.MULTILINE)
        minus_headers = re.findall(r"^--- a/(.*?)$", generated, flags=re.MULTILINE)
        if len(file_headers) > 1 or len(plus_headers) != 1 or len(minus_headers) != 1:
            return False, "patch must modify exactly one existing file"
        if plus_headers[0] != rel.as_posix() or minus_headers[0] != rel.as_posix():
            return False, "patch target does not exactly match requested file"
        if file_headers and file_headers[0] != (rel.as_posix(), rel.as_posix()):
            return False, "git patch header target mismatch"
        for path in plus_headers + minus_headers:
            candidate = Path(path)
            if candidate.is_absolute() or ".." in candidate.parts:
                return False, "patch path traversal rejected"
        forbidden_headers = re.findall(r"^\+\+\+ /dev/null$|^--- /dev/null$|^rename from |^rename to |^new file mode |^deleted file mode ", generated, flags=re.MULTILINE)
        if forbidden_headers:
            return False, "patch may not create, delete, or rename files"
        return True, "ok"

    @staticmethod
    def _bounded_source_prompt(agent, relative_path: str, objective_text: str, instruction: str, focused: str) -> str:
        """Build a source-edit prompt whose actual tokenized size fits the model."""
        tokenizer = agent.tokenizer
        context_limit = max(64, int(agent.model.cfg.context))
        requested_reply = max(64, int(getattr(agent.config.runtime, "chat_max_response_tokens", 256)))
        reply_reserve = min(max(64, context_limit // 3), requested_reply)
        prompt_budget = max(32, context_limit - reply_reserve)
        suffix = "\nEND SOURCE EXCERPT\n"
        objective = str(objective_text)[:180]
        requested_change = str(instruction)[:280]

        def header(objective_value: str, change_value: str) -> str:
            return (
                "Output one unified diff only. Edit one existing file; do not create/delete/rename.\n"
                f"Objective: {objective_value}\nRequested change: {change_value}\n"
                f"Target file: {relative_path}\n"
                "Keep public interfaces unless necessary. Do not use pretrained/external models.\n"
                "SOURCE:\n"
            )

        prefix = header(objective, requested_change)
        # Shrink descriptive text before sacrificing the fixed instructions.
        while len(tokenizer.encode(prefix + suffix, bos=False, eos=False)) + 1 > prompt_budget:
            if objective:
                objective = objective[:-min(24, len(objective))]
            elif len(requested_change) > 32:
                requested_change = requested_change[:-min(24, len(requested_change) - 32)]
            else:
                raise ValueError("source-edit instructions do not fit the model context")
            prefix = header(objective, requested_change)

        remaining = max(0, prompt_budget - 1 - len(tokenizer.encode(prefix + suffix, bos=False, eos=False)))
        source_ids = tokenizer.encode(str(focused), bos=False, eos=False)
        kept = source_ids[:remaining]
        while True:
            excerpt = tokenizer.decode(kept) if kept else ""
            prompt = prefix + excerpt + suffix
            if len(tokenizer.encode(prompt, bos=False, eos=False)) + 1 <= prompt_budget:
                return prompt
            if not kept:
                raise ValueError("source-edit prompt cannot be fitted to the model context")
            # Decoding a partial UTF-8 byte sequence can expand into U+FFFD. Trim
            # slightly more than the measured overflow and re-check the real encoding.
            overflow = len(tokenizer.encode(prompt, bos=False, eos=False)) + 1 - prompt_budget
            kept = kept[: max(0, len(kept) - max(1, overflow + 4))]

    @staticmethod
    def _focused_source_context(source: str, instruction: str, max_chars: int) -> str:
        """Keep the instruction + a relevant AST region inside the model context window."""
        lines = source.splitlines()
        try:
            tree = ast.parse(source)
            names = {m.group(1) for m in re.finditer(r"\b([A-Za-z_]\w*)\b", instruction)}
            candidates = [node for node in ast.walk(tree) if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and node.name in names]
            if candidates:
                node = min(candidates, key=lambda n: getattr(n, "end_lineno", n.lineno) - n.lineno)
                start = max(0, int(node.lineno) - 8)
                end = min(len(lines), int(getattr(node, "end_lineno", node.lineno)) + 8)
                snippet = "\n".join(lines[start:end])
                return snippet[:max_chars]
        except SyntaxError:
            pass
        return "\n".join(lines[:min(len(lines), max(1, max_chars // 80))])[:max_chars]

    def stage_source_patch(self, relative_path: str, content: str, *, rationale: str = "") -> Path:
        rel = self._editable_path(relative_path, autonomous=False)
        target = self.stage_root / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
        meta = target.with_suffix(target.suffix + ".meta.json")
        meta.write_text(
            json.dumps({"path": str(rel), "rationale": rationale, "created": time.time()}, indent=2),
            encoding="utf-8",
        )
        return target

    def _materialize_candidate(self, relative_path: str, candidate_content: str) -> Path:
        stamp = time.strftime("%Y%m%d_%H%M%S") + f"_{time.time_ns() % 1_000_000:06d}"
        candidate_root = self.stage_root / f"run_{stamp}"
        self._copy_tree(candidate_root)
        destination = candidate_root / relative_path
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(candidate_content, encoding="utf-8")
        return candidate_root

    def validate_source_patch(
        self,
        candidate: Path,
        relative_path: str | None = None,
        *,
        run_tests: bool = True,
        timeout: int = 180,
        active_bpb: float | None = None,
        bpb_tolerance: float = 1e-4,
        require_hard_sandbox: bool = False,
    ) -> dict:
        candidate = candidate.resolve()
        if self.stage_root not in candidate.parents:
            raise ValueError("candidate must live under the evolution candidate directory")
        if candidate.suffix != ".py":
            raise ValueError("only Python source patches are supported")
        rel = relative_path or candidate.relative_to(self.stage_root).as_posix()
        # Validate the target before joining it to a candidate root. Without this, a
        # caller-supplied ../ path could escape the isolated tree during materialization.
        safe_rel = self._editable_path(rel, autonomous=bool(require_hard_sandbox))
        rel = safe_rel.as_posix()
        content = candidate.read_text(encoding="utf-8")
        candidate_sha256 = hashlib.sha256(content.encode("utf-8")).hexdigest()
        original_path = (self.root / safe_rel).resolve()
        if self.root not in original_path.parents:
            raise ValueError("candidate target escapes project root")
        active_source_sha256 = hashlib.sha256(original_path.read_bytes()).hexdigest() if original_path.is_file() else None
        if original_path.exists() and content == original_path.read_text(encoding="utf-8"):
            return {"validated": False, "path": rel, "reason": "candidate_is_identical_to_original", "compile": False, "tests_run": False, "tests_passed": False, "test_files": [], "candidate_sha256": candidate_sha256, "active_source_sha256": active_source_sha256}
        candidate_root = self._materialize_candidate(rel, content)
        if require_hard_sandbox and self._hard_sandbox_prefix(candidate_root) is None:
            shutil.rmtree(candidate_root, ignore_errors=True)
            return {"validated": False, "path": rel, "reason": "hard_sandbox_unavailable", "compile": False, "tests_run": False, "tests_passed": False, "test_files": []}
        result = {"validated": False, "path": rel, "compile": False, "tests_run": False, "tests_passed": False, "test_files": [], "test_output": "", "candidate_root": str(candidate_root), "sandbox": "hard" if require_hard_sandbox else "process", "candidate_sha256": candidate_sha256, "active_source_sha256": active_source_sha256}
        try:
            try:
                compile_result=self._run_python(candidate_root,["-m","compileall","-q","."],timeout=timeout,hard_sandbox=require_hard_sandbox)
            except subprocess.TimeoutExpired as exc:
                result['test_output']=f'compile timeout after {timeout}s: {exc}'
                return result
            result['compile']=compile_result.returncode==0; result['validated']=result['compile']
            if not result['compile']:
                result['test_output']=(compile_result.stdout+compile_result.stderr)[-6000:]
                return result
            if run_tests:
                test_files=self._candidate_test_files(candidate_root)
                result['test_files']=test_files
                if not test_files:
                    result['test_output']='No tests/test_*.py files were found in candidate tree'
                    return result
                try:
                    test_result=self._run_python(candidate_root,["-m","pytest","-q",*test_files],timeout=timeout,hard_sandbox=require_hard_sandbox)
                except subprocess.TimeoutExpired as exc:
                    result.update(tests_run=True,test_output=f'pytest timeout after {timeout}s: {exc}')
                    return result
                result.update(tests_run=True,tests_passed=test_result.returncode==0,test_output=(test_result.stdout+'\n'+test_result.stderr)[-10000:])
                result['validated']=result['validated'] and result['tests_passed']
            if result['validated'] and active_bpb is not None:
                if require_hard_sandbox:
                    bpb_result = self._run_python(candidate_root,["-c", self._candidate_language_bpb_script()],timeout=timeout,hard_sandbox=True)
                    bpb_output = (bpb_result.stdout + "\n" + bpb_result.stderr).strip()
                    candidate_bpb = None
                    for _line in reversed(bpb_result.stdout.splitlines()):
                        try:
                            _payload = json.loads(_line.strip())
                        except json.JSONDecodeError:
                            continue
                        if _payload.get("model_bpb") is not None:
                            candidate_bpb = float(_payload["model_bpb"]); break
                else:
                    candidate_bpb,bpb_output=self._candidate_language_bpb(candidate_root,timeout)
                result['candidate_bpb']=candidate_bpb; result['bpb_output']=bpb_output
                if candidate_bpb is None:
                    result['bpb_gate']='unavailable'; result['validated']=False
                else:
                    result['bpb_regression']=float(candidate_bpb-active_bpb)
                    result['bpb_gate']='rejected' if candidate_bpb > active_bpb+max(0.0,float(bpb_tolerance)) else 'passed'
                    if result['bpb_gate']=='rejected': result['validated']=False
            elif result['validated']:
                result['bpb_gate']='skipped_no_reference'
            return result
        finally:
            self._record({'event':'source_validation',**result})
            shutil.rmtree(candidate_root,ignore_errors=True)

    def commit_source_patch(self, candidate: Path, relative_path: str, validation: dict | None = None) -> dict:
        candidate = candidate.resolve()
        if validation is not None and not validation.get("validated"):
            raise ValueError("candidate failed validation and cannot be committed")
        if validation is not None and validation.get("self_edit") and validation.get("sandbox") != "hard":
            raise ValueError("model-generated source commits require hard sandbox validation")
        if validation is not None:
            if not validation.get("validated"):
                raise ValueError("candidate validation record is not successful")
            actual_candidate_hash = hashlib.sha256(candidate.read_bytes()).hexdigest() if candidate.is_file() else ""
            expected_candidate_hash = str(validation.get("candidate_sha256", ""))
            if not expected_candidate_hash or actual_candidate_hash != expected_candidate_hash:
                raise ValueError("candidate content changed after validation")
        autonomous = bool(validation and validation.get("self_edit", False))
        rel = self._editable_path(relative_path, autonomous=autonomous)
        destination = (self.root / rel).resolve()
        if self.root not in destination.parents:
            raise ValueError("commit path escapes project root")
        if not candidate.exists():
            raise FileNotFoundError(candidate)
        if validation is not None:
            expected_active_hash = validation.get("active_source_sha256")
            if expected_active_hash is None:
                if destination.exists():
                    raise ValueError("active source appeared after candidate validation")
            elif not destination.is_file() or hashlib.sha256(destination.read_bytes()).hexdigest() != str(expected_active_hash):
                raise ValueError("active source changed after candidate validation")

        timestamp = time.strftime("%Y%m%d_%H%M%S") + f"_{time.time_ns() % 1_000_000:06d}"
        backup = self.backup_root / timestamp / rel
        backup.parent.mkdir(parents=True, exist_ok=True)
        destination_existed = destination.exists()
        if destination_existed:
            shutil.copy2(destination, backup)

        destination.parent.mkdir(parents=True, exist_ok=True)
        pending = self.history_path.parent / "pending_successor.json"
        pending.write_text(json.dumps({
            "path": rel.as_posix(),
            "backup": str(backup) if destination_existed else "",
            "created": time.time(),
        }, indent=2), encoding="utf-8")
        temp = destination.with_suffix(destination.suffix + ".evolution.tmp")
        temp.write_text(candidate.read_text(encoding="utf-8"), encoding="utf-8")
        os.replace(temp, destination)
        result = {"committed": True, "path": rel.as_posix(), "backup": str(backup) if destination_existed else "", "restart_required": True, "pending_successor": str(pending)}
        self._record({"event": "source_commit", **result})
        return result

    def approve_pending_source_change(self) -> dict:
        """Approve only the exact candidate that was independently revalidated.

        Pending files are mutable runtime state, so neither a previously recorded
        boolean nor a path alone is sufficient approval evidence. Recheck hashes,
        the unchanged active source, hard-sandbox availability, compile/tests, and the
        fixed-reference language gate immediately before committing.
        """
        pending_path = self.history_path.parent / "pending_source_approval.json"
        if not pending_path.exists():
            return {"approved": False, "reason": "no_pending_source_approval"}
        try:
            payload = json.loads(pending_path.read_text(encoding="utf-8"))
        except (OSError, ValueError, TypeError) as exc:
            return {"approved": False, "reason": "pending_approval_record_invalid", "error": repr(exc)}
        candidate = Path(str(payload.get("candidate", ""))).resolve()
        rel = str(payload.get("path", ""))
        validation = dict(payload.get("validation", {}))
        try:
            safe_rel = self._editable_path(rel, autonomous=True)
        except (ValueError, TypeError) as exc:
            return {"approved": False, "reason": "pending_target_rejected", "error": str(exc)}
        if self.stage_root not in candidate.parents or candidate.suffix != ".py":
            return {"approved": False, "reason": "pending_candidate_outside_stage_root"}
        if not candidate.is_file():
            return {"approved": False, "reason": "pending_candidate_missing"}
        candidate_hash = hashlib.sha256(candidate.read_bytes()).hexdigest()
        if candidate_hash != str(payload.get("candidate_sha256", "")):
            return {"approved": False, "reason": "pending_candidate_changed_after_validation"}
        active_source = (self.root / safe_rel).resolve()
        if self.root not in active_source.parents or not active_source.is_file():
            return {"approved": False, "reason": "active_source_missing_or_outside_root"}
        active_hash = hashlib.sha256(active_source.read_bytes()).hexdigest()
        if active_hash != str(payload.get("active_source_sha256", "")):
            return {"approved": False, "reason": "active_source_changed_after_candidate_validation"}
        if validation.get("sandbox") != "hard" or not validation.get("validated"):
            return {"approved": False, "reason": "candidate_lacks_hard_sandbox_validation"}
        active_bpb = payload.get("active_bpb")
        if active_bpb is None:
            return {"approved": False, "reason": "fixed_reference_score_missing"}
        tolerance = max(0.0, float(payload.get("bpb_tolerance", 1e-4)))
        fresh = self.validate_source_patch(
            candidate, safe_rel.as_posix(), run_tests=True,
            timeout=max(1, int(payload.get("timeout", 300))),
            active_bpb=float(active_bpb), bpb_tolerance=tolerance,
            require_hard_sandbox=True,
        )
        if not fresh.get("validated") or fresh.get("sandbox") != "hard":
            return {"approved": False, "reason": "candidate_failed_revalidation", "validation": fresh}
        fresh["self_edit"] = True
        result = self.commit_source_patch(candidate, safe_rel.as_posix(), fresh)
        candidate.unlink(missing_ok=True)
        candidate.with_suffix(candidate.suffix + ".meta.json").unlink(missing_ok=True)
        pending_path.unlink(missing_ok=True)
        return {"approved": True, "revalidation": fresh, **result}

    def propose_source_change(self, agent, relative_path: str, instruction: str, *, max_source_chars: int = 4200) -> dict:
        rel = self._editable_path(relative_path, autonomous=True)
        source = (self.root / rel).resolve()
        if self.root not in source.parents or not source.exists():
            raise FileNotFoundError(source)
        original = source.read_text(encoding="utf-8")
        original_sha256 = hashlib.sha256(original.encode("utf-8")).hexdigest()
        pending = self.history_path.parent / "pending_source_approval.json"
        if pending.exists():
            return {"changed": False, "accepted": False, "target": rel.as_posix(), "reason": "another_source_change_is_awaiting_human_approval"}
        context_chars = min(int(max_source_chars), int(getattr(agent.config.runtime, "evolution_source_context_chars", 4200)))
        objective = agent.objectives.current()
        objective_text = objective.text if objective else "Improve the system's general developmental learning capability."
        focused = self._focused_source_context(original, instruction, context_chars)
        try:
            prompt = self._bounded_source_prompt(agent, rel.as_posix(), objective_text, instruction, focused)
        except ValueError as exc:
            return {"changed": False, "accepted": False, "reason": "source_edit_prompt_exceeds_context", "error": str(exc)}
        generated = agent.model.generate(
            prompt,
            max_new_tokens=min(12000, max(1024, int(agent.config.runtime.chat_max_response_tokens) * 20)),
            temperature=0.65,
            top_k=max(16, int(agent.config.runtime.chat_top_k)),
            device=agent.device,
            state=agent.state,
        ).strip()
        if generated.startswith("```"):
            lines = generated.splitlines()
            if lines and lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            generated = "\n".join(lines).strip()
        if len(generated) < 32:
            return {"changed": False, "accepted": False, "reason": "self_model_generated_too_little_code"}

        candidate_content = generated
        if not generated.startswith("--- ") or "+++ " not in generated:
            return {"changed": False, "accepted": False, "reason": "self_model_must_generate_unified_diff"}
        if generated.startswith("--- ") and "+++ " in generated:
            # The small from-scratch model sometimes emits an incomplete patch;
            # use git apply inside an isolated copy and fall back to rejection.
            with tempfile.TemporaryDirectory(prefix="aletheia_patch_") as tmp:
                tmp_root = Path(tmp)
                temp_source = tmp_root / rel
                temp_source.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, temp_source)
                patch_file = tmp_root / "change.diff"
                patch_file.write_text(generated, encoding="utf-8")
                valid_patch, patch_reason = self._validate_patch_text(generated, rel.as_posix())
                if not valid_patch:
                    return {"changed": False, "accepted": False, "reason": "generated_patch_rejected", "details": patch_reason}
                check = subprocess.run(["git", "apply", "--check", str(patch_file)], cwd=tmp_root, capture_output=True, text=True)
                if check.returncode != 0:
                    return {"changed": False, "accepted": False, "reason": "generated_patch_rejected", "details": check.stderr.strip()}
                apply = subprocess.run(["git", "apply", str(patch_file)], cwd=tmp_root, capture_output=True, text=True)
                if apply.returncode != 0:
                    return {"changed": False, "accepted": False, "reason": "generated_patch_apply_failed", "details": apply.stderr.strip()}
                candidate_content = (tmp_root / rel).read_text(encoding="utf-8")

        timeout = int(getattr(agent.config.runtime, "evolution_test_timeout", 300))
        active_bpb, active_output = self._active_language_bpb(agent, timeout)
        if active_bpb is None:
            return {
                "changed": False,
                "accepted": False,
                "target": rel.as_posix(),
                "reason": "source_evolution_requires_a_fixed_language_reference",
                "evaluation_output": active_output[-10000:],
            }
        staged = self.stage_source_patch(rel.as_posix(), candidate_content, rationale=instruction)
        try:
            validation = self.validate_source_patch(
                staged,
                rel.as_posix(),
                run_tests=True,
                active_bpb=float(active_bpb),
                bpb_tolerance=float(getattr(agent.config.runtime, "evolution_source_bpb_tolerance", 1e-4)),
                timeout=timeout,
                require_hard_sandbox=True,
            )
            validation["self_edit"] = True
            if not validation.get("validated"):
                return {
                    "changed": False,
                    "accepted": False,
                    "target": rel.as_posix(),
                    "validation": validation,
                    "reason": "candidate_failed_isolated_validation",
                }
            # FINAL28 never commits model-generated source changes automatically,
            # even if a local config accidentally disables the approval flag.
            if not bool(getattr(agent.config.runtime, "evolution_require_human_approval", True)):
                return {
                    "changed": False, "accepted": False, "target": rel.as_posix(),
                    "validation": validation,
                    "reason": "automatic_source_edit_commit_disabled_in_final28",
                }
            validation["active_bpb"] = float(active_bpb)
            validation["bpb_tolerance"] = float(getattr(agent.config.runtime, "evolution_source_bpb_tolerance", 1e-4))
            payload = {
                "path": rel.as_posix(), "candidate": str(staged),
                "candidate_sha256": hashlib.sha256(staged.read_bytes()).hexdigest(),
                "active_source_sha256": original_sha256,
                "active_bpb": float(active_bpb),
                "bpb_tolerance": float(getattr(agent.config.runtime, "evolution_source_bpb_tolerance", 1e-4)),
                "timeout": timeout, "validation": validation,
                "instruction": instruction[:2000], "created": time.time(),
            }
            pending.parent.mkdir(parents=True, exist_ok=True)
            _pending_tmp = pending.with_suffix(".tmp")
            _pending_tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
            os.replace(_pending_tmp, pending)
            return {
                "changed": True, "accepted": False, "awaiting_approval": True,
                "target": rel.as_posix(), "validation": validation,
                "pending_approval": str(pending),
                "reason": "candidate_validated_awaiting_human_approval",
            }
        finally:
            if not bool((self.history_path.parent / "pending_source_approval.json").exists()):
                staged.unlink(missing_ok=True)
                staged.with_suffix(staged.suffix + ".meta.json").unlink(missing_ok=True)
        result = {
            "changed": True,
            "accepted": True,
            "target": rel.as_posix(),
            "validation": validation,
            "commit": committed,
            "reason": "candidate_passed_isolated_compile_and_tests",
        }
        self._record({"event": "model_source_evolution", **result})
        return result


@dataclass
class GenerationCandidate:
    candidate_id: str
    mutation: str
    seed: int
    path: str
    manifest: dict


class GenerationalEvolution:
    """Full-state local generational evolution.

    A generation is a set of isolated clones carrying the learned model, perception,
    speech, TTS, planner, memory, identity, objective and continuity state forward.
    Candidates differ only through generic numerical mutations or runtime-scale
    experiments; no world facts, task semantics or domain rules are embedded here.
    """

    STATE_EXCLUDED_DIRS = {
        "evolution", "browser_profile", "__pycache__", ".pytest_cache", "tmp", "cache"
    }
    VOLATILE_FILE_NAMES = {
        ".DS_Store"
    }

    def __init__(self, root: Path):
        self.root = Path(root).resolve()
        self.lab = EvolutionLab(self.root)
        self.evolution_root = self.root / "data" / "evolution"
        self.generations_root = self.evolution_root / "generations"
        self.generations_root.mkdir(parents=True, exist_ok=True)
        self.current_path = self.evolution_root / "current_generation.json"

    @staticmethod
    def _slug(value: str) -> str:
        return "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in str(value))[:80].strip("_") or "candidate"

    def current_generation(self) -> dict:
        if not self.current_path.exists():
            return {"generation": 0, "candidate_id": "root", "status": "active"}
        try:
            payload = json.loads(self.current_path.read_text(encoding="utf-8"))
            return payload if isinstance(payload, dict) else {"generation": 0, "candidate_id": "root", "status": "active"}
        except Exception:
            return {"generation": 0, "candidate_id": "root", "status": "corrupt_metadata"}

    @staticmethod
    def _iter_copy_source(root: Path):
        for path in root.iterdir():
            if path.name in {".git", ".venv", "venv", "__pycache__", ".pytest_cache", "dist", "build"}:
                continue
            yield path

    def _copy_tree_with_learned_state(self, destination: Path) -> list[str]:
        """Clone source + all learned state while excluding external/browser session state."""
        destination.mkdir(parents=True, exist_ok=True)
        migrated: list[str] = []

        def ignore(current_root: str, names: list[str]) -> set[str]:
            ignored: set[str] = set()
            current = Path(current_root)
            for name in names:
                rel = (current / name).relative_to(self.root)
                if rel.parts and rel.parts[0] in {".git", ".venv", "venv", "dist", "build"}:
                    ignored.add(name)
                    continue
                if rel.parts and rel.parts[0] == "data":
                    if len(rel.parts) >= 2 and rel.parts[1] in self.STATE_EXCLUDED_DIRS:
                        ignored.add(name)
                        continue
                if name in self.VOLATILE_FILE_NAMES:
                    ignored.add(name)
            return ignored

        shutil.copytree(self.root, destination, ignore=ignore, dirs_exist_ok=True)
        data_root = destination / "data"
        if data_root.exists():
            for path in data_root.rglob("*"):
                if path.is_file():
                    rel = path.relative_to(destination).as_posix()
                    if "/evolution/" in f"/{rel}" or rel.endswith("/browser_profile"):
                        continue
                    migrated.append(rel)
        return sorted(migrated)

    def _next_generation_id(self) -> int:
        current = self.current_generation()
        base = int(current.get("generation", 0) or 0)
        existing = []
        for item in self.generations_root.glob("generation_*"):
            try:
                existing.append(int(item.name.split("_", 1)[1]))
            except (IndexError, ValueError):
                continue
        return max([base, *existing], default=0) + 1

    def _mutate_checkpoint(self, path: Path, seed: int, scale: float, *, scopes: tuple[str, ...] = ()) -> dict:
        if not path.exists():
            return {"applied": False, "reason": "checkpoint_missing", "path": path.name}
        payload = torch.load(path, map_location="cpu", weights_only=True)
        if not isinstance(payload, dict):
            return {"applied": False, "reason": "checkpoint_not_mapping", "path": path.name}
        generator = torch.Generator(device="cpu")
        generator.manual_seed(int(seed))
        changed = 0
        total = 0
        for container_key in ("state_dict", "model", "vision", "audio", "object_discovery", "value_head"):
            value = payload.get(container_key)
            if not isinstance(value, dict):
                continue
            if scopes and container_key not in scopes:
                continue
            for key, tensor in list(value.items()):
                if not isinstance(tensor, torch.Tensor) or not tensor.is_floating_point() or tensor.numel() == 0:
                    continue
                std = float(tensor.float().std().item()) if tensor.numel() > 1 else float(abs(tensor.float().item()))
                amplitude = float(scale) * max(std, 1e-4)
                noise = torch.randn(tensor.shape, generator=generator, dtype=tensor.dtype) * amplitude
                value[key] = tensor + noise
                changed += int(tensor.numel())
                total += int(tensor.numel())
        if not changed and isinstance(payload.get("state_dict"), dict):
            value = payload["state_dict"]
            for key, tensor in list(value.items()):
                if isinstance(tensor, torch.Tensor) and tensor.is_floating_point() and tensor.numel():
                    std = float(tensor.float().std().item()) if tensor.numel() > 1 else float(abs(tensor.float().item()))
                    value[key] = tensor + torch.randn(tensor.shape, generator=generator, dtype=tensor.dtype) * (float(scale) * max(std, 1e-4))
                    changed += int(tensor.numel())
                    total += int(tensor.numel())
        if not changed:
            return {"applied": False, "reason": "no_float_tensors", "path": path.name}
        tmp = path.with_suffix(path.suffix + ".tmp")
        torch.save(payload, tmp)
        tmp.replace(path)
        return {"applied": True, "path": path.name, "tensors_changed": changed, "tensors_total": total}

    def _mutate_runtime(self, config_path: Path, seed: int, *, agent_config) -> dict:
        payload = json.loads(config_path.read_text(encoding="utf-8"))
        runtime = payload.setdefault("runtime", {})
        fields = [str(item) for item in getattr(agent_config.runtime, "evolution_runtime_mutations", []) if str(item)]
        if not fields:
            return {"applied": False, "reason": "no_runtime_mutation_fields"}
        index = int(seed) % len(fields)
        field_name = fields[index]
        old = runtime.get(field_name)
        if not isinstance(old, (int, float)) or isinstance(old, bool):
            return {"applied": False, "reason": "runtime_field_not_numeric", "field": field_name}
        scales = [float(x) for x in getattr(agent_config.runtime, "evolution_runtime_scales", [0.85, 1.15])]
        if not scales:
            return {"applied": False, "reason": "no_runtime_scales", "field": field_name}
        scale = scales[(int(seed) // max(1, len(fields))) % len(scales)]
        new = float(old) * scale
        runtime[field_name] = new
        config_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        return {"applied": True, "field": field_name, "before": float(old), "after": new, "scale": scale}

    def _apply_mutation(self, candidate_root: Path, mutation: str, seed: int, agent_config) -> dict:
        scale = float(getattr(agent_config.runtime, "evolution_mutation_scale", 0.005))
        mutation = str(mutation)
        if mutation == "baseline":
            return {"mutation": mutation, "applied": True}
        if mutation == "model_noise":
            return {"mutation": mutation, **self._mutate_checkpoint(candidate_root / "data" / "aletheia.pt", seed, scale)}
        if mutation == "vision_noise":
            return {"mutation": mutation, **self._mutate_checkpoint(candidate_root / "data" / "perception.pt", seed, scale, scopes=("vision", "object_discovery"))}
        if mutation == "audio_noise":
            return {"mutation": mutation, **self._mutate_checkpoint(candidate_root / "data" / "perception.pt", seed, scale, scopes=("audio",))}
        if mutation == "planner_noise":
            return {"mutation": mutation, **self._mutate_checkpoint(candidate_root / "data" / "planning.pt", seed, scale)}
        if mutation == "speech_noise":
            return {"mutation": mutation, **self._mutate_checkpoint(candidate_root / "data" / "speech_ctc.pt", seed, scale)}
        if mutation == "tts_noise":
            return {"mutation": mutation, **self._mutate_checkpoint(candidate_root / "data" / "voice_tts.pt", seed, scale)}
        if mutation == "runtime_mutation":
            return {"mutation": mutation, **self._mutate_runtime(candidate_root / "config.json", seed, agent_config=agent_config)}
        return {"mutation": mutation, "applied": False, "reason": "unknown_mutation"}

    def _prune_generation_candidates(self, generation_root: Path, candidates: list[dict], keep: int) -> list[str]:
        keep = max(1, int(keep))
        ranked = sorted(
            candidates,
            key=lambda item: (
                bool(item.get("benchmark", {}).get("ok")),
                not bool(item.get("score", {}).get("hard_regression", True)),
                float(item.get("score", {}).get("score", float("-inf"))),
            ),
            reverse=True,
        )
        retained: list[str] = []
        for item in ranked[:keep]:
            path = Path(str(item.get("path", ""))).resolve()
            if path.exists() and generation_root.resolve() in path.parents:
                retained.append(str(path))
        for item in ranked[keep:]:
            path = Path(str(item.get("path", ""))).resolve()
            if path.exists() and generation_root.resolve() in path.parents:
                shutil.rmtree(path, ignore_errors=True)
        return retained

    @staticmethod
    def _benchmark_script() -> str:
        return r'''
import json, math, random
from pathlib import Path
import torch
from aletheia.agent import AletheiaAgent
from aletheia.config import load_config

root = Path('.')
cfg = load_config(root / 'config.json')
seed = int(__import__('os').environ.get('ALETHEIA_BENCHMARK_SEED', getattr(cfg.runtime, 'evolution_generation_seed', 1337)))
random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)
agent = AletheiaAgent(root, cfg)
metrics = {}
try:
    manifest = {}
    manifest_path = root / 'data' / 'evolution_manifest.json'
    if manifest_path.exists():
        try: manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
        except Exception: manifest = {}
    train_steps = max(0, int(getattr(cfg.runtime, 'evolution_candidate_train_steps', 0)))
    if train_steps and manifest.get('mutation') != 'baseline' and agent.memory.lesson_count() > 0:
        agent._train_recent_trajectory(train_steps)
    fixed = agent.evaluate_fixed_reference()
    if fixed.get('available') and math.isfinite(float(fixed.get('model_bpb', float('nan')))):
        metrics['language_bpb'] = float(fixed['model_bpb'])
    repeats = max(1, int(getattr(cfg.runtime, 'evolution_benchmark_repeats', 3)))
    max_items = max(2, int(getattr(cfg.runtime, 'evolution_benchmark_max_items', 64)))
    holdout_fraction = min(0.5, max(0.05, float(getattr(cfg.runtime, 'evolution_benchmark_holdout_fraction', 0.20))))
    is_candidate = manifest.get('mutation') not in {None, 'baseline'}
    def split_pool(pool):
        items = list(pool)
        if len(items) < 4:
            return items, items
        cut = int(round(len(items) * (1.0 - holdout_fraction)))
        cut = min(max(2, cut), len(items) - 1)
        return items[:cut], items[cut:]
    def ssl_loss_from(pool, augmenter, encoder):
        if len(pool) < 2: return None
        items = pool[:max_items]
        batch = torch.stack([x.float() / 255.0 if x.dtype == torch.uint8 else x.float() for x in items], 0).to(agent.device)
        values=[]
        encoder.eval()
        for rep in range(repeats):
            rep_seed = seed + rep * 1009
            random.seed(rep_seed); torch.manual_seed(rep_seed)
            a = augmenter(batch) if encoder is agent.perception.vision else torch.stack([augmenter(x) for x in batch], 0)
            torch.manual_seed(rep_seed + 7919)
            b = augmenter(batch) if encoder is agent.perception.vision else torch.stack([augmenter(x) for x in batch], 0)
            with torch.no_grad():
                z1 = encoder.encode_raw(a); z2 = encoder.encode_raw(b)
                values.append(float(agent.perception._ssl_loss(z1, z2).cpu()))
        return float(sum(values)/len(values))
    vision_train, vision_eval = split_pool(list(agent.perception._vision_buffer))
    audio_train, audio_eval = split_pool(list(agent.perception._audio_buffer))
    if is_candidate and train_steps:
        if len(vision_train) >= int(getattr(agent.perception, 'vision_batch_size', 1)):
            agent.perception.train_visual_observations(vision_train, steps=train_steps)
        if len(audio_train) >= int(getattr(agent.perception, 'audio_batch_size', 1)):
            agent.perception.train_audio_observations(audio_train, steps=train_steps)
    v = ssl_loss_from(vision_eval, agent.perception._vision_augment, agent.perception.vision)
    if v is not None and math.isfinite(v): metrics['vision_ssl_loss']=v
    a = ssl_loss_from(audio_eval, agent.perception._augment_audio, agent.perception.audio)
    if a is not None and math.isfinite(a): metrics['audio_ssl_loss']=a
    speech_manifest = getattr(agent.speech, 'last_manifest', '')
    if speech_manifest:
        try: cer = agent.speech.evaluate_manifest(speech_manifest, max_examples=int(getattr(cfg.runtime,'speech_eval_max_examples',256)))
        except Exception: cer = None
        if cer is not None and math.isfinite(float(cer)): metrics['speech_cer']=float(cer)
    tts_manifest = getattr(agent.tts, 'last_manifest', '')
    if tts_manifest:
        try: loss = agent.tts.evaluate_manifest(tts_manifest, max_examples=int(getattr(cfg.runtime,'tts_eval_max_examples',64)))
        except Exception: loss = None
        if loss is not None and math.isfinite(float(loss)): metrics['tts_holdout_loss']=float(loss)

finally:
    try: agent.close()
    except Exception: pass
print(json.dumps(metrics, sort_keys=True))
'''

    def benchmark_candidate(self, candidate_root: Path, timeout: int = 180, seed_override: int | None = None, *, require_hard_sandbox: bool = True) -> dict:
        try:
            if require_hard_sandbox and self._hard_sandbox_prefix(candidate_root) is None:
                return {"ok": False, "error": "hard_candidate_sandbox_unavailable", "metrics": {}, "sandbox": "unavailable"}
            env_extra = {"PYTHONPATH": str(candidate_root)}
            if seed_override is not None:
                env_extra["ALETHEIA_BENCHMARK_SEED"] = str(int(seed_override))
            result = self._run_python(
                candidate_root, ["-c", self._benchmark_script()],
                timeout=max(5, int(timeout)), hard_sandbox=bool(require_hard_sandbox), env_extra=env_extra,
            )
        except subprocess.TimeoutExpired as exc:
            return {"ok": False, "error": f"benchmark timeout: {exc}", "metrics": {}}
        except Exception as exc:
            return {"ok": False, "error": repr(exc), "metrics": {}}
        output = (result.stdout + "\n" + result.stderr).strip()
        if result.returncode != 0:
            return {"ok": False, "error": output[-10000:], "metrics": {}, "sandbox": "hard" if require_hard_sandbox else "process"}
        payload = None
        for line in reversed(result.stdout.splitlines()):
            try:
                candidate = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(candidate, dict):
                payload = candidate
                break
        if payload is None:
            return {"ok": False, "error": output[-10000:], "metrics": {}, "sandbox": "hard" if require_hard_sandbox else "process"}
        return {"ok": True, "metrics": payload, "output": output[-10000:], "sandbox": "hard" if require_hard_sandbox else "process"}

    @staticmethod
    def _ratio_for_metric(baseline: float, candidate: float, *, lower_is_better: bool) -> float:
        eps = 1e-8
        if lower_is_better:
            return (float(baseline) + eps) / (float(candidate) + eps)
        return (float(candidate) + eps) / (float(baseline) + eps)

    def score_candidate(self, baseline: dict, candidate: dict, runtime) -> dict:
        weights_cfg = getattr(runtime, "evolution_benchmark_weights", {}) or {}
        weights = {str(k): float(v) for k, v in dict(weights_cfg).items() if float(v) > 0}
        lower_is_better = {
            "language_bpb": True,
            "vision_ssl_loss": True,
            "audio_ssl_loss": True,
            "speech_cer": True,
            "tts_holdout_loss": True,
        }
        ratios = {}
        used_weights = {}
        # Only metrics with an explicitly defined direction are eligible. This avoids
        # accidentally treating an unknown metric as an implicit reward rule.
        for metric, weight in weights.items():
            if metric not in lower_is_better or metric not in baseline or metric not in candidate:
                continue
            b = float(baseline[metric])
            c = float(candidate[metric])
            if not (math.isfinite(b) and math.isfinite(c)):
                continue
            ratios[metric] = self._ratio_for_metric(b, c, lower_is_better=lower_is_better[metric])
            used_weights[metric] = float(weight)
        if not ratios:
            return {"score": 1.0, "ratios": {}, "hard_regression": False, "reason": "no_shared_benchmark_metrics"}
        total = sum(used_weights.values())
        log_score = sum((used_weights[k] / total) * math.log(max(1e-6, ratios[k])) for k in ratios)
        score = float(math.exp(log_score))
        max_reg = float(getattr(runtime, "evolution_max_metric_regression", 0.05))
        hard = any(ratio < (1.0 - max_reg) for ratio in ratios.values())
        return {
            "score": score,
            "ratios": ratios,
            "weights": used_weights,
            "hard_regression": hard,
            "shared_metrics": sorted(ratios),
        }

    def _candidate_specifications(self, agent) -> list[str]:
        configured = [str(x) for x in getattr(agent.config.runtime, "evolution_candidate_mutations", []) if str(x)]
        if configured:
            return configured
        return ["baseline", "model_noise", "vision_noise", "audio_noise", "planner_noise", "tts_noise"]

    def _write_manifest(self, candidate_root: Path, payload: dict) -> None:
        manifest = candidate_root / "data" / "evolution_manifest.json"
        manifest.parent.mkdir(parents=True, exist_ok=True)
        manifest.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    def run_generation(self, agent, *, population_size: int | None = None, auto_promote: bool | None = None) -> dict:
        """Evaluate every candidate on the same predeclared set of random seeds.

        Candidate selection is fail-closed: a benchmark adapter that cannot accept a
        seed override, an incomplete seed run, or any material per-seed regression
        cannot result in promotion. A mean score alone is not enough.
        """
        agent.save()
        generation_id = self._next_generation_id()
        population = int(population_size or getattr(agent.config.runtime, "evolution_population_size", 6))
        population = max(2, min(32, population))
        auto = bool(getattr(agent.config.runtime, "evolution_auto_promote", False)) if auto_promote is None else bool(auto_promote)
        seed_base = int(getattr(agent.config.runtime, "evolution_generation_seed", 1337)) + generation_id * 100003
        generation_root = self.generations_root / f"generation_{generation_id:05d}"
        generation_root.mkdir(parents=True, exist_ok=True)
        require_hard_sandbox = bool(getattr(agent.config.runtime, "evolution_require_hard_sandbox", True))
        benchmark_timeout = int(getattr(agent.config.runtime, "evolution_test_timeout", 300))
        benchmark_params = inspect.signature(self.benchmark_candidate).parameters
        if "seed_override" not in benchmark_params:
            result = {
                "generation": generation_id,
                "status": "aborted",
                "reason": "benchmark_adapter_missing_seed_override",
                "details": "FINAL28 requires paired, seeded benchmark runs; this adapter cannot prove that seeds were applied.",
                "promoted": False,
                "restart_required": False,
            }
            self.lab._record({"event": "generation_aborted", **result})
            return result

        raw_seeds = getattr(agent.config.runtime, "evolution_eval_seeds", [1337, 2027, 3037])
        try:
            eval_seeds = list(dict.fromkeys(int(seed) for seed in raw_seeds))
        except (TypeError, ValueError):
            eval_seeds = []
        if len(eval_seeds) < 3:
            result = {
                "generation": generation_id,
                "status": "aborted",
                "reason": "insufficient_unique_evaluation_seeds",
                "configured_seeds": eval_seeds,
                "minimum_required": 3,
                "promoted": False,
                "restart_required": False,
            }
            self.lab._record({"event": "generation_aborted", **result})
            return result

        def run_benchmark(candidate_root: Path, *, seed_override: int) -> dict:
            kwargs = {"timeout": benchmark_timeout, "seed_override": int(seed_override)}
            if "require_hard_sandbox" in benchmark_params:
                kwargs["require_hard_sandbox"] = require_hard_sandbox
            try:
                return self.benchmark_candidate(candidate_root, **kwargs)
            except Exception as exc:
                return {"ok": False, "error": f"benchmark_exception: {type(exc).__name__}: {exc}", "metrics": {}}

        def run_seeded_benchmark(candidate_root: Path) -> dict:
            runs: list[dict] = []
            for seed in eval_seeds:
                report = run_benchmark(candidate_root, seed_override=seed)
                if not report.get("ok") or not isinstance(report.get("metrics"), dict):
                    return {
                        "ok": False,
                        "error": f"seed_{seed}_failed: {report.get('error', 'benchmark returned invalid metrics')}",
                        "seeds": list(eval_seeds),
                        "completed_seeds": [row["seed"] for row in runs],
                        "per_seed": runs,
                        "metrics": {},
                    }
                clean_metrics = {}
                for key, value in report["metrics"].items():
                    try:
                        number = float(value)
                    except (TypeError, ValueError):
                        continue
                    if math.isfinite(number):
                        clean_metrics[str(key)] = number
                if not clean_metrics:
                    return {"ok": False, "error": f"seed_{seed}_returned_no_finite_metrics", "seeds": list(eval_seeds), "completed_seeds": [row["seed"] for row in runs], "per_seed": runs, "metrics": {}}
                runs.append({"seed": seed, "metrics": clean_metrics})
            shared_metrics = set(runs[0]["metrics"])
            for row in runs[1:]:
                shared_metrics.intersection_update(row["metrics"])
            aggregate = {
                metric: sum(row["metrics"][metric] for row in runs) / len(runs)
                for metric in sorted(shared_metrics)
            }
            if not aggregate:
                return {"ok": False, "error": "no_metrics_shared_across_all_seeds", "seeds": list(eval_seeds), "per_seed": runs, "metrics": {}}
            return {
                "ok": True,
                "seeds": list(eval_seeds),
                "seed_count": len(runs),
                "aggregation": "arithmetic_mean_per_metric_over_paired_seed_runs",
                "metrics": aggregate,
                "per_seed": runs,
            }

        baseline = run_seeded_benchmark(self.root)
        if not baseline.get("ok"):
            result = {"generation": generation_id, "status": "aborted", "reason": "baseline_seeded_benchmark_failed", "baseline": baseline, "eval_seeds": eval_seeds, "promoted": False, "restart_required": False}
            self.lab._record({"event": "generation_aborted", **result})
            return result
        baseline_by_seed = {row["seed"]: row["metrics"] for row in baseline["per_seed"]}

        specs = self._candidate_specifications(agent)
        if len(specs) < population:
            for idx in range(len(specs), population):
                specs.append(specs[1 + ((idx - 1) % max(1, len(specs) - 1))] if len(specs) > 1 else "model_noise")
        specs = specs[:population]
        candidates: list[dict] = []
        min_improvement = max(0.0, float(getattr(agent.config.runtime, "evolution_min_improvement", 0.01)))
        required_positive_seeds = (2 * len(eval_seeds) + 2) // 3

        for index, mutation in enumerate(specs):
            candidate_id = f"g{generation_id:05d}c{index:02d}_{self._slug(mutation)}"
            candidate_root = generation_root / candidate_id
            migrated = self._copy_tree_with_learned_state(candidate_root)
            mutation_seed = seed_base + index
            mutation_result = self._apply_mutation(candidate_root, mutation, mutation_seed, agent.config)
            manifest = {
                "generation": generation_id,
                "candidate_id": candidate_id,
                "parent": self.current_generation().get("candidate_id", "root"),
                "mutation_seed": mutation_seed,
                "evaluation_seeds": list(eval_seeds),
                "mutation": mutation,
                "mutation_result": mutation_result,
                "migrated_files": migrated,
                "created": time.time(),
                "evaluation_policy": "paired identical seeds; aggregate means plus per-seed non-regression gate",
            }
            self._write_manifest(candidate_root, manifest)
            benchmark = run_seeded_benchmark(candidate_root)
            if benchmark.get("ok"):
                score = self.score_candidate(baseline["metrics"], benchmark["metrics"], agent.config.runtime)
                per_seed_scores = []
                for row in benchmark["per_seed"]:
                    seed = row["seed"]
                    paired = self.score_candidate(baseline_by_seed[seed], row["metrics"], agent.config.runtime)
                    per_seed_scores.append({"seed": seed, **paired})
                positive_seed_count = sum(1 for row in per_seed_scores if float(row.get("score", 1.0)) > 1.0)
                per_seed_regression = any(bool(row.get("hard_regression", True)) or not row.get("shared_metrics") for row in per_seed_scores)
                enough_seed_wins = positive_seed_count >= required_positive_seeds
                if not score.get("shared_metrics"):
                    per_seed_regression = True
                score["aggregate_score"] = float(score.get("score", 1.0))
                score["paired_seed_scores"] = per_seed_scores
                score["positive_seed_count"] = positive_seed_count
                score["required_positive_seeds"] = required_positive_seeds
                score["per_seed_non_regression"] = not per_seed_regression
                score["enough_positive_seeds"] = enough_seed_wins
                score["hard_regression"] = bool(score.get("hard_regression", True) or per_seed_regression or not enough_seed_wins)
                if score["hard_regression"]:
                    score["reason"] = "paired_seed_non_regression_or_consistency_gate_failed"
            else:
                score = {"score": 0.0, "ratios": {}, "hard_regression": True, "reason": benchmark.get("error", "seeded_benchmark_failed"), "shared_metrics": []}
            result_item = {
                "candidate_id": candidate_id,
                "mutation": mutation,
                "mutation_seed": mutation_seed,
                "path": str(candidate_root),
                "mutation_result": mutation_result,
                "benchmark": benchmark,
                "score": score,
                "manifest": manifest,
            }
            candidates.append(result_item)
            self.lab._record({"event": "generation_candidate", "generation": generation_id, "candidate_id": candidate_id, "mutation": mutation, "mutation_seed": mutation_seed, "evaluation_seeds": eval_seeds, "score": score})

        promotable = [
            item for item in candidates
            if item["benchmark"].get("ok")
            and not item["score"].get("hard_regression", True)
            and float(item["score"].get("score", 0.0)) >= 1.0 + min_improvement
        ]
        winner = max(promotable, key=lambda item: float(item["score"]["score"])) if promotable else None
        result = {
            "generation": generation_id,
            "status": "completed",
            "eval_seeds": eval_seeds,
            "baseline": baseline,
            "population": candidates,
            "winner": winner,
            "confirmation": {
                "passed": winner is not None,
                "method": "paired_multi_seed_non_regression",
                "seeds": list(eval_seeds),
                "minimum_positive_seeds": required_positive_seeds,
            },
            "promoted": False,
            "restart_required": False,
        }
        if winner is not None and auto:
            promotion = self.promote_candidate(
                winner["path"],
                generation=generation_id,
                candidate_id=winner["candidate_id"],
                score=winner["score"],
                keep_generations=int(getattr(agent.config.runtime, "evolution_keep_generations", 3)),
            )
            result.update({"promoted": True, **promotion})
        retained = self._prune_generation_candidates(
            generation_root,
            candidates,
            keep=max(1, int(getattr(agent.config.runtime, "evolution_keep_candidates", 2))),
        )
        result["retained_candidates"] = retained
        self.lab._record({"event": "generation_completed", "generation": generation_id, "promoted": result["promoted"], "restart_required": result["restart_required"], "winner": winner["candidate_id"] if winner else None, "retained_candidates": retained, "eval_seeds": eval_seeds})
        return result

    @staticmethod
    def _promotion_excluded(name: str) -> bool:
        return str(name) in {"evolution", "browser_profile", "evolution_manifest.json"}

    @classmethod
    def apply_pending_promotion(cls, root: Path) -> dict:
        """Apply a staged learned generation before runtime state is opened."""
        root = Path(root).resolve()
        evolution_root = root / "data" / "evolution"
        pending = evolution_root / "pending_promotion.json"
        transaction_root = evolution_root / "promotion_transaction"
        journal = transaction_root / "journal.json"
        if not pending.exists():
            if journal.exists():
                try:
                    payload = json.loads(journal.read_text(encoding="utf-8"))
                    if payload.get("status") != "complete":
                        cls._rollback_promotion(root, payload)
                except Exception:
                    pass
                return {"applied": False, "status": "recovered_previous_transaction"}
            return {"applied": False, "status": "clean"}
        try:
            # If a previous process died during the swap, recover that transaction
            # before touching the pending successor. This preserves atomicity across
            # crashes, not just ordinary exceptions in the current process.
            if journal.exists():
                try:
                    prior = json.loads(journal.read_text(encoding="utf-8"))
                except Exception:
                    prior = {}
                if prior.get("status") != "complete":
                    cls._rollback_promotion(root, prior)
                    transaction_root.mkdir(parents=True, exist_ok=True)
                    journal = transaction_root / "journal.json"
            payload = json.loads(pending.read_text(encoding="utf-8"))
            candidate = Path(payload["candidate_path"]).resolve()
            if evolution_root not in candidate.parents:
                raise ValueError("pending candidate is outside the managed evolution tree")
            candidate_data = candidate / "data"
            if not candidate_data.exists():
                raise FileNotFoundError(candidate_data)
            transaction_root.mkdir(parents=True, exist_ok=True)
            stage = transaction_root / "stage"
            old = transaction_root / "old"
            shutil.rmtree(stage, ignore_errors=True)
            shutil.rmtree(old, ignore_errors=True)
            stage.mkdir(parents=True, exist_ok=True)
            old.mkdir(parents=True, exist_ok=True)
            staged_names: list[str] = []
            for child in candidate_data.iterdir():
                if cls._promotion_excluded(child.name):
                    continue
                target = stage / child.name
                if child.is_dir():
                    shutil.copytree(child, target)
                else:
                    shutil.copy2(child, target)
                staged_names.append(child.name)
            candidate_config = candidate / "config.json"
            staged_config = transaction_root / "config.json.stage"
            if candidate_config.exists():
                shutil.copy2(candidate_config, staged_config)
            active_data = root / "data"
            journal_payload = {
                "schema_version": 2, "status": "applying",
                "generation": int(payload.get("generation", 0)),
                "candidate_id": str(payload.get("candidate_id", "")),
                "candidate_path": str(candidate),
                "active_data": str(active_data), "old_dir": str(old),
                "stage_dir": str(stage), "staged_names": staged_names,
                "old_names": [], "activated_names": [], "staged_config": str(staged_config),
                "created": time.time(),
            }
            journal.write_text(json.dumps(journal_payload, indent=2), encoding="utf-8")
            moved_old: list[str] = []
            for child in active_data.iterdir():
                if cls._promotion_excluded(child.name):
                    continue
                os.replace(child, old / child.name)
                moved_old.append(child.name)
                journal_payload["old_names"] = list(moved_old)
                journal.write_text(json.dumps(journal_payload, indent=2), encoding="utf-8")
            activated: list[str] = []
            for child in stage.iterdir():
                os.replace(child, active_data / child.name)
                activated.append(child.name)
                journal_payload["activated_names"] = list(activated)
                journal.write_text(json.dumps(journal_payload, indent=2), encoding="utf-8")
            if staged_config.exists():
                active_config = root / "config.json"
                if active_config.exists():
                    os.replace(active_config, old / "config.json.previous")
                os.replace(staged_config, active_config)
            current_path = evolution_root / "current_generation.json"
            parent = str(payload.get("parent", "root"))
            promoted = {
                "generation": int(payload.get("generation", 0)),
                "candidate_id": str(payload.get("candidate_id", "")),
                "parent": parent, "score": payload.get("score", {}),
                "activated": time.time(), "restart_required": False,
                "migrated_files": sorted(staged_names), "status": "active",
            }
            tmp_current = current_path.with_suffix(".tmp")
            tmp_current.write_text(json.dumps(promoted, indent=2, ensure_ascii=False), encoding="utf-8")
            os.replace(tmp_current, current_path)
            journal_payload["status"] = "complete"
            journal.write_text(json.dumps(journal_payload, indent=2), encoding="utf-8")
            pending.unlink(missing_ok=True)
            keep = max(1, int(payload.get("keep_generations", 3)))
            GenerationalEvolution._prune_after_startup_static(evolution_root, keep)
            shutil.rmtree(transaction_root, ignore_errors=True)
            return {"applied": True, "status": "activated", "promotion": promoted}
        except Exception as exc:
            try:
                if journal.exists():
                    cls._rollback_promotion(root, json.loads(journal.read_text(encoding="utf-8")))
            except Exception as rollback_exc:
                return {"applied": False, "status": "rollback_failed", "error": repr(exc), "rollback_error": repr(rollback_exc)}
            return {"applied": False, "status": "apply_failed_rolled_back", "error": repr(exc)}

    @staticmethod
    def _rollback_promotion(root: Path, journal: dict) -> None:
        root = Path(root).resolve()
        active = Path(journal.get("active_data", root / "data"))
        old = Path(journal.get("old_dir", root / "data" / "evolution" / "promotion_transaction" / "old"))
        stage = Path(journal.get("stage_dir", root / "data" / "evolution" / "promotion_transaction" / "stage"))
        # Only remove files that were actually activated from the candidate stage.
        # `staged_names` describes what was prepared, not what reached the active tree.
        # Using it here can delete an untouched parent file after an early swap failure.
        for name in journal.get("activated_names", []):
            destination = active / str(name)
            if destination.exists():
                if destination.is_dir() and not destination.is_symlink():
                    shutil.rmtree(destination, ignore_errors=True)
                else:
                    destination.unlink(missing_ok=True)
        for name in journal.get("old_names", []):
            backup = old / str(name)
            destination = active / str(name)
            if backup.exists() and not destination.exists():
                shutil.move(str(backup), str(destination))
        previous = old / "config.json.previous"
        if previous.exists():
            active_config = root / "config.json"
            if active_config.exists():
                active_config.unlink(missing_ok=True)
            shutil.move(str(previous), str(active_config))
        transaction_root = old.parent
        shutil.rmtree(stage, ignore_errors=True)
        shutil.rmtree(old, ignore_errors=True)
        shutil.rmtree(transaction_root, ignore_errors=True)

    def promote_candidate(self, candidate_path: str | Path, *, generation: int, candidate_id: str, score: dict, keep_generations: int = 3) -> dict:
        candidate_root = Path(candidate_path).resolve()
        if self.generations_root not in candidate_root.parents:
            raise ValueError("candidate must live in the managed generation tree")
        pending = self.evolution_root / "pending_promotion.json"
        if pending.exists():
            raise RuntimeError("a generation promotion is already pending restart")
        parent = self.current_generation().get("candidate_id", "root")
        pending_payload = {
            "schema_version": 2, "generation": int(generation),
            "candidate_id": str(candidate_id), "parent": str(parent),
            "candidate_path": str(candidate_root), "score": score,
            "created": time.time(), "restart_required": True,
            "keep_generations": max(1, int(keep_generations)),
            "status": "pending_restart",
        }
        pending.write_text(json.dumps(pending_payload, indent=2, ensure_ascii=False), encoding="utf-8")
        self.lab._record({"event": "generation_promotion_staged", **pending_payload})
        return {"promotion": pending_payload, "restart_required": True, "backup": "created_transactionally_at_next_startup", "pending_promotion": str(pending)}

    @staticmethod
    def _prune_after_startup_static(evolution_root: Path, keep: int) -> None:
        generations_root = Path(evolution_root) / "generations"
        dirs = []
        for path in generations_root.glob("generation_*"):
            if path.is_dir():
                try: dirs.append((int(path.name.split("_", 1)[1]), path))
                except (IndexError, ValueError): pass
        dirs.sort(reverse=True)
        for _, path in dirs[max(1, int(keep)):]:
            shutil.rmtree(path, ignore_errors=True)

    def _prune_generations(self, keep: int) -> None:
        dirs = []
        for path in self.generations_root.glob("generation_*"):
            if path.is_dir():
                try:
                    dirs.append((int(path.name.split("_", 1)[1]), path))
                except (IndexError, ValueError):
                    continue
        dirs.sort(reverse=True)
        for _, path in dirs[max(1, int(keep)):]:
            shutil.rmtree(path, ignore_errors=True)

    def lineage(self) -> list[dict]:
        rows: list[dict] = []
        for path in sorted(self.generations_root.glob("generation_*/g*_*/data/evolution_manifest.json")):
            try:
                rows.append(json.loads(path.read_text(encoding="utf-8")))
            except Exception:
                continue
        rows.sort(key=lambda item: (int(item.get("generation", 0)), str(item.get("candidate_id", ""))))
        return rows
