"""Durable, source-separated upload-to-learning pipeline for Aletheia.

The service deliberately separates extraction, splitting, indexing, training, and
acceptance. It never fits the tokenizer on the job holdout, and the holdout text is
kept in the jobs database rather than written into a normal curriculum directory.
All model updates still use Aletheia's own randomly initialized model and its normal
lesson acceptance checks; a second source-level holdout gate is applied afterwards.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import re
import sqlite3
import sys
import tempfile
import time
import uuid
from pathlib import Path
from typing import Any

from .curriculum import iter_text_files, read_text_file

SUPPORTED_SUFFIXES = {
    ".txt", ".md", ".markdown", ".pdf", ".docx", ".epub",
    ".html", ".htm", ".xhtml", ".parquet",
}
DEFAULT_MAX_UPLOAD_BYTES = 50 * 1024 * 1024
DEFAULT_MAX_EXTRACTED_CHARS = 5_000_000
DEFAULT_HOLDOUT_FRACTION = 0.20

_STATES = {
    "queued", "extracting", "splitting", "indexing", "training", "evaluating",
    "accepted", "rejected", "failed",
}
_ACTIVE_STATES = {"extracting", "splitting", "indexing", "training", "evaluating"}
_TERMINAL_STATES = {"accepted", "rejected", "failed"}


def _pid_is_alive(pid: int) -> bool:
    """Return whether a process ID appears alive on the current host."""
    pid = int(pid)
    if pid <= 0:
        return False
    if sys.platform == "win32":
        # Windows does not consistently accept POSIX signal 0. Inspect the process
        # handle instead, treating access/inspection failures conservatively as alive.
        try:
            import ctypes
            from ctypes import wintypes

            kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
            kernel32.OpenProcess.argtypes = (wintypes.DWORD, wintypes.BOOL, wintypes.DWORD)
            kernel32.OpenProcess.restype = wintypes.HANDLE
            kernel32.GetExitCodeProcess.argtypes = (wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD))
            kernel32.GetExitCodeProcess.restype = wintypes.BOOL
            kernel32.CloseHandle.argtypes = (wintypes.HANDLE,)
            kernel32.CloseHandle.restype = wintypes.BOOL
            handle = kernel32.OpenProcess(0x1000, False, pid)  # PROCESS_QUERY_LIMITED_INFORMATION
            if not handle:
                return ctypes.get_last_error() == 5  # ERROR_ACCESS_DENIED: assume alive
            try:
                code = wintypes.DWORD()
                if not kernel32.GetExitCodeProcess(handle, ctypes.byref(code)):
                    return True
                return int(code.value) == 259  # STILL_ACTIVE
            finally:
                kernel32.CloseHandle(handle)
        except Exception:
            return True
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        # Conservatively treat platform-specific permission/inspection errors as alive.
        return True
    return True


def _sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _normalized_text(text: str) -> str:
    return re.sub(r"\s+", " ", str(text).replace("\x00", " ")).strip().casefold()


def _text_fingerprint(text: str) -> str:
    return _sha256(_normalized_text(text).encode("utf-8", errors="replace"))


def _shingles(text: str, width: int = 5, max_samples: int = 20_000) -> set[str]:
    words = re.findall(r"[^\W_]+", _normalized_text(text), flags=re.UNICODE)
    if len(words) < width:
        return {" ".join(words)} if words else set()
    step = max(1, math.ceil((len(words) - width + 1) / max(1, max_samples)))
    return {
        _sha256(" ".join(words[index:index + width]).encode("utf-8"))[:24]
        for index in range(0, len(words) - width + 1, step)
    }


def _validate_separated_texts(train_text: str, evaluation_text: str) -> None:
    """Reject obvious train/holdout leakage, including after deterministic size caps."""
    train_normalized = _normalized_text(train_text)
    eval_normalized = _normalized_text(evaluation_text)
    if not train_normalized or not eval_normalized:
        raise ValueError("split produced empty training or evaluation data")
    if train_normalized == eval_normalized or train_normalized in eval_normalized or eval_normalized in train_normalized:
        raise ValueError("split appears to contain overlapping text; refusing to train")
    train_shingles = _shingles(train_text)
    eval_shingles = _shingles(evaluation_text)
    if len(eval_shingles) >= 16 and len(eval_shingles & train_shingles) / max(1, len(eval_shingles)) >= 0.85:
        raise ValueError("evaluation units substantially overlap training units; refusing to train")


def _split_units(units: list[str], *, fraction: float, seed: str, strategy: str) -> dict[str, Any]:
    # Exact duplicate paragraphs/sentences must not straddle train and holdout.
    units = [unit.strip() for unit in units if unit and unit.strip()]
    unique_units: list[str] = []
    seen_units: set[str] = set()
    for unit in units:
        canonical = _normalized_text(unit)
        if canonical and canonical not in seen_units:
            seen_units.add(canonical)
            unique_units.append(unit)
    units = unique_units
    if len(units) < 2:
        raise ValueError("source is too short to make a separate training and evaluation split")
    # Stable pseudo-random ordering based on content and source identity, not process RNG.
    ranked = sorted(
        range(len(units)),
        key=lambda idx: hashlib.sha256(f"{seed}:{idx}:{units[idx]}".encode("utf-8", errors="replace")).digest(),
    )
    holdout_count = max(1, min(len(units) - 1, int(round(len(units) * fraction))))
    holdout_indices = set(ranked[:holdout_count])
    train_units = [unit for idx, unit in enumerate(units) if idx not in holdout_indices]
    eval_units = [unit for idx, unit in enumerate(units) if idx in holdout_indices]
    # Keep source order inside each split. No unit is copied to both sides.
    train = "\n\n".join(train_units).strip()
    holdout = "\n\n".join(eval_units).strip()
    if len(train) < 32 or len(holdout) < 24:
        raise ValueError("not enough text for a meaningful separated split; provide a longer source")
    return {
        "train_text": train,
        "evaluation_text": holdout,
        "strategy": strategy,
        "unit_count": len(units),
        "train_units": len(train_units),
        "evaluation_units": len(eval_units),
        "train_chars": len(train),
        "evaluation_chars": len(holdout),
    }


def split_source_text(text: str, *, source_hash: str, holdout_fraction: float = DEFAULT_HOLDOUT_FRACTION) -> dict[str, Any]:
    """Split a document before any model chunking, without shared exact text units.

    Prefer paragraph groups, then sentence groups, then a whitespace-token split for
    unpunctuated text. The last fallback is explicitly identified as weaker because a
    contiguous text stream can have more lexical dependence across its boundary.
    """
    if not 0.05 <= float(holdout_fraction) <= 0.40:
        raise ValueError("holdout_fraction must be between 0.05 and 0.40")
    cleaned = re.sub(r"\x00", " ", str(text)).strip()
    if len(cleaned) < 80:
        raise ValueError("source contains fewer than 80 characters; provide more material for evaluation")
    paragraphs = [part.strip() for part in re.split(r"\n\s*\n", cleaned) if part.strip()]
    if len(paragraphs) >= 2:
        result = _split_units(paragraphs, fraction=holdout_fraction, seed=source_hash, strategy="paragraph_grouped")
    else:
        sentences = [part.strip() for part in re.split(r"(?<=[.!?。！？])\s+", cleaned) if part.strip()]
        if len(sentences) >= 2:
            result = _split_units(sentences, fraction=holdout_fraction, seed=source_hash, strategy="sentence_grouped")
        else:
            words = cleaned.split()
            if len(words) < 20:
                raise ValueError("source contains too few independent text units for a safe evaluation split")
            # This is a non-overlapping contiguous split: useful for basic coverage but
            # weaker than paragraph/sentence isolation, and reported as such.
            cut = max(1, min(len(words) - 1, int(len(words) * (1.0 - holdout_fraction))))
            train = " ".join(words[:cut]).strip()
            holdout = " ".join(words[cut:]).strip()
            if len(train) < 32 or len(holdout) < 24:
                raise ValueError("source does not have enough text for a useful evaluation holdout")
            result = {
                "train_text": train,
                "evaluation_text": holdout,
                "strategy": "contiguous_word_fallback_weaker_isolation",
                "unit_count": len(words),
                "train_units": cut,
                "evaluation_units": len(words) - cut,
                "train_chars": len(train),
                "evaluation_chars": len(holdout),
            }
    _validate_separated_texts(result["train_text"], result["evaluation_text"])
    result["train_sha256"] = _text_fingerprint(result["train_text"])
    result["evaluation_sha256"] = _text_fingerprint(result["evaluation_text"])
    return result


def _cap_text(text: str, maximum: int, *, seed: str) -> tuple[str, bool, int]:
    """Bound per-job compute while sampling whole units deterministically where possible."""
    value = str(text).strip()
    maximum = max(64, int(maximum))
    if len(value) <= maximum:
        units = [part for part in re.split(r"\n\s*\n", value) if part.strip()]
        if len(units) < 2:
            units = [part for part in re.split(r"(?<=[.!?。！？])\s+", value) if part.strip()]
        return value, False, max(1, len(units))
    units = [part.strip() for part in re.split(r"\n\s*\n", value) if part.strip()]
    if len(units) < 2:
        units = [part.strip() for part in re.split(r"(?<=[.!?。！？])\s+", value) if part.strip()]
    if len(units) >= 2:
        order = sorted(
            range(len(units)),
            key=lambda index: hashlib.sha256(f"cap:{seed}:{index}:{units[index]}".encode("utf-8", errors="replace")).digest(),
        )
        selected: list[int] = []
        used = 0
        for index in order:
            separator = 2 if selected else 0
            if used + separator + len(units[index]) <= maximum:
                selected.append(index)
                used += separator + len(units[index])
        if selected:
            output = "\n\n".join(units[index] for index in sorted(selected))
            return output, True, len(selected)
    # An individually oversized unit has no safe natural boundary; cap at a word
    # boundary and disclose that the job used a truncated view of the source.
    clipped = value[:maximum]
    boundary = clipped.rfind(" ")
    if boundary >= 64:
        clipped = clipped[:boundary]
    if len(clipped.strip()) < 64:
        raise ValueError("source cannot be safely bounded to the configured training/evaluation budget")
    return clipped.strip(), True, 1


class LearningPipeline:
    """SQLite-backed learning job queue rooted in one Aletheia project instance."""

    def __init__(
        self,
        root: Path,
        *,
        max_upload_bytes: int = DEFAULT_MAX_UPLOAD_BYTES,
        max_extracted_chars: int = DEFAULT_MAX_EXTRACTED_CHARS,
        max_training_chars: int = 250_000,
        max_evaluation_chars: int = 100_000,
        holdout_fraction: float = DEFAULT_HOLDOUT_FRACTION,
        near_duplicate_threshold: float = 0.96,
    ):
        self.root = Path(root).expanduser().resolve()
        self.data_dir = self.root / "data"
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.workspace = self.data_dir / ".learning_jobs"
        self.incoming_dir = self.workspace / "incoming"
        self.incoming_dir.mkdir(parents=True, exist_ok=True)
        # Keep generated reports inside the hidden job workspace so recursive curriculum
        # discovery does not mistake report JSON for user-supplied learning material.
        self.report_dir = self.workspace / "reports"
        self.report_dir.mkdir(parents=True, exist_ok=True)
        self.db_path = self.data_dir / "learning_jobs.sqlite3"
        self.max_upload_bytes = max(1, int(max_upload_bytes))
        self.max_extracted_chars = max(80, int(max_extracted_chars))
        self.max_training_chars = max(64, int(max_training_chars))
        self.max_evaluation_chars = max(64, int(max_evaluation_chars))
        self.holdout_fraction = float(holdout_fraction)
        self.near_duplicate_threshold = min(1.0, max(0.5, float(near_duplicate_threshold)))
        self.db = sqlite3.connect(self.db_path, timeout=30.0)
        self.db.row_factory = sqlite3.Row
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute("PRAGMA foreign_keys=ON")
        self.db.execute("PRAGMA busy_timeout=30000")
        self.db.execute(
            """CREATE TABLE IF NOT EXISTS learning_jobs (
                id TEXT PRIMARY KEY,
                created_at REAL NOT NULL,
                updated_at REAL NOT NULL,
                state TEXT NOT NULL,
                progress INTEGER NOT NULL DEFAULT 0,
                status_message TEXT NOT NULL DEFAULT '',
                source_name TEXT NOT NULL,
                source_path TEXT NOT NULL DEFAULT '',
                source_sha256 TEXT NOT NULL,
                source_bytes INTEGER NOT NULL,
                provenance_json TEXT NOT NULL DEFAULT '{}',
                extracted_sha256 TEXT NOT NULL DEFAULT '',
                train_sha256 TEXT NOT NULL DEFAULT '',
                evaluation_sha256 TEXT NOT NULL DEFAULT '',
                split_strategy TEXT NOT NULL DEFAULT '',
                train_chars INTEGER NOT NULL DEFAULT 0,
                evaluation_chars INTEGER NOT NULL DEFAULT 0,
                train_text TEXT NOT NULL DEFAULT '',
                evaluation_text TEXT NOT NULL DEFAULT '',
                metrics_json TEXT NOT NULL DEFAULT '{}',
                report_path TEXT NOT NULL DEFAULT '',
                error TEXT NOT NULL DEFAULT '',
                duplicate_of TEXT NOT NULL DEFAULT '',
                attempts INTEGER NOT NULL DEFAULT 0,
                worker_pid INTEGER NOT NULL DEFAULT 0
            )"""
        )
        # Forward-compatible migration for databases created by earlier FINAL33 builds.
        columns = {str(row[1]) for row in self.db.execute("PRAGMA table_info(learning_jobs)").fetchall()}
        if "worker_pid" not in columns:
            self.db.execute("ALTER TABLE learning_jobs ADD COLUMN worker_pid INTEGER NOT NULL DEFAULT 0")
        self.db.execute("CREATE INDEX IF NOT EXISTS idx_learning_jobs_state_created ON learning_jobs(state, created_at)")
        self.db.execute("CREATE INDEX IF NOT EXISTS idx_learning_jobs_source_hash ON learning_jobs(source_sha256)")
        self.db.execute("CREATE INDEX IF NOT EXISTS idx_learning_jobs_extracted_hash ON learning_jobs(extracted_sha256)")
        self.db.commit()
        self._recover_interrupted_jobs()

    def close(self) -> None:
        self.db.close()

    def __enter__(self) -> "LearningPipeline":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def _recover_interrupted_jobs(self) -> None:
        """Recover safe pre-training jobs; quarantine jobs interrupted during model updates."""
        now = time.time()
        rows = self.db.execute("SELECT id,state,source_path,train_text,evaluation_text,worker_pid FROM learning_jobs WHERE state IN ('extracting','splitting','indexing','training','evaluating')").fetchall()
        for row in rows:
            state = str(row["state"])
            job_id = str(row["id"])
            # Opening the queue must not steal a live process's in-progress job.
            if _pid_is_alive(int(row["worker_pid"] or 0)):
                continue
            if state in {"extracting", "splitting", "indexing"}:
                # If the durable split has been committed, a rerun can safely resume from it.
                message = "Recovered after interruption; safe to resume from the last durable stage."
                self.db.execute(
                    "UPDATE learning_jobs SET state='queued',progress=0,worker_pid=0,status_message=?,updated_at=? WHERE id=?",
                    (message, now, job_id),
                )
            else:
                # A process may have saved weights before writing the external evaluation
                # decision. Do not silently accept/retry this candidate.
                self.db.execute(
                    "UPDATE learning_jobs SET state='failed',progress=100,worker_pid=0,status_message='Interrupted during model update; review checkpoint and job metrics before retrying.',error='Process stopped during training/evaluation; final job-level acceptance was not recorded.',updated_at=? WHERE id=?",
                    (now, job_id),
                )
        self.db.commit()

    def _row(self, job_id: str) -> sqlite3.Row:
        row = self.db.execute("SELECT * FROM learning_jobs WHERE id=?", (str(job_id),)).fetchone()
        if row is None:
            raise KeyError(f"learning job not found: {job_id}")
        return row

    @staticmethod
    def _public(row: sqlite3.Row, *, include_text: bool = False) -> dict[str, Any]:
        item = dict(row)
        for field in ("provenance_json", "metrics_json"):
            try:
                item[field.removesuffix("_json")] = json.loads(item.pop(field) or "{}")
            except (ValueError, TypeError):
                item[field.removesuffix("_json")] = {}
        item.pop("worker_pid", None)
        if not include_text:
            item.pop("train_text", None)
            item.pop("evaluation_text", None)
        return item

    def get_job(self, job_id: str) -> dict[str, Any]:
        return self._public(self._row(job_id))

    def list_jobs(self, limit: int = 50, *, state: str | None = None) -> list[dict[str, Any]]:
        limit = max(1, min(500, int(limit)))
        if state is not None:
            if state not in _STATES:
                raise ValueError(f"unknown learning job state: {state}")
            rows = self.db.execute(
                "SELECT * FROM learning_jobs WHERE state=? ORDER BY created_at DESC LIMIT ?",
                (state, limit),
            ).fetchall()
        else:
            rows = self.db.execute("SELECT * FROM learning_jobs ORDER BY created_at DESC LIMIT ?", (limit,)).fetchall()
        return [self._public(row) for row in rows]

    def _find_duplicate(self, source_sha256: str) -> sqlite3.Row | None:
        return self.db.execute(
            "SELECT * FROM learning_jobs WHERE source_sha256=? AND state IN ('queued','extracting','splitting','indexing','training','evaluating','accepted','rejected') ORDER BY created_at DESC LIMIT 1",
            (source_sha256,),
        ).fetchone()

    def _store_payload(self, raw: bytes, *, filename: str, license_name: str, source_uri: str, title: str | None) -> dict[str, Any]:
        suffix = Path(filename).suffix.lower()
        if suffix not in SUPPORTED_SUFFIXES:
            raise ValueError(f"unsupported file type {suffix or '(no extension)'}; supported: {', '.join(sorted(SUPPORTED_SUFFIXES))}")
        if len(raw) > self.max_upload_bytes:
            raise ValueError(f"upload is {len(raw)} bytes; configured limit is {self.max_upload_bytes} bytes")
        if not raw:
            raise ValueError("empty upload")
        digest = _sha256(raw)
        timestamp = time.time()
        job_id = uuid.uuid4().hex
        target = self.incoming_dir / f"{job_id}{suffix}"
        # Stage to an adjacent temporary file and rename only after durable write.
        fd, tmp_name = tempfile.mkstemp(prefix=f"{job_id}.", suffix=".tmp", dir=self.incoming_dir)
        try:
            with os.fdopen(fd, "wb") as handle:
                handle.write(raw)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(tmp_name, target)
        finally:
            try:
                Path(tmp_name).unlink(missing_ok=True)
            except OSError:
                pass
        provenance = {
            "license": str(license_name or "unspecified; caller must confirm usage rights")[:500],
            "source_uri": str(source_uri or "")[:2000],
            "ingestion": "local_upload_or_paste",
            "title": str(title or Path(filename).stem)[:500],
        }
        try:
            self.db.execute("BEGIN IMMEDIATE")
            duplicate = self._find_duplicate(digest)
            if duplicate is not None:
                self.db.rollback()
                target.unlink(missing_ok=True)
                return {"duplicate": True, **self._public(duplicate)}
            self.db.execute(
                """INSERT INTO learning_jobs(
                    id,created_at,updated_at,state,progress,status_message,source_name,source_path,
                    source_sha256,source_bytes,provenance_json
                ) VALUES(?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    job_id, timestamp, timestamp, "queued", 0, "Waiting for an available worker.",
                    str(filename)[:500], str(target), digest, len(raw), json.dumps(provenance, ensure_ascii=False),
                ),
            )
            self.db.commit()
        except Exception:
            self.db.rollback()
            target.unlink(missing_ok=True)
            raise
        return {"duplicate": False, **self.get_job(job_id)}

    def enqueue_file(
        self,
        path: Path,
        *,
        license_name: str = "",
        source_uri: str = "",
        title: str | None = None,
    ) -> dict[str, Any]:
        source = Path(path).expanduser().resolve()
        if not source.is_file():
            raise FileNotFoundError(source)
        if source.suffix.lower() not in SUPPORTED_SUFFIXES:
            raise ValueError(f"unsupported file type {source.suffix or '(no extension)'}; supported: {', '.join(sorted(SUPPORTED_SUFFIXES))}")
        size = source.stat().st_size
        if size > self.max_upload_bytes:
            raise ValueError(f"file is {size} bytes; configured limit is {self.max_upload_bytes} bytes")
        # Chunked read limits transient memory and detects a file that grows after stat().
        digest = hashlib.sha256()
        payload = bytearray()
        with source.open("rb") as handle:
            while True:
                chunk = handle.read(1024 * 1024)
                if not chunk:
                    break
                payload.extend(chunk)
                digest.update(chunk)
                if len(payload) > self.max_upload_bytes:
                    raise ValueError("file exceeded the upload size limit while being read")
        stored = self._store_payload(
            bytes(payload), filename=source.name, license_name=license_name, source_uri=source_uri,
            title=title,
        )
        # Guard against unexpected difference if a source changed during the read.
        if stored.get("duplicate") is False and stored.get("source_sha256") != digest.hexdigest():
            with self.db:
                self.db.execute("UPDATE learning_jobs SET state='failed',error=?,updated_at=? WHERE id=?", ("Source changed while being staged.", time.time(), stored["id"]))
            raise RuntimeError("source changed while being staged; job was quarantined")
        return stored

    def enqueue_text(
        self,
        text: str,
        *,
        title: str = "pasted-text.txt",
        license_name: str = "user-provided",
        source_uri: str = "",
    ) -> dict[str, Any]:
        content = str(text)
        raw = content.encode("utf-8", errors="strict")
        name = Path(title or "pasted-text.txt").name
        if Path(name).suffix.lower() not in SUPPORTED_SUFFIXES:
            name = f"{name}.txt"
        return self._store_payload(raw, filename=name, license_name=license_name, source_uri=source_uri, title=title)

    def _update(self, job_id: str, *, state: str | None = None, progress: int | None = None,
                status_message: str | None = None, metrics: dict[str, Any] | None = None,
                error: str | None = None, **fields: Any) -> None:
        allowed_fields = {
            "source_path", "extracted_sha256", "train_sha256", "evaluation_sha256", "split_strategy",
            "train_chars", "evaluation_chars", "train_text", "evaluation_text", "report_path",
            "duplicate_of", "attempts", "worker_pid",
        }
        if state in _TERMINAL_STATES and "worker_pid" not in fields:
            fields["worker_pid"] = 0
        unknown = set(fields) - allowed_fields
        if unknown:
            raise ValueError(f"unknown learning-job fields: {sorted(unknown)}")
        parts: list[str] = ["updated_at=?"]
        values: list[Any] = [time.time()]
        for name, value in (("state", state), ("progress", progress), ("status_message", status_message), ("error", error)):
            if value is not None:
                parts.append(f"{name}=?")
                values.append(value)
        if metrics is not None:
            parts.append("metrics_json=?")
            values.append(json.dumps(metrics, ensure_ascii=False, allow_nan=False))
        for name, value in fields.items():
            parts.append(f"{name}=?")
            values.append(value)
        values.append(job_id)
        self.db.execute(f"UPDATE learning_jobs SET {', '.join(parts)} WHERE id=?", values)
        self.db.commit()

    def _extract(self, row: sqlite3.Row) -> str:
        path = Path(str(row["source_path"]))
        if not path.is_file():
            raise FileNotFoundError("staged upload is missing and no durable text split is available")
        if path.stat().st_size > self.max_upload_bytes:
            raise ValueError("staged upload exceeds configured size limit")
        if path.suffix.lower() == ".parquet":
            lessons = list(iter_text_files(path, max_chars=1_000_000, parquet_max_source_bytes=self.max_upload_bytes))
            text = "\n\n".join(lesson.text for lesson in lessons if lesson.text.strip())
        else:
            text = read_text_file(path).text
        text = str(text).replace("\x00", " ").strip()
        if len(text) > self.max_extracted_chars:
            raise ValueError(f"extracted text exceeded {self.max_extracted_chars:,} character safety limit")
        if len(text) < 80:
            raise ValueError("file produced fewer than 80 readable characters")
        return text

    def _duplicate_extracted_content(self, digest: str, train_text: str, evaluation_text: str, *, exclude_job_id: str = "") -> dict[str, Any] | None:
        row = self.db.execute(
            "SELECT id,train_text,evaluation_text FROM learning_jobs WHERE extracted_sha256=? AND id<>? AND state IN ('accepted','queued','extracting','splitting','indexing','training','evaluating') ORDER BY created_at DESC LIMIT 1",
            (digest, str(exclude_job_id)),
        ).fetchone()
        if row is not None:
            return {"id": str(row["id"]), "reason": "same normalized extracted content"}
        candidate = _shingles(train_text + "\n\n" + evaluation_text)
        if not candidate:
            return None
        accepted_rows = self.db.execute(
            "SELECT id,train_text,evaluation_text FROM learning_jobs WHERE state='accepted' AND train_text<>'' ORDER BY updated_at DESC LIMIT 30"
        ).fetchall()
        for previous in accepted_rows:
            old = _shingles(str(previous["train_text"]) + "\n\n" + str(previous["evaluation_text"]))
            if not old:
                continue
            union = len(candidate | old)
            similarity = len(candidate & old) / max(1, union)
            if similarity >= self.near_duplicate_threshold:
                return {"id": str(previous["id"]), "reason": f"near-duplicate 5-gram Jaccard similarity {similarity:.3f}"}
        return None

    @staticmethod
    def _snapshot_agent(agent: Any) -> dict[str, Any]:
        snapshot: dict[str, Any] = {
            "model": {key: value.detach().clone() for key, value in agent.model.state_dict().items()},
            "optimizer": __import__("copy").deepcopy(agent.optimizer.state_dict()),
        }
        state = getattr(agent, "state", None)
        snapshot["state"] = state.detach().clone() if hasattr(state, "detach") else state
        names = (
            "total_tokens", "processed_tokens", "training_steps", "last_language_loss", "last_teaching_loss",
            "last_lesson_id", "last_lesson_accepted", "last_lesson_fixed_gain_delta", "_cognitive_step",
        )
        for name in names:
            if hasattr(agent, name):
                value = getattr(agent, name)
                if name == "_unique_training_fingerprints":
                    value = set(value)
                snapshot[name] = value
        if hasattr(agent, "_unique_training_fingerprints"):
            snapshot["_unique_training_fingerprints"] = set(agent._unique_training_fingerprints)
        return snapshot

    @staticmethod
    def _restore_agent(agent: Any, snapshot: dict[str, Any], lesson_id: int | None = None) -> None:
        agent.model.load_state_dict(snapshot["model"])
        agent.optimizer.load_state_dict(snapshot["optimizer"])
        if "state" in snapshot and hasattr(agent, "state"):
            agent.state = snapshot["state"]
        for name, value in snapshot.items():
            if name in {"model", "optimizer", "state"}:
                continue
            setattr(agent, name, value)
        if lesson_id is not None and hasattr(agent, "memory"):
            try:
                agent.memory.update_learning_status(lesson_id, accepted=False, replay_eligible=False)
                db = getattr(agent.memory, "db", None)
                row = db.execute(
                    "SELECT loss_after,learning_progress,prediction_error FROM memories WHERE id=?",
                    (int(lesson_id),),
                ).fetchone() if db is not None else None
                loss_after = float(row[0]) if row is not None else 0.0
                progress = float(row[1]) if row is not None else 0.0
                prediction_error = float(row[2]) if row is not None else 0.0
                agent.memory.update_learning_metrics(
                    lesson_id, loss_after=loss_after, learning_progress=progress,
                    prediction_error=prediction_error, action_name="teacher_external_holdout_rollback",
                )
            except Exception:
                pass
        if hasattr(agent, "save"):
            agent.save()

    def _check_existing_memory_overlap(self, agent: Any, evaluation_text: str) -> dict[str, Any] | None:
        """Reject holdouts already represented in any accepted lesson; test near-duplicates recently."""
        db = getattr(getattr(agent, "memory", None), "db", None)
        if db is None:
            return None
        candidate = _shingles(evaluation_text)
        eval_units = [
            _normalized_text(part)
            for part in re.split(r"\n\s*\n", evaluation_text)
            if len(_normalized_text(part)) >= 80
        ]
        # Exact normalized held-out units are searched against all accepted lesson rows,
        # not only recent ones. This catches a document previously taught through --teach.
        if eval_units:
            rows = db.execute(
                "SELECT id,source,title,text FROM memories WHERE kind='lesson' AND accepted=1 ORDER BY created DESC"
            ).fetchall()
            for row in rows:
                old_text = _normalized_text(str(row[3]))
                for unit in eval_units:
                    if unit in old_text:
                        return {
                            "lesson_id": int(row[0]), "source": str(row[1]), "title": str(row[2]),
                            "match": "exact_normalized_holdout_unit", "reason": "a held-out paragraph already exists inside accepted lesson memory",
                        }
        else:
            rows = db.execute(
                "SELECT id,source,title,text FROM memories WHERE kind='lesson' AND accepted=1 ORDER BY created DESC LIMIT 500"
            ).fetchall()
        # Near-duplicate shingle coverage is more expensive, so compare a bounded recent
        # window. Exact document/paragraph overlap above is checked across the full history.
        if len(candidate) >= 16:
            recent_rows = rows[:500]
            for row in recent_rows:
                old = _shingles(str(row[3]))
                if not old:
                    continue
                coverage = len(candidate & old) / max(1, len(candidate))
                if coverage >= 0.80:
                    return {
                        "lesson_id": int(row[0]), "source": str(row[1]), "title": str(row[2]),
                        "holdout_shingle_coverage": round(float(coverage), 6),
                        "reason": "the proposed evaluation split substantially overlaps an accepted lesson",
                    }
        return None

    def run_next(self, agent: Any, *, max_regression_bpb: float = 0.08, epochs: int | None = None) -> dict[str, Any] | None:
        row = self.db.execute("SELECT id FROM learning_jobs WHERE state='queued' ORDER BY created_at ASC LIMIT 1").fetchone()
        if row is None:
            return None
        return self.run_job(str(row["id"]), agent, max_regression_bpb=max_regression_bpb, epochs=epochs)

    def run_job(self, job_id: str, agent: Any, *, max_regression_bpb: float = 0.08, epochs: int | None = None) -> dict[str, Any]:
        row = self._row(job_id)
        if row["state"] != "queued":
            raise ValueError(f"job {job_id} is {row['state']}; only queued jobs can run")
        if max_regression_bpb < 0 or not math.isfinite(float(max_regression_bpb)):
            raise ValueError("max_regression_bpb must be finite and non-negative")
        # Compare-and-swap prevents concurrent CLI instances from training on the same job.
        claim = self.db.execute(
            "UPDATE learning_jobs SET state='extracting', progress=1, status_message=?, worker_pid=?, updated_at=? WHERE id=? AND state='queued'",
            ("Claimed by worker process; preparing source-separated learning job.", os.getpid(), time.time(), str(job_id)),
        )
        self.db.commit()
        if claim.rowcount != 1:
            current = self._row(job_id)
            raise ValueError(f"job {job_id} could not be claimed; current state is {current['state']} (another worker may own it)")
        row = self._row(job_id)
        metrics: dict[str, Any] = {
            "pipeline_version": "0.33.0",
            "source_sha256": str(row["source_sha256"]),
            "source_bytes": int(row["source_bytes"]),
            "provenance": json.loads(row["provenance_json"] or "{}"),
            "max_holdout_regression_bpb": float(max_regression_bpb),
        }
        lesson_id: int | None = None
        snapshot: dict[str, Any] | None = None
        try:
            if str(row["train_text"]).strip() and str(row["evaluation_text"]).strip():
                train_text = str(row["train_text"])
                eval_text = str(row["evaluation_text"])
                self._update(job_id, state="indexing", progress=28, status_message="Resuming from durable source-separated split.", metrics=metrics)
            else:
                self._update(job_id, state="extracting", progress=8, status_message="Extracting supported text from uploaded file.", metrics=metrics)
                row = self._row(job_id)
                full_text = self._extract(row)
                extracted_sha = _text_fingerprint(full_text)
                self._update(job_id, state="splitting", progress=18, status_message="Separating source groups before model chunking.", extracted_sha256=extracted_sha)
                split = split_source_text(full_text, source_hash=str(row["source_sha256"]), holdout_fraction=self.holdout_fraction)
                original_train_chars = len(split["train_text"])
                original_eval_chars = len(split["evaluation_text"])
                original_train_units = int(split["train_units"])
                original_eval_units = int(split["evaluation_units"])
                split["train_text"], train_capped, train_units_used = _cap_text(
                    split["train_text"], self.max_training_chars, seed=str(row["source_sha256"]) + ":train"
                )
                split["evaluation_text"], eval_capped, eval_units_used = _cap_text(
                    split["evaluation_text"], self.max_evaluation_chars, seed=str(row["source_sha256"]) + ":eval"
                )
                # Capping samples whole units but can alter shingle ratios. Recheck the
                # exact capped views that will train/evaluate, not only their larger parents.
                _validate_separated_texts(split["train_text"], split["evaluation_text"])
                split["train_chars"] = len(split["train_text"])
                split["evaluation_chars"] = len(split["evaluation_text"])
                split["source_train_units_before_cap"] = original_train_units
                split["source_evaluation_units_before_cap"] = original_eval_units
                split["train_units"] = train_units_used
                split["evaluation_units"] = eval_units_used
                split["train_sha256"] = _text_fingerprint(split["train_text"])
                split["evaluation_sha256"] = _text_fingerprint(split["evaluation_text"])
                split["source_train_chars_before_cap"] = original_train_chars
                split["source_evaluation_chars_before_cap"] = original_eval_chars
                split["training_capped"] = train_capped
                split["evaluation_capped"] = eval_capped
                duplicate = self._duplicate_extracted_content(extracted_sha, split["train_text"], split["evaluation_text"], exclude_job_id=job_id)
                if duplicate:
                    metrics.update({
                        "extracted_sha256": extracted_sha,
                        "duplicate_reason": duplicate["reason"],
                        "duplicate_of": duplicate["id"],
                    })
                    self._update(job_id, state="rejected", progress=100, status_message="Duplicate material rejected before model update.",
                                 metrics=metrics, duplicate_of=duplicate["id"], error=f"duplicate of job {duplicate['id']}: {duplicate['reason']}")
                    self._write_report(job_id, metrics)
                    Path(str(row["source_path"])).unlink(missing_ok=True)
                    self._update(job_id, source_path="")
                    return self.get_job(job_id)
                train_text, eval_text = split["train_text"], split["evaluation_text"]
                metrics.update({
                    "extracted_sha256": extracted_sha,
                    "split_strategy": split["strategy"],
                    "split_units": split["unit_count"],
                    "train_units": split["train_units"],
                    "evaluation_units": split["evaluation_units"],
                    "train_chars": split["train_chars"],
                    "evaluation_chars": split["evaluation_chars"],
                    "source_train_chars_before_cap": split["source_train_chars_before_cap"],
                    "source_evaluation_chars_before_cap": split["source_evaluation_chars_before_cap"],
                    "source_train_units_before_cap": split["source_train_units_before_cap"],
                    "source_evaluation_units_before_cap": split["source_evaluation_units_before_cap"],
                    "training_capped": split["training_capped"],
                    "evaluation_capped": split["evaluation_capped"],
                    "training_char_budget": self.max_training_chars,
                    "evaluation_char_budget": self.max_evaluation_chars,
                    "train_sha256": split["train_sha256"],
                    "evaluation_sha256": split["evaluation_sha256"],
                    "split_note": "Job holdout is stored only in SQLite; it is never passed to training, replay or tokenizer fitting.",
                })
                self._update(
                    job_id, state="indexing", progress=28, status_message="Persisting separated text and provenance before training.",
                    metrics=metrics, extracted_sha256=extracted_sha, train_sha256=split["train_sha256"],
                    evaluation_sha256=split["evaluation_sha256"], split_strategy=split["strategy"],
                    train_chars=len(train_text), evaluation_chars=len(eval_text), train_text=train_text,
                    evaluation_text=eval_text, source_path="",
                )
                # The original file is no longer needed after the durable split commit.
                try:
                    Path(str(row["source_path"])).unlink(missing_ok=True)
                except OSError:
                    pass
            if not train_text or not eval_text:
                raise ValueError("durable source split is incomplete")
            # Indexing stores provenance/split hashes in the job record. Only train_text is
            # passed into agent memory; evaluation_text remains unavailable to replay.
            contaminated = self._check_existing_memory_overlap(agent, eval_text)
            if contaminated:
                metrics.update({"decision": "rejected", "decision_reason": "source holdout overlaps accepted prior training data", "holdout_contamination": contaminated})
                self._update(job_id, state="rejected", progress=100, status_message="Evaluation holdout overlaps earlier accepted lesson data; no model update was run.", metrics=metrics, error="holdout contamination detected; source-level score would not be independent")
                self._write_report(job_id, metrics)
                return self.get_job(job_id)
            before = float(agent.evaluate_text_bpb(eval_text))
            if not math.isfinite(before):
                raise RuntimeError("pre-training evaluation returned a non-finite score")
            metrics["source_holdout_bpb_before"] = before
            metrics["fixed_reference_before"] = agent.evaluate_fixed_reference() if hasattr(agent, "evaluate_fixed_reference") else {"available": False}
            self._update(job_id, state="training", progress=40, status_message="Training from the training split; evaluation split is withheld.", metrics=metrics, attempts=int(row["attempts"]) + 1)
            snapshot = self._snapshot_agent(agent)
            from .curriculum import Lesson
            lesson = Lesson(
                title=str(row["source_name"]),
                source=f"learning-job://{job_id}/{str(row['source_sha256'])}",
                text=train_text,
                kind="lesson",
            )
            result = agent.teach_text(lesson, epochs=epochs, dialogue=False, preserve_literal=True)
            lesson_id = int(result.get("lesson_id")) if result.get("lesson_id") is not None else None
            metrics["lesson_training_result"] = result
            self._update(job_id, state="evaluating", progress=80, status_message="Evaluating on the withheld source split and fixed reference.", metrics=metrics)
            after = float(agent.evaluate_text_bpb(eval_text))
            if not math.isfinite(after):
                raise RuntimeError("post-training evaluation returned a non-finite score")
            fixed_after = agent.evaluate_fixed_reference() if hasattr(agent, "evaluate_fixed_reference") else {"available": False}
            delta = after - before
            internal_accepted = bool(result.get("lesson_accepted", True))
            external_accepted = delta <= float(max_regression_bpb)
            accepted = internal_accepted and external_accepted
            metrics.update({
                "source_holdout_bpb_after": after,
                "source_holdout_delta_bpb": delta,
                "fixed_reference_after": fixed_after,
                "internal_lesson_accepted": internal_accepted,
                "external_holdout_accepted": external_accepted,
                "decision": "accepted" if accepted else "rejected",
                "decision_reason": (
                    "internal lesson checks and source holdout gate passed" if accepted else
                    "internal lesson acceptance gate rejected the update" if not internal_accepted else
                    f"source holdout regression {delta:.6f} bpb exceeded tolerance {float(max_regression_bpb):.6f}"
                ),
            })
            if not accepted:
                self._restore_agent(agent, snapshot, lesson_id)
            else:
                # Persist a final accepted checkpoint after all source-level checks.
                agent.save()
            self._update(
                job_id, state="accepted" if accepted else "rejected", progress=100,
                status_message="Learning update passed source-level evaluation." if accepted else "Learning update was rejected; prior model state restored.",
                metrics=metrics, error="" if accepted else str(metrics["decision_reason"]),
            )
            self._write_report(job_id, metrics)
            return self.get_job(job_id)
        except Exception as exc:
            if snapshot is not None:
                try:
                    self._restore_agent(agent, snapshot, lesson_id)
                except Exception as restore_exc:
                    metrics["restore_error"] = repr(restore_exc)
            metrics["exception"] = repr(exc)
            self._update(job_id, state="failed", progress=100, status_message="Job failed; the error and available recovery result were recorded.", metrics=metrics, error=f"{type(exc).__name__}: {exc}")
            self._write_report(job_id, metrics)
            # Staged raw source is intentionally removed after a job has reached a terminal state.
            latest = self._row(job_id)
            if latest["source_path"]:
                try:
                    Path(str(latest["source_path"])).unlink(missing_ok=True)
                except OSError:
                    pass
                self._update(job_id, source_path="")
            return self.get_job(job_id)

    def _write_report(self, job_id: str, metrics: dict[str, Any]) -> Path:
        path = self.report_dir / f"{job_id}.json"
        self._update(job_id, report_path=str(path))
        report = {
            "schema_version": 1,
            "job_id": str(job_id),
            "generated_at": time.time(),
            "job": self.get_job(job_id),
            "metrics": metrics,
        }
        fd, temporary_name = tempfile.mkstemp(prefix=f"{job_id}.", suffix=".json.tmp", dir=self.report_dir)
        try:
            with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
                handle.write(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False, default=str) + "\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary_name, path)
        finally:
            try:
                Path(temporary_name).unlink(missing_ok=True)
            except OSError:
                pass
        return path

    def retry(self, job_id: str) -> dict[str, Any]:
        """Queue a failed job again using its durable split, without re-extracting uploads."""
        row = self._row(job_id)
        if row["state"] not in {"failed", "rejected"}:
            raise ValueError("only failed or rejected jobs can be explicitly retried")
        if not row["train_text"] or not row["evaluation_text"]:
            raise ValueError("job has no durable separated text; upload the source again")
        self._update(job_id, state="queued", progress=0, status_message="Manually retried using the existing train/evaluation split.", error="", metrics={})
        return self.get_job(job_id)
