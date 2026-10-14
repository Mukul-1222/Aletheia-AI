"""Leakage-aware evaluation manifests and statistically honest metric summaries.

This module deliberately keeps byte-prediction, retrieval, task-outcome, and modality
metrics separate. It does not infer semantic ability from a language-model loss.
"""
from __future__ import annotations

import hashlib
import math
import random
from collections import defaultdict
from typing import Iterable, Mapping, Sequence

ALLOWED_SPLITS = frozenset({"train", "development", "fixed-heldout", "transfer-heldout"})


def sha256_text(text: str) -> str:
    return hashlib.sha256(str(text).encode("utf-8", errors="replace")).hexdigest()


def validate_source_separated_manifest(records: Sequence[Mapping]) -> list[str]:
    """Return errors for invalid or contaminated document-level manifests.

    Each record must represent a source/document, not an overlapping chunk. The same
    source or exact content hash may not appear in multiple splits. Held-out records
    must explicitly prohibit memory/replay ingestion.
    """
    errors: list[str] = []
    source_splits: dict[str, set[str]] = defaultdict(set)
    hash_splits: dict[str, set[str]] = defaultdict(set)
    seen_ids: set[str] = set()
    for index, record in enumerate(records):
        prefix = f"record[{index}]"
        if not isinstance(record, Mapping):
            errors.append(f"{prefix}: record must be a mapping")
            continue
        item_id = str(record.get("item_id", "")).strip()
        source_id = str(record.get("source_id", "")).strip()
        split = str(record.get("split", "")).strip()
        digest = str(record.get("sha256", "")).strip().lower()
        if not item_id:
            errors.append(f"{prefix}: missing item_id")
        elif item_id in seen_ids:
            errors.append(f"{prefix}: duplicate item_id {item_id!r}")
        seen_ids.add(item_id)
        if not source_id:
            errors.append(f"{prefix}: missing source_id")
        if split not in ALLOWED_SPLITS:
            errors.append(f"{prefix}: invalid split {split!r}")
        if len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
            errors.append(f"{prefix}: sha256 must be a 64-character hex digest")
        if source_id and split in ALLOWED_SPLITS:
            source_splits[source_id].add(split)
        if digest and split in ALLOWED_SPLITS:
            hash_splits[digest].add(split)
        if split in {"fixed-heldout", "transfer-heldout"} and record.get("memory_replay_allowed") is not False:
            errors.append(f"{prefix}: held-out data must set memory_replay_allowed=false")
        if split in {"fixed-heldout", "transfer-heldout"} and record.get("tokenizer_training_allowed") is not False:
            errors.append(f"{prefix}: held-out data must set tokenizer_training_allowed=false")
    for source, splits in source_splits.items():
        if len(splits) > 1:
            errors.append(f"source {source!r} crosses splits: {', '.join(sorted(splits))}")
    for digest, splits in hash_splits.items():
        if len(splits) > 1:
            errors.append(f"content hash {digest} crosses splits: {', '.join(sorted(splits))}")
    return errors


def summarize_bpb_rows(
    rows: Iterable[Mapping], *, bootstrap_samples: int = 2000, seed: int = 1337,
    confidence: float = 0.95,
) -> dict:
    """Aggregate language NLL using bytes as the denominator and bootstrap documents.

    Required row fields are source_id, domain, nll_bits, and byte_count. A row's
    nll_bits must be the total negative log-likelihood in bits, not a per-byte mean.
    The confidence interval is over source/document BPB values, not byte-level tokens.
    """
    normalized: list[dict] = []
    for row in rows:
        source_id = str(row.get("source_id", "")).strip()
        domain = str(row.get("domain", "unknown")).strip() or "unknown"
        nll_bits = float(row.get("nll_bits", math.nan))
        byte_count = int(row.get("byte_count", 0))
        if not source_id or byte_count <= 0 or not math.isfinite(nll_bits) or nll_bits < 0:
            continue
        normalized.append({"source_id": source_id, "domain": domain, "nll_bits": nll_bits, "byte_count": byte_count})
    if not normalized:
        return {"status": "unavailable", "reason": "no_valid_rows", "documents": 0, "bytes": 0, "bpb": None, "ci": None, "by_domain": {}}

    # Multiple windows from a source combine before uncertainty is estimated, so
    # overlapping or differently sized windows cannot masquerade as independent docs.
    by_source: dict[str, dict] = {}
    for row in normalized:
        current = by_source.setdefault(row["source_id"], {"source_id": row["source_id"], "domain": row["domain"], "nll_bits": 0.0, "byte_count": 0})
        if current["domain"] != row["domain"]:
            raise ValueError(f"source {row['source_id']!r} has inconsistent domains")
        current["nll_bits"] += row["nll_bits"]
        current["byte_count"] += row["byte_count"]
    documents = list(by_source.values())
    total_bits = sum(item["nll_bits"] for item in documents)
    total_bytes = sum(item["byte_count"] for item in documents)
    by_domain: dict[str, dict] = {}
    for domain in sorted({item["domain"] for item in documents}):
        subset = [item for item in documents if item["domain"] == domain]
        by_domain[domain] = {
            "documents": len(subset),
            "bytes": sum(item["byte_count"] for item in subset),
            "bpb": sum(item["nll_bits"] for item in subset) / sum(item["byte_count"] for item in subset),
        }
    conf = min(0.999, max(0.50, float(confidence)))
    tail = (1.0 - conf) / 2.0
    samples = max(0, int(bootstrap_samples))
    ci = None
    if len(documents) >= 2 and samples:
        rng = random.Random(int(seed))
        draws = []
        for _ in range(samples):
            picked = [documents[rng.randrange(len(documents))] for _ in documents]
            denom = sum(item["byte_count"] for item in picked)
            draws.append(sum(item["nll_bits"] for item in picked) / max(1, denom))
        draws.sort()
        lo = draws[min(len(draws) - 1, int(tail * len(draws)))]
        hi = draws[min(len(draws) - 1, int((1.0 - tail) * len(draws)))]
        ci = [lo, hi]
    return {
        "status": "scored", "documents": len(documents), "bytes": total_bytes,
        "nll_bits": total_bits, "bpb": total_bits / total_bytes, "ci": ci,
        "confidence": conf, "bootstrap_samples": samples if len(documents) >= 2 else 0,
        "seed": int(seed), "by_domain": by_domain,
        "interpretation": "next-byte prediction only; not a measure of understanding or task success",
    }
