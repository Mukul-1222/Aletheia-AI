"""Deterministic, group-aware train/holdout splitting for manifest-backed datasets.

Grouping uses explicit corpus metadata when available. If the manifest does not supply
usable group IDs (or has fewer than two groups), the helper falls back to a seeded
example-level split and reports that weaker guarantee explicitly.
"""
from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass
from typing import Callable, Generic, Sequence, TypeVar

T = TypeVar("T")


@dataclass(frozen=True)
class DatasetSplit(Generic[T]):
    train: list[T]
    holdout: list[T]
    strategy: str
    train_groups: int
    holdout_groups: int


def stable_manifest_seed(items: Sequence[object], *, manifest: str = "") -> int:
    """Return a stable seed that changes when manifest examples change."""
    digest = hashlib.sha256()
    digest.update(str(manifest).encode("utf-8", errors="replace"))
    digest.update(b"\0")
    # Row order and absolute installation paths must not silently change a split.
    # Callers provide content/relative-path metadata; sorting makes reordered CSV/JSONL
    # manifests produce the same seed while duplicate rows still affect the digest.
    for item_repr in sorted(repr(item) for item in items):
        digest.update(item_repr.encode("utf-8", errors="replace"))
        digest.update(b"\n")
    return int.from_bytes(digest.digest()[:8], "big")


def split_examples(
    items: Sequence[T],
    *,
    holdout_fraction: float,
    seed: int,
    group_key: Callable[[T], str | None] | None = None,
) -> DatasetSplit[T]:
    """Split a manifest deterministically, keeping known groups in a single split.

    The function never leaks a known group across training and holdout. If no explicit
    group IDs exist or only one group exists, it uses an example-level fallback and
    records that fact so callers cannot mistake it for group isolation.
    """
    values = list(items)
    if len(values) < 2:
        return DatasetSplit(values, [], "insufficient_examples", len(values), 0)
    fraction = min(0.8, max(0.0, float(holdout_fraction)))
    if fraction <= 0.0:
        return DatasetSplit(values, [], "holdout_disabled", len(values), 0)

    groups: dict[str, list[T]] = {}
    if group_key is not None:
        for item in values:
            key = group_key(item)
            if key is not None and str(key).strip():
                groups.setdefault(str(key).strip(), []).append(item)
    strategy = "grouped_metadata"
    if len(groups) < 2 or sum(len(group) for group in groups.values()) != len(values):
        groups = {f"__example_{i}": [item] for i, item in enumerate(values)}
        strategy = "example_level_no_usable_group_metadata"

    group_names = list(groups)
    rng = random.Random(int(seed))
    rng.shuffle(group_names)
    requested = max(1, int(round(len(values) * fraction)))
    holdout_names: list[str] = []
    holdout_size = 0
    # Select whole groups toward requested item count without ever taking all groups.
    for name in group_names:
        if len(holdout_names) >= len(group_names) - 1:
            break
        current_error = abs(requested - holdout_size)
        next_error = abs(requested - (holdout_size + len(groups[name])))
        if not holdout_names or next_error <= current_error:
            holdout_names.append(name)
            holdout_size += len(groups[name])
        if holdout_size >= requested:
            break
    if not holdout_names:
        holdout_names = [group_names[-1]]
    if len(holdout_names) >= len(group_names):
        holdout_names = holdout_names[:-1]
    holdout_set = set(holdout_names)
    train = [item for name, group in groups.items() if name not in holdout_set for item in group]
    holdout = [item for name, group in groups.items() if name in holdout_set for item in group]
    # Stable source ordering within each resulting split; membership is seeded above.
    return DatasetSplit(train, holdout, strategy, len(group_names) - len(holdout_names), len(holdout_names))
