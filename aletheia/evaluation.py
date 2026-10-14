from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import math
from pathlib import Path
from typing import Iterable


def _split_bytes(text: str, train_fraction: float = 0.8) -> tuple[bytes, bytes]:
    data = text.encode("utf-8", errors="replace")
    if len(data) < 8:
        return data, b""
    split = min(len(data) - 1, max(1, int(len(data) * float(train_fraction))))
    return data[:split], data[split - 1:]


def tuned_byte_unigram_bpb(text: str, train_fraction: float = 0.8) -> tuple[float, float]:
    data = text.encode("utf-8", errors="replace")
    if len(data) < 32:
        return 8.0, 1.0
    split = min(len(data) - 1, max(8, int(len(data) * float(train_fraction))))
    fit = data[:split]
    validation = fit[max(0, int(len(fit) * 0.8) - 1):]
    candidates = (0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0)

    def score(train: bytes, holdout: bytes, alpha: float) -> float:
        counts = [float(alpha)] * 256
        total = 256.0 * float(alpha)
        for value in train:
            counts[value] += 1.0
            total += 1.0
        nll = -sum(math.log2(counts[value] / total) for value in holdout[1:])
        return nll / max(1, len(holdout) - 1)

    best = min(candidates, key=lambda a: score(fit[:max(8, len(fit) * 4 // 5)], validation, a))
    test = data[split - 1:]
    return score(fit, test, best), best


def _ngram_score(train: bytes, test: bytes, order: int, alpha: float) -> float:
    """Sparse interpolated-backoff byte n-gram score."""
    if len(train) < 2 or len(test) < 2:
        return 8.0
    tables: list[dict[tuple[int, ...], Counter[int]]] = [defaultdict(Counter) for _ in range(order)]
    for idx, value in enumerate(train):
        for n in range(1, min(order, idx + 1) + 1):
            context = tuple(train[idx - n + 1:idx]) if n > 1 else tuple()
            tables[n - 1][context][value] += 1

    nll = 0.0
    evaluated = 0
    vocab = 256.0
    for idx in range(1, len(test)):
        value = test[idx]
        chosen = None
        for n in range(order, 0, -1):
            context = tuple(test[max(0, idx - (n - 1)):idx]) if n > 1 else tuple()
            counts = tables[n - 1].get(context)
            if counts:
                chosen = counts
                total = float(sum(counts.values()))
                break
        if chosen is None:
            prob = 1.0 / vocab
        else:
            prob = (float(chosen.get(value, 0)) + float(alpha)) / (total + float(alpha) * vocab)
        nll -= math.log2(max(prob, 1e-12))
        evaluated += 1
    return nll / max(1, evaluated)


def tuned_byte_ngram_bpb(text: str, order: int, train_fraction: float = 0.8) -> tuple[float, float]:
    data = text.encode("utf-8", errors="replace")
    if len(data) < 128:
        return 8.0, 1.0
    test_split = min(len(data) - 1, max(32, int(len(data) * float(train_fraction))))
    fit = data[:test_split]
    final_test = data[test_split - 1:]
    inner_split = min(len(fit) - 1, max(32, int(len(fit) * 0.8)))
    inner_train = fit[:inner_split]
    validation = fit[inner_split - 1:]
    candidates = (0.001, 0.005, 0.01, 0.03, 0.05, 0.1, 0.25, 0.5, 1.0)
    best_alpha = min(candidates, key=lambda alpha: _ngram_score(inner_train, validation, order, alpha))
    return _ngram_score(fit, final_test, order, best_alpha), best_alpha


def byte_unigram_bpb(text: str, train_fraction: float = 0.8, alpha: float = 1.0) -> float:
    data = text.encode("utf-8", errors="replace")
    train, test = _split_bytes(text, train_fraction)
    if len(test) < 2:
        return 8.0
    counts = [float(alpha)] * 256
    total = 256.0 * float(alpha)
    for value in train:
        counts[value] += 1.0
        total += 1.0
    nll = -sum(math.log2(counts[value] / total) for value in test[1:])
    return nll / max(1, len(test) - 1)


def byte_bigram_bpb(text: str, train_fraction: float = 0.8, alpha: float = 1.0) -> float:
    return _ngram_score(*_split_bytes(text, train_fraction), 2, alpha)


def tuned_byte_bigram_bpb(text: str, train_fraction: float = 0.8) -> tuple[float, float]:
    return tuned_byte_ngram_bpb(text, 2, train_fraction)


def best_byte_baseline_bpb(text: str, train_fraction: float = 0.8) -> dict:
    unigram, unigram_alpha = tuned_byte_unigram_bpb(text, train_fraction=train_fraction)
    bigram, bigram_alpha = tuned_byte_bigram_bpb(text, train_fraction=train_fraction)
    trigram, trigram_alpha = tuned_byte_ngram_bpb(text, 3, train_fraction=train_fraction)
    fourgram, fourgram_alpha = tuned_byte_ngram_bpb(text, 4, train_fraction=train_fraction)
    candidates = {
        "unigram": (unigram, unigram_alpha),
        "bigram": (bigram, bigram_alpha),
        "trigram": (trigram, trigram_alpha),
        "fourgram": (fourgram, fourgram_alpha),
    }
    kind, (score, alpha) = min(candidates.items(), key=lambda item: item[1][0])
    return {
        "bpb": score,
        "kind": kind,
        "order": {"unigram": 1, "bigram": 2, "trigram": 3, "fourgram": 4}[kind],
        "alpha": alpha,
        "alternates": {key: value[0] for key, value in candidates.items()},
    }


def corpus_fingerprint(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()


def load_reference(root: Path, relative: str, max_chars: int = 0) -> str:
    """Load only the evaluation corpus belonging to this Aletheia root.

    Never fall back to the installed package directory: doing so can silently
    make one developmental instance evaluate against another instance's corpus.
    """
    path = (Path(root).resolve() / relative).resolve()
    root_resolved = Path(root).resolve()
    try:
        path.relative_to(root_resolved)
    except ValueError as exc:
        raise ValueError("evaluation corpus path must remain inside the Aletheia root") from exc
    if not path.exists():
        raise FileNotFoundError(path)
    text = path.read_text(encoding="utf-8")
    return text[:max_chars] if max_chars and max_chars > 0 else text
