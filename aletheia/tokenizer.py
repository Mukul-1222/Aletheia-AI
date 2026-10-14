"""From-scratch byte-fallback BPE tokenizer.

The base vocabulary is always the 256 raw UTF-8 byte values plus PAD/BOS/EOS.
Optional merge tokens are learned only from user-supplied text; no external
vocabulary, tokenizer model, or pretrained resource is used.
"""

from __future__ import annotations

import json
import re
import heapq
from collections import Counter
from pathlib import Path
from typing import Iterable, Sequence

PAD = 256
BOS = 257
EOS = 258
BASE_VOCAB_SIZE = 259
VOCAB_SIZE = BASE_VOCAB_SIZE
SCHEMA_VERSION = 2


class TokenizerError(ValueError):
    pass


class ByteFallbackTokenizer:
    """Legacy tokenizer: exact UTF-8 bytes with three control tokens."""

    name = "byte"
    vocab_size = BASE_VOCAB_SIZE

    def encode(self, text: str, *, bos: bool = True, eos: bool = True) -> list[int]:
        return self.encode_bytes(text.encode("utf-8", errors="replace"), bos=bos, eos=eos)

    def encode_bytes(self, raw: bytes, *, bos: bool = True, eos: bool = True) -> list[int]:
        out = [BOS] if bos else []
        out.extend(raw)
        if eos:
            out.append(EOS)
        return out

    def decode(self, tokens: Sequence[int]) -> str:
        raw = bytes(int(t) for t in tokens if 0 <= int(t) < 256)
        return raw.decode("utf-8", errors="replace")

    def tokenize_count(self, text: str) -> int:
        return len(text.encode("utf-8", errors="replace"))

    def token_byte_lengths(self, tokens: Sequence[int]) -> list[int]:
        """Return bytes represented by each token; control tokens contribute zero."""
        return [1 if 0 <= int(token) < 256 else 0 for token in tokens]

    def state(self) -> dict:
        return {"schema_version": SCHEMA_VERSION, "kind": self.name}

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.state(), sort_keys=True), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "ByteFallbackTokenizer":
        _ = path
        return cls()


class LearnedBPETokenizer:
    """Byte-fallback BPE learned from supplied UTF-8 text only.

    The merge representation preserves exact bytes: every learned token is a byte
    sequence, so decoding a sequence of tokens reconstructs the original bytes.
    """

    name = "bpe"

    def __init__(self, merges: Sequence[tuple[int, int]], token_bytes: Sequence[bytes]):
        self.merges = [tuple(map(int, pair)) for pair in merges]
        self.token_bytes = [bytes(value) for value in token_bytes]
        if len(self.token_bytes) < BASE_VOCAB_SIZE:
            self.token_bytes = [bytes([i]) for i in range(256)] + [b"", b"", b""]
        self.vocab_size = len(self.token_bytes)
        self._merge_rank = {pair: idx for idx, pair in enumerate(self.merges)}
        self._trie: dict = {}
        for token_id in range(BASE_VOCAB_SIZE, self.vocab_size):
            raw = self.token_bytes[token_id]
            if not raw:
                continue
            node = self._trie
            for byte in raw:
                node = node.setdefault(int(byte), {})
            node[None] = token_id
        self.max_token_bytes = max((len(v) for v in self.token_bytes[BASE_VOCAB_SIZE:] if v), default=1)

    @classmethod
    def empty(cls) -> "LearnedBPETokenizer":
        return cls([], [bytes([i]) for i in range(256)] + [b"", b"", b""])

    @staticmethod
    def _segments(text: str, max_segments: int = 50000) -> list[bytes]:
        # Whitespace/non-whitespace segmentation is purely mechanical and preserves
        # every byte. It prevents BPE training from merging unrelated document tails.
        values = re.findall(r"\s*\S+|\s+", text, flags=re.UNICODE)
        return [part.encode("utf-8", errors="replace") for part in values[:max_segments] if part]

    @staticmethod
    def _apply_pair(sequence: tuple[int, ...], pair: tuple[int, int], merged_id: int) -> tuple[int, ...]:
        left, right = pair
        out: list[int] = []
        idx = 0
        while idx < len(sequence):
            if idx + 1 < len(sequence) and sequence[idx] == left and sequence[idx + 1] == right:
                out.append(merged_id)
                idx += 2
            else:
                out.append(sequence[idx])
                idx += 1
        return tuple(out)

    @classmethod
    def train(
        cls,
        texts: Iterable[str],
        *,
        vocab_size: int = 1024,
        min_frequency: int = 2,
        max_training_bytes: int = 4_000_000,
        max_segments: int = 50000,
    ) -> tuple["LearnedBPETokenizer", dict]:
        target_vocab = max(BASE_VOCAB_SIZE, int(vocab_size))
        min_frequency = max(1, int(min_frequency))
        byte_budget = max(1, int(max_training_bytes))
        segment_map: Counter[tuple[int, ...]] = Counter()
        consumed = 0
        for text in texts:
            if consumed >= byte_budget:
                break
            raw = str(text).encode("utf-8", errors="replace")
            if not raw:
                continue
            remaining = max(0, byte_budget - consumed)
            clipped = raw[:remaining]
            consumed += len(clipped)
            decoded = clipped.decode("utf-8", errors="ignore")
            for segment in cls._segments(decoded, max_segments=max_segments):
                segment_map[tuple(segment)] += 1

        token_bytes = [bytes([i]) for i in range(256)] + [b"", b"", b""]
        merges: list[tuple[int, int]] = []
        sequences = dict(segment_map)

        def sequence_pairs(sequence: tuple[int, ...]) -> Counter[tuple[int, int]]:
            counts: Counter[tuple[int, int]] = Counter()
            if len(sequence) < 2:
                return counts
            for left, right in zip(sequence, sequence[1:]):
                counts[(left, right)] += 1
            return counts

        # Count each pair once, then update the affected sequences after every merge.
        # This avoids recounting the entire corpus for every vocabulary slot.
        pair_counts: Counter[tuple[int, int]] = Counter()
        for sequence, weight in sequences.items():
            for pair, count in sequence_pairs(sequence).items():
                pair_counts[pair] += count * weight

        while len(token_bytes) < target_vocab:
            eligible = [(count, pair) for pair, count in pair_counts.items() if count >= min_frequency]
            if not eligible:
                break
            # Highest frequency first; lexicographic tie-breaking makes training deterministic.
            _, pair = max(eligible, key=lambda item: (item[0], tuple(-x for x in item[1])))
            merged_id = len(token_bytes)
            left, right = pair
            if left >= len(token_bytes) or right >= len(token_bytes):
                break
            raw = token_bytes[left] + token_bytes[right]
            if not raw:
                break
            token_bytes.append(raw)
            merges.append(pair)

            updated: dict[tuple[int, ...], int] = {}
            for sequence, weight in sequences.items():
                old_pairs = sequence_pairs(sequence)
                if pair not in old_pairs:
                    updated[sequence] = updated.get(sequence, 0) + weight
                    continue
                merged = cls._apply_pair(sequence, pair, merged_id)
                new_pairs = sequence_pairs(merged)
                for affected_pair, count in old_pairs.items():
                    pair_counts[affected_pair] -= count * weight
                for affected_pair, count in new_pairs.items():
                    pair_counts[affected_pair] += count * weight
                updated[merged] = updated.get(merged, 0) + weight
            # Remove zero/negative entries so future selection stays compact.
            pair_counts += Counter()
            sequences = updated

        tokenizer = cls(merges, token_bytes)
        stats = {
            "kind": tokenizer.name,
            "vocab_size": tokenizer.vocab_size,
            "merges": len(merges),
            "training_bytes": consumed,
            "segments": len(segment_map),
            "min_frequency": min_frequency,
        }
        return tokenizer, stats

    def _bpe_ids(self, raw: bytes) -> list[int]:
        """Apply ranked BPE merges in O(n log n) rather than rescanning all bytes.

        The heap orders merges by learned rank and then source position. Version stamps
        discard stale heap entries after neighbouring tokens are replaced.
        """
        if not raw:
            return []
        tokens = list(raw)
        size = len(tokens)
        previous = [idx - 1 for idx in range(size)]
        following = [idx + 1 if idx + 1 < size else -1 for idx in range(size)]
        alive = [True] * size
        versions = [0] * size
        heap: list[tuple[int, int, int, int]] = []

        def enqueue(left_index: int) -> None:
            if left_index < 0 or not alive[left_index]:
                return
            right_index = following[left_index]
            if right_index < 0 or not alive[right_index]:
                return
            rank = self._merge_rank.get((tokens[left_index], tokens[right_index]))
            if rank is not None:
                heapq.heappush(heap, (int(rank), left_index, versions[left_index], versions[right_index]))

        for idx in range(size - 1):
            enqueue(idx)

        while heap:
            rank, left_index, left_version, right_version = heapq.heappop(heap)
            if not alive[left_index] or versions[left_index] != left_version:
                continue
            right_index = following[left_index]
            if right_index < 0 or not alive[right_index] or versions[right_index] != right_version:
                continue
            if self._merge_rank.get((tokens[left_index], tokens[right_index])) != rank:
                continue

            left_neighbor = previous[left_index]
            right_neighbor = following[right_index]
            tokens[left_index] = BASE_VOCAB_SIZE + rank
            versions[left_index] += 1
            alive[right_index] = False
            versions[right_index] += 1
            following[left_index] = right_neighbor
            if right_neighbor >= 0:
                previous[right_neighbor] = left_index
            enqueue(left_neighbor)
            enqueue(left_index)

        result: list[int] = []
        cursor = 0
        while cursor >= 0:
            if alive[cursor]:
                result.append(tokens[cursor])
            cursor = following[cursor]
        return result

    def encode_bytes(self, raw: bytes, *, bos: bool = True, eos: bool = True) -> list[int]:
        out = [BOS] if bos else []
        out.extend(self._bpe_ids(raw))
        if eos:
            out.append(EOS)
        return out

    def encode(self, text: str, *, bos: bool = True, eos: bool = True) -> list[int]:
        return self.encode_bytes(text.encode("utf-8", errors="replace"), bos=bos, eos=eos)

    def decode(self, tokens: Sequence[int]) -> str:
        raw = bytearray()
        for token in tokens:
            token_id = int(token)
            if 0 <= token_id < 256:
                raw.append(token_id)
            elif BASE_VOCAB_SIZE <= token_id < self.vocab_size:
                raw.extend(self.token_bytes[token_id])
        return bytes(raw).decode("utf-8", errors="replace")

    def tokenize_count(self, text: str) -> int:
        return len(self.encode(text, bos=False, eos=False))

    def token_byte_lengths(self, tokens: Sequence[int]) -> list[int]:
        """Return exact UTF-8 byte counts represented by each token."""
        lengths: list[int] = []
        for token in tokens:
            token_id = int(token)
            if 0 <= token_id < 256:
                lengths.append(1)
            elif BASE_VOCAB_SIZE <= token_id < self.vocab_size:
                lengths.append(len(self.token_bytes[token_id]))
            else:
                lengths.append(0)
        return lengths

    def state(self) -> dict:
        return {
            "schema_version": SCHEMA_VERSION,
            "kind": self.name,
            "vocab_size": self.vocab_size,
            "merges": [[int(a), int(b)] for a, b in self.merges],
            "token_bytes": [value.hex() for value in self.token_bytes],
        }

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(json.dumps(self.state(), ensure_ascii=False), encoding="utf-8")
        tmp.replace(path)

    @classmethod
    def load(cls, path: Path) -> "LearnedBPETokenizer":
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("kind") != cls.name:
            raise TokenizerError("tokenizer file is not a learned BPE tokenizer")
        raw_merges = payload.get("merges", [])
        raw_bytes = payload.get("token_bytes", [])
        merges = [(int(a), int(b)) for a, b in raw_merges]
        token_bytes = [bytes.fromhex(str(value)) for value in raw_bytes]
        tokenizer = cls(merges, token_bytes)
        expected = int(payload.get("vocab_size", tokenizer.vocab_size))
        if expected != tokenizer.vocab_size:
            raise TokenizerError("tokenizer vocabulary size does not match serialized tokens")
        return tokenizer


# Backward-compatible module functions used by old tests and small standalone probes.
# New Agent instances use their persisted tokenizer object directly.
_BYTE = ByteFallbackTokenizer()

def encode(text: str, *, bos: bool = True, eos: bool = True) -> list[int]:
    return _BYTE.encode(text, bos=bos, eos=eos)


def decode(tokens: list[int]) -> str:
    return _BYTE.decode(tokens)


def load_tokenizer(path: Path) -> ByteFallbackTokenizer | LearnedBPETokenizer:
    if not Path(path).exists():
        return ByteFallbackTokenizer()
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if payload.get("kind") == "bpe":
            return LearnedBPETokenizer.load(Path(path))
    except (OSError, ValueError, TypeError, TokenizerError):
        pass
    return ByteFallbackTokenizer()
