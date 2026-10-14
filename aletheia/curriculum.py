from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable
import re

from .documents import is_document, read_document

_CHAT_SPEAKER_RE = re.compile(r"^\s*\*[A-Za-z][^:\n]{0,30}:\s*(.*)$")
_CHAT_META_RE = re.compile(r"^\s*[@%$][^:\n]*(?::\s*.*)?$")


def prepare_language_text(text: str, *, strip_transcript_annotations: bool = True) -> tuple[str, dict]:
    """Prepare natural-language training text while preserving raw transcript text for dialogue.

    CHILDES/CHAT exports contain speaker labels, dependent annotation tiers, and metadata.
    Those structural lines are useful to the dialogue parser but are not language content we
    want the byte-level language model to imitate. Cleaning is applied only when the input
    strongly resembles a transcript; ordinary prose and ordinary ``*``-prefixed text pass
    through unchanged.
    """
    raw = str(text)
    if not strip_transcript_annotations:
        return raw, {
            "transcript_detected": False,
            "raw_chars": len(raw),
            "cleaned_chars": len(raw),
            "removed_chars": 0,
            "annotation_lines_removed": 0,
            "speaker_labels_removed": 0,
        }

    lines = raw.splitlines()
    speaker_lines = [line for line in lines if _CHAT_SPEAKER_RE.match(line)]
    speaker_labels = {line.split(":", 1)[0].strip() for line in speaker_lines}
    has_chat_boundary = any(re.match(r"^\s*@(?:begin|end)\b", line, re.IGNORECASE) for line in lines)
    transcript_detected = has_chat_boundary or (len(speaker_lines) >= 4 and len(speaker_labels) >= 2)
    if not transcript_detected:
        return raw, {
            "transcript_detected": False,
            "raw_chars": len(raw),
            "cleaned_chars": len(raw),
            "removed_chars": 0,
            "annotation_lines_removed": 0,
            "speaker_labels_removed": 0,
        }

    cleaned: list[str] = []
    removed_chars = 0
    annotation_lines_removed = 0
    speaker_labels_removed = 0

    for line in lines:
        stripped = line.strip()
        if not stripped:
            if cleaned and cleaned[-1] != "":
                cleaned.append("")
            continue
        if _CHAT_META_RE.match(line):
            removed_chars += len(line) + 1
            annotation_lines_removed += 1
            continue
        match = _CHAT_SPEAKER_RE.match(line)
        if match:
            body = match.group(1).strip()
            if body:
                cleaned.append(body)
            speaker_prefix = line[: match.start(1)]
            removed_chars += len(speaker_prefix)
            speaker_labels_removed += 1
            continue
        cleaned.append(stripped)

    while cleaned and cleaned[-1] == "":
        cleaned.pop()
    result = "\n".join(cleaned).strip()
    return result, {
        "transcript_detected": True,
        "raw_chars": len(raw),
        "cleaned_chars": len(result),
        "removed_chars": max(0, len(raw) - len(result)),
        "annotation_lines_removed": annotation_lines_removed,
        "speaker_labels_removed": speaker_labels_removed,
    }


@dataclass(frozen=True)
class Lesson:
    title: str
    source: str
    text: str
    kind: str = "lesson"


def _read_bytes(path: Path) -> str:
    raw = path.read_bytes()
    if b"\x00" in raw[:65536]:
        raise ValueError(f"binary file skipped: {path}")
    sample = raw[:65536]
    control = sum(1 for b in sample if b < 9 or (13 < b < 32))
    if sample and control / len(sample) > 0.02:
        raise ValueError(f"binary-like file skipped: {path}")
    # UTF-16 is accepted only when a BOM explicitly identifies it. Arbitrary
    # even-length cp1252 files can decode as UTF-16 without raising an error.
    if raw.startswith((b"\xff\xfe", b"\xfe\xff")):
        try:
            return raw.decode("utf-16")
        except UnicodeDecodeError:
            pass
    for encoding in ("utf-8-sig", "cp1252", "latin-1"):
        try:
            return raw.decode(encoding)
        except UnicodeDecodeError:
            continue
    return raw.decode("utf-8", errors="replace")


def read_text_file(path: Path, *, max_chars: int = 0) -> Lesson:
    path = path.expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(path)
    if path.is_dir():
        raise IsADirectoryError(path)
    if path.suffix.lower() == ".parquet":
        raise ValueError("Parquet is a streaming dataset source; use iter_text_files or --teach/--pretrain on the parquet file")
    text = read_document(path) if is_document(path) else _read_bytes(path)
    text = text.replace("\x00", " ").strip()
    if max_chars and max_chars > 0:
        text = text[:max_chars]
    if not text:
        raise ValueError(f"no readable text found in {path}")
    return Lesson(title=path.name, source=str(path), text=text)



def iter_parquet_lessons(
    path: Path,
    *,
    shard_max_chars: int = 1_000_000,
    text_column: str = "text",
    batch_size: int = 128,
    max_source_bytes: int | None = None,
) -> Iterable[Lesson]:
    """Stream a Parquet text column into bounded lessons without loading the dataset into RAM.

    ``max_source_bytes=None`` means unlimited. When a byte budget is supplied, the total
    UTF-8 bytes emitted by this iterator (including shard separators) never exceed it.
    """
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise ValueError("Parquet support needs 'pyarrow': pip install pyarrow") from exc

    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    parquet = pq.ParquetFile(path)
    schema_names = list(parquet.schema_arrow.names)
    candidates = [text_column, "text", "content", "body"]
    selected = next((name for name in candidates if name and name in schema_names), None)
    if selected is None:
        string_columns = []
        for name in schema_names:
            try:
                if str(parquet.schema_arrow.field(name).type).lower() in {"string", "large_string"}:
                    string_columns.append(name)
            except Exception:
                continue
        selected = string_columns[0] if string_columns else None
    if selected is None:
        raise ValueError(f"no readable text column found in {path.name}; available columns: {schema_names}")

    limit = max(1, int(shard_max_chars))
    source_cap = None if max_source_bytes is None else max(0, int(max_source_bytes))
    consumed = 0
    shard_parts: list[str] = []
    shard_chars = 0
    row_start = 0

    def utf8_prefix(value: str, budget: int) -> str:
        if budget <= 0:
            return ""
        raw = value.encode("utf-8", errors="replace")
        if len(raw) <= budget:
            return value
        clipped = raw[:budget]
        return clipped.decode("utf-8", errors="ignore").strip()

    def flush_parts(title: str, source_suffix: str = "#parquet") -> Lesson | None:
        if not shard_parts:
            return None
        text = "\n\n".join(shard_parts)
        shard_parts.clear()
        return Lesson(title=title, source=f"{path}{source_suffix}", text=text, kind="lesson")

    stop = False
    for batch in parquet.iter_batches(batch_size=max(1, int(batch_size)), columns=[selected], use_threads=True):
        for value in batch.column(0).to_pylist():
            row_index = row_start
            row_start += 1
            if value is None:
                continue
            text = str(value).strip()
            if not text:
                continue
            if source_cap is not None and consumed >= source_cap:
                stop = True
                break

            # If the next piece would overflow the byte budget, first flush the current
            # shard, then clip the incoming text to the exact remaining UTF-8 capacity.
            separator_bytes = len("\n\n".encode("utf-8")) if shard_parts else 0
            available_for_text = None if source_cap is None else source_cap - consumed - separator_bytes
            if available_for_text is not None and available_for_text <= 0:
                lesson = flush_parts(f"{path.name} rows up to {row_index - 1}")
                if lesson is not None:
                    encoded = len(lesson.text.encode("utf-8", errors="replace"))
                    consumed += encoded
                    yield lesson
                stop = True
                break
            if available_for_text is not None:
                text = utf8_prefix(text, available_for_text)
            if not text:
                stop = True if source_cap is not None else False
                break

            # Respect character-based shard size independently of the byte budget.
            if len(shard_parts) and shard_chars + len(text) + 2 > limit:
                lesson = flush_parts(f"{path.name} rows through {row_index - 1}")
                if lesson is not None:
                    encoded = len(lesson.text.encode("utf-8", errors="replace"))
                    consumed += encoded
                    yield lesson
                    shard_chars = 0

                # Recompute remaining byte capacity after flushing.
                if source_cap is not None:
                    available_for_text = source_cap - consumed
                    text = utf8_prefix(text, available_for_text)
                    if not text:
                        stop = True
                        break

            if len(text) <= limit:
                shard_parts.append(text)
                shard_chars += len(text) + (2 if len(shard_parts) > 1 else 0)
            else:
                # A single row can span multiple character-bounded lessons.
                start = 0
                while start < len(text):
                    piece = text[start:start + limit]
                    start += limit
                    if not piece:
                        continue
                    lesson = Lesson(
                        title=f"{path.name} parquet row {row_index}",
                        source=f"{path}#row={row_index}",
                        text=piece,
                        kind="lesson",
                    )
                    encoded = len(piece.encode("utf-8", errors="replace"))
                    if source_cap is not None and consumed + encoded > source_cap:
                        piece = utf8_prefix(piece, source_cap - consumed)
                        encoded = len(piece.encode("utf-8", errors="replace"))
                        if not piece:
                            stop = True
                            break
                        lesson = Lesson(title=lesson.title, source=lesson.source, text=piece, kind=lesson.kind)
                    consumed += encoded
                    yield lesson
                if stop:
                    break
                continue

            # Do not count bytes twice: the text is only charged when the shard is emitted.
            if source_cap is not None:
                projected = len("\n\n".join(shard_parts).encode("utf-8", errors="replace"))
                if consumed + projected >= source_cap:
                    lesson = flush_parts(f"{path.name} parquet capped shard")
                    if lesson is not None:
                        encoded = len(lesson.text.encode("utf-8", errors="replace"))
                        remaining = max(0, source_cap - consumed)
                        if encoded > remaining:
                            clipped = utf8_prefix(lesson.text, remaining)
                            if clipped:
                                lesson = Lesson(title=lesson.title, source=lesson.source, text=clipped, kind="lesson")
                                encoded = len(clipped.encode("utf-8", errors="replace"))
                            else:
                                lesson = None
                                encoded = 0
                        if lesson is not None:
                            consumed += encoded
                            yield lesson
                    stop = consumed >= source_cap
                    if stop:
                        break
        if stop:
            break

    if shard_parts and (source_cap is None or consumed < source_cap):
        lesson = flush_parts(f"{path.name} parquet final shard")
        if lesson is not None:
            encoded = len(lesson.text.encode("utf-8", errors="replace"))
            if source_cap is not None and consumed + encoded > source_cap:
                clipped = utf8_prefix(lesson.text, source_cap - consumed)
                if clipped:
                    lesson = Lesson(title=lesson.title, source=lesson.source, text=clipped, kind="lesson")
                    encoded = len(clipped.encode("utf-8", errors="replace"))
                else:
                    lesson = None
                    encoded = 0
            if lesson is not None:
                consumed += encoded
                yield lesson

def iter_text_files(
    path: Path,
    *,
    max_chars: int = 0,
    parquet_max_source_bytes: int | None = None,
    on_skip: Callable[[Path, str], None] | None = None,
) -> Iterable[Lesson]:
    root = path.expanduser().resolve()
    if root.is_file():
        if root.suffix.lower() == ".parquet":
            yield from iter_parquet_lessons(root, shard_max_chars=max_chars or 1_000_000, max_source_bytes=parquet_max_source_bytes)
        else:
            yield read_text_file(root, max_chars=max_chars)
        return
    if not root.is_dir():
        raise FileNotFoundError(root)
    for child in sorted(root.rglob("*")):
        if not child.is_file():
            continue
        try:
            relative = child.relative_to(root)
        except ValueError:
            continue
        if any(part.startswith(".") for part in relative.parts):
            continue
        try:
            if child.suffix.lower() == ".parquet":
                yield from iter_parquet_lessons(child, shard_max_chars=max_chars or 1_000_000, max_source_bytes=parquet_max_source_bytes)
            else:
                yield read_text_file(child, max_chars=max_chars)
        except (UnicodeDecodeError, ValueError, OSError, RuntimeError) as exc:
            if on_skip is not None:
                on_skip(child, str(exc))
            continue


def split_lesson_text(text: str, max_chars: int, *, title: str, source: str, kind: str = "lesson") -> list[Lesson]:
    """Split a large lesson into bounded, independently evaluated shards.

    Boundaries prefer paragraph breaks, then line breaks. For line-oriented sources,
    the original single-newline turn structure is preserved so structural dialogue is
    not accidentally broken by synthetic blank lines.
    """
    text = str(text).strip()
    limit = int(max_chars)
    if not text or limit <= 0 or len(text) <= limit:
        return [Lesson(title=title, source=source, text=text, kind=kind)] if text else []

    units = [u for u in re.split(r"\n\s*\n", text) if u.strip()]
    separator = "\n\n"
    if len(units) <= 1:
        # Line-oriented sources such as CHILDES/CHAT often contain one speaker turn
        # per physical line with no blank lines. Preserve that single-newline structure
        # while sharding; inserting a blank line between every turn would destroy the
        # very dialogue runs that the dialogue parser is meant to learn from.
        units = [u for u in text.splitlines() if u.strip()]
        separator = "\n"
    if not units:
        units = [text]

    shards: list[str] = []
    current: list[str] = []
    current_len = 0
    for unit in units:
        unit = unit.strip()
        if not unit:
            continue
        if len(unit) > limit:
            if current:
                shards.append(separator.join(current))
                current, current_len = [], 0
            lines = [line.strip() for line in unit.splitlines() if line.strip()] or [unit]
            line_buf: list[str] = []
            line_len = 0
            for line in lines:
                if len(line) > limit:
                    if line_buf:
                        shards.append("\n".join(line_buf))
                        line_buf, line_len = [], 0
                    start = 0
                    while start < len(line):
                        shards.append(line[start:start + limit])
                        start += limit
                    continue
                extra = len(line) + (1 if line_buf else 0)
                if line_buf and line_len + extra > limit:
                    shards.append("\n".join(line_buf))
                    line_buf, line_len = [line], len(line)
                else:
                    line_buf.append(line)
                    line_len += extra
            if line_buf:
                shards.append("\n".join(line_buf))
            continue

        extra = len(unit) + (len(separator) if current else 0)
        if current and current_len + extra > limit:
            shards.append(separator.join(current))
            current, current_len = [unit], len(unit)
        else:
            current.append(unit)
            current_len += extra
    if current:
        shards.append(separator.join(current))

    total = len(shards)
    return [
        Lesson(
            title=f"{title} [shard {index}/{total}]",
            source=f"{source}#shard={index}/{total}",
            text=shard,
            kind=kind,
        )
        for index, shard in enumerate(shards, 1)
        if shard.strip()
    ]
