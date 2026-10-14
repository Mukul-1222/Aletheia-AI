from __future__ import annotations

import json
import re
import hashlib
from datetime import datetime, timezone

from .evaluation_protocol import validate_source_separated_manifest
from pathlib import Path
from urllib.request import Request, urlopen

DEFAULT_SOURCES = [
    ("Pride and Prejudice", "https://www.gutenberg.org/cache/epub/1342/pg1342.txt", "literature"),
    ("The Adventures of Sherlock Holmes", "https://www.gutenberg.org/cache/epub/1661/pg1661.txt", "literature"),
    ("Moby Dick", "https://www.gutenberg.org/cache/epub/2701/pg2701.txt", "literature"),
    ("The Time Machine", "https://www.gutenberg.org/cache/epub/35/pg35.txt", "literature"),
    ("Frankenstein", "https://www.gutenberg.org/cache/epub/84/pg84.txt", "literature"),
    ("Alice's Adventures in Wonderland", "https://www.gutenberg.org/cache/epub/11/pg11.txt", "literature"),
    ("A Tale of Two Cities", "https://www.gutenberg.org/cache/epub/98/pg98.txt", "literature"),
    ("The Picture of Dorian Gray", "https://www.gutenberg.org/cache/epub/174/pg174.txt", "literature"),
    ("Adventures of Huckleberry Finn", "https://www.gutenberg.org/cache/epub/76/pg76.txt", "literature"),
    ("The Metamorphosis", "https://www.gutenberg.org/cache/epub/5200/pg5200.txt", "literature"),
    ("On the Origin of Species", "https://www.gutenberg.org/cache/epub/1228/pg1228.txt", "natural_science"),
    ("The Federalist Papers", "https://www.gutenberg.org/cache/epub/18/pg18.txt", "political_theory"),
    ("The Wealth of Nations", "https://www.gutenberg.org/cache/epub/3300/pg3300.txt", "economics"),
    ("The Republic", "https://www.gutenberg.org/cache/epub/1497/pg1497.txt", "philosophy"),
    ("The Art of War", "https://www.gutenberg.org/cache/epub/132/pg132.txt", "strategy"),
    ("Walden", "https://www.gutenberg.org/cache/epub/205/pg205.txt", "essays"),
    ("The Souls of Black Folk", "https://www.gutenberg.org/cache/epub/408/pg408.txt", "history_and_society"),
    ("The Autobiography of Benjamin Franklin", "https://www.gutenberg.org/cache/epub/20203/pg20203.txt", "biography"),
    ("The Prince", "https://www.gutenberg.org/cache/epub/1232/pg1232.txt", "political_theory"),
    ("Common Sense", "https://www.gutenberg.org/cache/epub/147/pg147.txt", "political_theory"),
]


def _fetch(url: str, timeout: float = 30.0) -> str:
    request = Request(url, headers={"User-Agent": "Aletheia-evaluation-builder/1.0"})
    with urlopen(request, timeout=timeout) as response:
        return response.read().decode("utf-8", errors="replace")


def _clean_gutenberg(text: str) -> str:
    start = re.search(r"\*\*\* START OF (?:THE|THIS) PROJECT GUTENBERG EBOOK[^\n]*\*\*\*", text, re.I)
    end = re.search(r"\*\*\* END OF (?:THE|THIS) PROJECT GUTENBERG EBOOK[^\n]*\*\*\*", text, re.I)
    if start:
        text = text[start.end():]
    if end:
        text = text[: end.start()]
    text = re.sub(r"\r\n?", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _sample_prose(text: str, target_bytes: int, seed: int) -> str:
    """Select diverse whole paragraphs until the byte budget is filled.

    Selection is deterministic for a given seed and never duplicates paragraphs.
    The previous stride sampler could underfill badly when paragraph lengths varied.
    """
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if len(p.strip()) >= 200]
    if not paragraphs or target_bytes <= 0:
        return ""
    # Deterministic coverage ordering without Python's process-randomized hash().
    import hashlib

    scored = []
    for index, paragraph in enumerate(paragraphs):
        digest = hashlib.sha256(f"{seed}:{index}".encode("utf-8")).digest()
        score = int.from_bytes(digest[:8], "big")
        scored.append((score, index, paragraph))
    scored.sort(key=lambda item: item[0])

    selected: list[str] = []
    total = 0
    seen: set[int] = set()
    # Round-robin by coarse book position: take an early, middle and later slice.
    # This keeps the fixed corpus varied while still filling the requested budget.
    buckets: list[list[tuple[int, str]]] = [[] for _ in range(8)]
    for _, index, paragraph in scored:
        bucket = min(7, (index * 8) // max(1, len(paragraphs)))
        buckets[bucket].append((index, paragraph))

    cursor = 0
    while total < target_bytes and cursor < len(paragraphs) * 2:
        bucket_index = cursor % len(buckets)
        bucket = buckets[bucket_index]
        offset = cursor // len(buckets)
        if offset < len(bucket):
            index, paragraph = bucket[offset]
            if index not in seen:
                piece = ("\n\n" if selected else "") + paragraph
                piece_bytes = len(piece.encode("utf-8"))
                if total + piece_bytes > target_bytes:
                    remaining = target_bytes - total
                    if remaining > 0:
                        raw = piece.encode("utf-8")[:remaining]
                        selected.append(raw.decode("utf-8", errors="ignore"))
                        total = target_bytes
                    break
                selected.append(piece.lstrip("\n"))
                total += piece_bytes
                seen.add(index)
        cursor += 1

    return "\n\n".join(selected)



def _iter_exclusion_files(paths: list[str] | None, *, base_dir: Path | None = None) -> list[Path]:
    files: list[Path] = []
    base = Path(base_dir).expanduser().resolve() if base_dir is not None else Path.cwd().resolve()
    for raw in paths or []:
        candidate = Path(raw).expanduser()
        if not candidate.is_absolute():
            rooted = (base / candidate).resolve()
            # Prefer project-relative configuration paths; fall back to cwd for legacy
            # invocations that explicitly prepared paths relative to their shell.
            candidate = rooted if rooted.exists() else candidate.resolve()
        if candidate.is_file():
            files.append(candidate.resolve())
        elif candidate.is_dir():
            files.extend(sorted(p.resolve() for p in candidate.rglob("*.txt") if p.is_file() and not any(part.startswith(".") for part in p.relative_to(candidate).parts)))
    return files


def _read_exclusion_text(path: Path) -> str:
    raw = path.read_bytes()
    # Only attempt UTF-16 when the file marks its byte order explicitly.
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


def _normalized_words(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", re.sub(r"\s+", " ", text.lower()))


def _build_overlap_index(paths: list[str] | None, *, shingle_words: int = 16, base_dir: Path | None = None) -> tuple[set[str], set[int], list[str]]:
    files = _iter_exclusion_files(paths, base_dir=base_dir)
    paragraph_hashes: set[str] = set()
    shingles: set[int] = set()
    loaded: list[str] = []
    import hashlib
    n = max(8, int(shingle_words))
    for path in files:
        try:
            text = _read_exclusion_text(path)
        except (OSError, UnicodeError):
            continue
        loaded.append(str(path))
        for paragraph in re.split(r"\n\s*\n", text):
            normalized = re.sub(r"\s+", " ", paragraph).strip().lower()
            if not normalized:
                continue
            paragraph_hashes.add(hashlib.sha256(normalized.encode("utf-8", errors="replace")).hexdigest())
            words = _normalized_words(normalized)
            for i in range(0, max(0, len(words) - n + 1), max(1, n // 2)):
                digest = hashlib.blake2b(" ".join(words[i:i+n]).encode("utf-8"), digest_size=8).hexdigest()
                shingles.add(int(digest, 16))
    return paragraph_hashes, shingles, loaded


def _paragraph_overlaps_training(paragraph: str, paragraph_hashes: set[str], shingles: set[int], *, shingle_words: int, match_fraction: float) -> bool:
    import hashlib
    normalized = re.sub(r"\s+", " ", paragraph).strip().lower()
    if not normalized:
        return False
    digest = hashlib.sha256(normalized.encode("utf-8", errors="replace")).hexdigest()
    if digest in paragraph_hashes:
        return True
    words = _normalized_words(normalized)
    n = max(8, int(shingle_words))
    if len(words) < n:
        return False
    total = max(1, len(words) - n + 1)
    matches = 0
    for i in range(0, total, max(1, n // 2)):
        value = int(hashlib.blake2b(" ".join(words[i:i+n]).encode("utf-8"), digest_size=8).hexdigest(), 16)
        if value in shingles:
            matches += 1
            if matches >= 2 and matches / max(1, (total + max(1, n//2) - 1)//max(1, n//2)) >= float(match_fraction):
                return True
    return False

def build_reference(path: Path, target_bytes: int = 80000, sources: list[tuple[str, str] | tuple[str, str, str]] | None = None, *, exclude_paths: list[str] | None = None, overlap_words: int = 16, overlap_match_fraction: float = 0.20) -> dict:
    using_default_sources = sources is None
    sources = DEFAULT_SOURCES if sources is None else sources
    if target_bytes < 50000:
        raise ValueError("target_bytes must be at least 50000")

    remaining = int(target_bytes)
    paragraph_records: list[dict] = []
    project_root = path.parent.parent if path.parent.name.lower() == "evaluation" else path.parent
    exclusion_paragraph_hashes, exclusion_shingles, loaded_exclusions = _build_overlap_index(
        exclude_paths, shingle_words=overlap_words, base_dir=project_root,
    )
    errors = []
    loaded_sources: list[str] = []
    loaded_domains: set[str] = set()
    per_source = max(6000, (target_bytes + len(sources) - 1) // len(sources) + 512)

    for index, source in enumerate(sources):
        if remaining <= 0:
            break
        if len(source) == 2:
            name, url = source
            domain = "general_prose"
        elif len(source) == 3:
            name, url, domain = source
        else:
            errors.append({"title": str(source[0]) if source else "unknown", "url": "", "error": "source must be a (title, URL) or (title, URL, domain) tuple"})
            continue
        try:
            raw = _fetch(url)
            cleaned = _clean_gutenberg(raw)
            piece = _sample_prose(cleaned, min(per_source, remaining), seed=index * 17 + 11)
        except Exception as exc:
            errors.append({"title": name, "url": url, "error": repr(exc)})
            continue
        if not piece:
            continue
        loaded_sources.append(name)
        loaded_domains.add(str(domain))
        paragraphs = [p.strip() for p in re.split(r"\n\s*\n", piece) if p.strip()]
        for paragraph_index, paragraph in enumerate(paragraphs):
            if _paragraph_overlaps_training(
                paragraph, exclusion_paragraph_hashes, exclusion_shingles,
                shingle_words=overlap_words, match_fraction=overlap_match_fraction,
            ):
                continue
            paragraph_records.append({
                "title": name,
                "url": url,
                "domain": str(domain),
                "index": paragraph_index,
                "text": paragraph,
            })

    minimum_sources = min(8, len(sources))
    minimum_domains = min(4, len({str(source[2]) if len(source) > 2 else "general_prose" for source in sources})) if using_default_sources else 1
    if len(loaded_sources) < minimum_sources:
        detail = "; ".join(
            f"{item['title']}: {item['error']}" for item in errors
        )
        raise RuntimeError(
            f"evaluation corpus loaded only {len(loaded_sources)}/{len(sources)} "
            f"sources; need at least {minimum_sources}"
            + (f"; failed sources: {detail}" if detail else "")
        )

    if not paragraph_records:
        raise RuntimeError("no evaluation sources could be loaded")
    if using_default_sources and len(loaded_domains) < minimum_domains:
        raise RuntimeError(
            f"evaluation corpus loaded only {len(loaded_domains)} distinct domains; need at least {minimum_domains}"
        )

    # Shuffle across sources before writing the fixed corpus. This prevents the
    # final 20% scored region from being dominated by the last source(s).
    import hashlib
    paragraph_records.sort(
        key=lambda row: int.from_bytes(
            hashlib.sha256(
                f"Aletheia-eval:{row['title']}:{row['index']}:{row['text']}".encode("utf-8")
            ).digest()[:8],
            "big",
        )
    )

    selected: list[str] = []
    selected_meta: list[dict] = []
    total = 0
    for row in paragraph_records:
        separator = "\n\n" if selected else ""
        piece = separator + row["text"]
        raw_piece = piece.encode("utf-8")
        if total + len(raw_piece) <= target_bytes:
            selected.append(row["text"])
            selected_meta.append(row)
            total += len(raw_piece)
            continue
        remaining_bytes = target_bytes - total
        separator_bytes = len(separator.encode("utf-8"))
        if remaining_bytes > separator_bytes + 16:
            available = remaining_bytes - separator_bytes
            raw_text = row["text"].encode("utf-8")
            clipped = raw_text[:available].decode("utf-8", errors="ignore")
            if clipped:
                selected.append(clipped)
                selected_meta.append({**row, "text": clipped, "clipped": True})
                total = len("\n\n".join(selected).encode("utf-8"))
        break

    combined = "\n\n".join(selected)
    raw = combined.encode("utf-8")

    # A fixed reference must be an auditable concatenation of the selected source
    # records. Do not append untracked bytes after the manifest has been assembled:
    # if the real selected corpus is too small, fail and request more source material.
    if len(raw) < 50000:
        detail = f"; failed_sources={len(errors)}" if errors else ""
        raise RuntimeError(
            f"evaluation corpus is only {len(raw)} bytes; need at least 50000" + detail
        )
    # Enforce diversity on the bytes that are actually written, not on every
    # source fetched. A source may load successfully but contribute nothing to the
    # selected fixed-length corpus.
    selected_titles = {str(row.get("title", "unknown")) for row in selected_meta}
    selected_domains = {str(row.get("domain", "general_prose")) for row in selected_meta}
    minimum_selected_sources = min(8, len(sources))
    minimum_selected_domains = min(
        4, len({str(source[2]) if len(source) > 2 else "general_prose" for source in sources})
    ) if using_default_sources else 1
    if len(selected_titles) < minimum_selected_sources:
        raise RuntimeError(
            f"selected evaluation corpus contains only {len(selected_titles)} sources; "
            f"need at least {minimum_selected_sources}"
        )
    if len(selected_domains) < minimum_selected_domains:
        raise RuntimeError(
            f"selected evaluation corpus contains only {len(selected_domains)} domains; "
            f"need at least {minimum_selected_domains}"
        )

    # Never pad with synthetic bytes. A corpus above the minimum may still be
    # shorter than the requested target when source material runs out.
    raw = raw[:target_bytes]

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(raw)

    # Persist source identity and exact selected text so evaluation can report
    # per-source/domain metrics. All reference sources are fixed-heldout; they are
    # not tokenizer-training or memory/replay sources.
    source_bytes: dict[str, int] = {}
    paragraph_hashes: list[dict] = []
    manifest_records: list[dict] = []
    for selected_index, paragraph in enumerate(selected):
        meta = selected_meta[selected_index] if selected_index < len(selected_meta) else {}
        normalized = re.sub(r"\s+", " ", paragraph).strip()
        digest = hashlib.sha256(paragraph.encode("utf-8")).hexdigest()
        normalized_digest = hashlib.sha256(normalized.encode("utf-8")).hexdigest()
        url = str(meta.get("url", ""))
        title = str(meta.get("title", "unknown"))
        source_id = hashlib.sha256(url.encode("utf-8")).hexdigest() if url else hashlib.sha256(title.encode("utf-8")).hexdigest()
        item_id = hashlib.sha256(f"{source_id}:{meta.get('index', selected_index)}:{digest}".encode("utf-8")).hexdigest()
        paragraph_record = {
            "item_id": item_id,
            "source_id": source_id,
            "title": title,
            "url": url,
            "domain": str(meta.get("domain", "general_prose")),
            "index": int(meta.get("index", selected_index)),
            "sha256": digest,
            "normalized_sha256": normalized_digest,
            "chars": len(paragraph),
            "bytes": len(paragraph.encode("utf-8")),
            "text": paragraph,
            "split": "fixed-heldout",
            "memory_replay_allowed": False,
            "tokenizer_training_allowed": False,
            "clipped": bool(meta.get("clipped", False)),
        }
        paragraph_hashes.append(paragraph_record)
        manifest_records.append(paragraph_record)
        source_bytes[title] = source_bytes.get(title, 0) + len(paragraph.encode("utf-8"))

    manifest_errors = validate_source_separated_manifest(manifest_records)
    if manifest_errors:
        raise RuntimeError("invalid evaluation source manifest: " + "; ".join(manifest_errors))
    retrieved_at = datetime.now(timezone.utc).isoformat()
    manifest = []
    for title, size in sorted(source_bytes.items()):
        source_rows = [row for row in selected_meta if row.get("title") == title]
        source_url = next((str(row.get("url", "")) for row in source_rows), "")
        source_text = "\n\n".join(str(row.get("text", "")) for row in source_rows)
        is_gutenberg = "gutenberg.org/" in source_url.lower()
        manifest.append({
            "title": title,
            "domain": next((str(row.get("domain", "general_prose")) for row in source_rows), "general_prose"),
            "bytes": size,
            "source_id": hashlib.sha256(source_url.encode("utf-8")).hexdigest() if source_url else hashlib.sha256(title.encode("utf-8")).hexdigest(),
            "selected_content_sha256": hashlib.sha256(source_text.encode("utf-8")).hexdigest(),
            "url": source_url,
            "retrieved_at_utc": retrieved_at,
            "provenance_note": "Project Gutenberg text; source/jurisdiction terms still require operator review" if is_gutenberg else "Caller-supplied source; license and provenance not independently verified",
            "split": "fixed-heldout",
            "memory_replay_allowed": False,
            "tokenizer_training_allowed": False,
        })
    manifest_path = path.with_suffix(".sources.json")
    manifest_path.write_text(json.dumps({"schema_version": 3, "sources": manifest, "errors": errors, "excluded_training_paths": loaded_exclusions, "retrieved_at_utc": retrieved_at, "selected_domains": sorted(selected_domains), "split_policy": "all source-separated reference documents are fixed-heldout; never train tokenizer or memory/replay on them"}, indent=2), encoding="utf-8")
    hashes_path = path.with_suffix(".paragraphs.json")
    hashes_path.write_text(json.dumps({"schema_version": 3, "dataset_split": "fixed-heldout", "paragraphs": paragraph_hashes, "manifest_validation_errors": manifest_errors}, indent=2, ensure_ascii=False), encoding="utf-8")
    return {
        "path": str(path),
        "bytes": len(raw[:target_bytes]),
        "sources": manifest,
        "domains": sorted({str(row.get("domain", "general_prose")) for row in selected_meta}),
        "errors": errors,
        "paragraph_hashes": str(hashes_path),
        "excluded_training_paths": loaded_exclusions,
    }
