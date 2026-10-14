"""Document readers: turn PDF, Word, EPUB and HTML files into plain training text.

These are file-format parsers (they extract the characters that are stored in a file).
They contain no language model, no vocabulary and no rules about meaning. What Aletheia
learns from the extracted text is still decided only by its own training.

Only the PDF reader needs an extra package (pypdf, a pure-Python PDF parser). Word,
EPUB and HTML use the standard library plus beautifulsoup4, which Aletheia already needs.
Scanned (image-only) PDFs contain no text to extract; they need OCR, which is not done here.
"""
from __future__ import annotations

import posixpath
import re
import xml.etree.ElementTree as ET
import zipfile
from io import BytesIO
from collections import Counter
from pathlib import Path
from urllib.parse import unquote

DOCUMENT_SUFFIXES = {".pdf", ".docx", ".epub", ".html", ".htm", ".xhtml"}

_MAX_ZIP_MEMBER_BYTES = 200 * 1024 * 1024

_W = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"
_BLOCK_TAGS = ["p", "h1", "h2", "h3", "h4", "h5", "h6", "li", "blockquote", "pre", "tr", "dd", "dt", "figcaption"]
_DROP_TAGS = ["script", "style", "noscript", "nav", "header", "footer", "aside", "form", "template"]


def is_document(path: Path) -> bool:
    return Path(path).suffix.lower() in DOCUMENT_SUFFIXES


def read_document(path: Path) -> str:
    """Return the plain text of a supported document, or raise ValueError with a reason."""
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        text = _read_pdf(path)
    elif suffix == ".docx":
        text = _read_docx(path)
    elif suffix == ".epub":
        text = _read_epub(path)
    elif suffix in {".html", ".htm", ".xhtml"}:
        text = _html_to_text(_decode(path.read_bytes()))
    else:
        raise ValueError(f"unsupported document type: {path.name}")
    text = text.replace("\x00", " ").strip()
    if not text.strip():
        raise ValueError(f"no extractable text in {path.name}")
    return text


def _decode(raw: bytes) -> str:
    """Decode text without guessing UTF-16 for arbitrary even-length byte strings."""
    if raw.startswith((b"\xff\xfe\x00\x00", b"\x00\x00\xfe\xff")):
        try:
            return raw.decode("utf-32")
        except UnicodeDecodeError:
            pass
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


def read_document_bytes(raw: bytes, *, name: str = "download", content_type: str = "") -> str:
    """Extract text from downloaded document bytes using the same readers as local files."""
    suffix = Path(name.split("?", 1)[0]).suffix.lower()
    ctype = (content_type or "").lower()
    if suffix == ".pdf" or "application/pdf" in ctype:
        return _read_pdf_bytes(raw, name)
    if suffix in {".html", ".htm", ".xhtml"} or "html" in ctype:
        return _html_to_text(_decode(raw))
    if "text/plain" in ctype:
        return _decode(raw)
    if suffix == ".docx":
        tmp = BytesIO(raw)
        try:
            with zipfile.ZipFile(tmp) as archive:
                root = ET.fromstring(_zip_member(archive, "word/document.xml"))
        except (zipfile.BadZipFile, KeyError, ET.ParseError) as exc:
            raise ValueError(f"unreadable Word document {name}: {exc}") from exc
        paragraphs = []
        for paragraph in root.iter(f"{_W}p"):
            pieces = []
            for node in paragraph.iter():
                if node.tag == f"{_W}t" and node.text:
                    pieces.append(node.text)
                elif node.tag == f"{_W}tab":
                    pieces.append("\t")
                elif node.tag in (f"{_W}br", f"{_W}cr"):
                    pieces.append("\n")
            text = "".join(pieces).strip()
            if text:
                paragraphs.append(text)
        return "\n\n".join(paragraphs)
    if suffix == ".epub":
        tmp = BytesIO(raw)
        try:
            with zipfile.ZipFile(tmp) as archive:
                names = set(archive.namelist())
                order = _epub_reading_order(archive, names)
                parts = []
                for member in order:
                    try:
                        part = _html_to_text(_decode(_zip_member(archive, member)))
                    except KeyError:
                        continue
                    if part.strip():
                        parts.append(part)
        except (zipfile.BadZipFile, ET.ParseError) as exc:
            raise ValueError(f"unreadable EPUB {name}: {exc}") from exc
        return "\n\n".join(parts)
    return _decode(raw)


# ----------------------------------------------------------------------------- PDF
def _read_pdf(path: Path) -> str:
    try:
        from pypdf import PdfReader
    except ImportError as exc:
        raise ValueError("PDF support needs the 'pypdf' package: pip install pypdf") from exc
    try:
        reader = PdfReader(str(path), strict=False)
    except Exception as exc:  # corrupt or truncated file
        raise ValueError(f"unreadable PDF {path.name}: {exc}") from exc
    if reader.is_encrypted:
        try:
            unlocked = reader.decrypt("")
        except Exception:
            unlocked = 0
        if not unlocked:
            raise ValueError(f"encrypted PDF skipped (password required): {path.name}")
    pages: list[str] = []
    for page in reader.pages:
        try:
            pages.append(page.extract_text() or "")
        except Exception:
            pages.append("")
    text = _clean_pdf_pages(pages)
    if not text.strip():
        raise ValueError(
            f"no extractable text in {path.name} (scanned or image-only PDF? it needs OCR first)"
        )
    return text


def _read_pdf_bytes(raw: bytes, name: str) -> str:
    try:
        from pypdf import PdfReader
    except ImportError as exc:
        raise ValueError("PDF support needs the 'pypdf' package: pip install pypdf") from exc
    try:
        reader = PdfReader(BytesIO(raw), strict=False)
    except Exception as exc:
        raise ValueError(f"unreadable PDF {name}: {exc}") from exc
    if reader.is_encrypted:
        try:
            unlocked = reader.decrypt("")
        except Exception:
            unlocked = 0
        if not unlocked:
            raise ValueError(f"encrypted PDF skipped (password required): {name}")
    pages = []
    for page in reader.pages:
        try:
            pages.append(page.extract_text() or "")
        except Exception:
            pages.append("")
    text = _clean_pdf_pages(pages)
    if not text.strip():
        raise ValueError(f"no extractable text in {name} (scanned or image-only PDF? it needs OCR first)")
    return text


def _clean_pdf_pages(pages: list[str]) -> str:
    """Remove text that repeats on most pages (running headers/footers) and page numbers,
    undo end-of-line hyphenation, and join wrapped lines into paragraphs."""
    page_lines = [[line.strip() for line in page.splitlines()] for page in pages]

    def edge_positions(lines: list[str]) -> set[int]:
        """Indexes of the first two and last two non-empty lines: where headers/footers sit."""
        filled = [i for i, line in enumerate(lines) if line]
        return set(filled[:2]) | set(filled[-2:])

    repeated: set[str] = set()
    if len(pages) >= 5:
        seen = Counter()
        for lines in page_lines:
            keys = {
                re.sub(r"\d+", "#", lines[i])
                for i in edge_positions(lines)
                if len(lines[i]) <= 100
            }
            for key in keys:
                seen[key] += 1
        repeated = {key for key, count in seen.items() if count / len(pages) >= 0.6}

    kept_pages: list[str] = []
    for lines in page_lines:
        edges = edge_positions(lines)
        kept = []
        for index, line in enumerate(lines):
            if line and index in edges and re.sub(r"\d+", "#", line) in repeated:
                continue
            if index in edges and line.isdigit() and len(line) <= 6:
                continue
            kept.append(line)
        kept_pages.append("\n".join(kept))

    text = "\n".join(kept_pages)
    text = re.sub(r"([a-z])-\n([a-z])", r"\1\2", text)
    paragraphs = []
    for block in re.split(r"\n\s*\n", text):
        flowed = " ".join(part for part in (line.strip() for line in block.splitlines()) if part)
        flowed = re.sub(r"[ \t]+", " ", flowed)
        if flowed:
            paragraphs.append(flowed)
    return "\n\n".join(paragraphs)


# ---------------------------------------------------------------------------- DOCX
def _zip_member(archive: zipfile.ZipFile, name: str) -> bytes:
    info = archive.getinfo(name)
    if info.file_size > _MAX_ZIP_MEMBER_BYTES:
        raise ValueError(f"archive member too large: {name}")
    return archive.read(name)


def _read_docx(path: Path) -> str:
    try:
        with zipfile.ZipFile(path) as archive:
            root = ET.fromstring(_zip_member(archive, "word/document.xml"))
    except (zipfile.BadZipFile, KeyError, ET.ParseError) as exc:
        raise ValueError(f"unreadable Word document {path.name}: {exc}") from exc
    paragraphs = []
    for paragraph in root.iter(f"{_W}p"):
        pieces = []
        for node in paragraph.iter():
            if node.tag == f"{_W}t" and node.text:
                pieces.append(node.text)
            elif node.tag == f"{_W}tab":
                pieces.append("\t")
            elif node.tag in (f"{_W}br", f"{_W}cr"):
                pieces.append("\n")
        text = "".join(pieces).strip()
        if text:
            paragraphs.append(text)
    return "\n\n".join(paragraphs)


# ----------------------------------------------------------------------------- HTML
def _html_to_text(markup: str) -> str:
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(markup, "html.parser")
    for tag in soup(_DROP_TAGS):
        tag.decompose()
    paragraphs = []
    for element in soup.find_all(_BLOCK_TAGS):
        if element.find(_BLOCK_TAGS):  # only leaf blocks, so text is not repeated
            continue
        text = " ".join(element.get_text(" ", strip=True).split())
        if text:
            paragraphs.append(text)
    if not paragraphs:
        text = "\n".join(line.strip() for line in soup.get_text("\n").splitlines() if line.strip())
        return text
    return "\n\n".join(paragraphs)


# ----------------------------------------------------------------------------- EPUB
def _read_epub(path: Path) -> str:
    try:
        with zipfile.ZipFile(path) as archive:
            names = set(archive.namelist())
            order = _epub_reading_order(archive, names)
            parts = []
            for name in order:
                try:
                    text = _html_to_text(_decode(_zip_member(archive, name)))
                except KeyError:
                    continue
                if text.strip():
                    parts.append(text)
    except (zipfile.BadZipFile, ET.ParseError) as exc:
        raise ValueError(f"unreadable EPUB {path.name}: {exc}") from exc
    return "\n\n".join(parts)


def _epub_reading_order(archive: zipfile.ZipFile, names: set[str]) -> list[str]:
    try:
        container = ET.fromstring(_zip_member(archive, "META-INF/container.xml"))
        rootfile = next(el for el in container.iter() if el.tag.endswith("rootfile"))
        opf_path = rootfile.attrib["full-path"]
        opf = ET.fromstring(_zip_member(archive, opf_path))
        base = posixpath.dirname(opf_path)
        manifest = {
            el.attrib["id"]: el.attrib["href"]
            for el in opf.iter()
            if el.tag.endswith("}item") or el.tag == "item"
            if "id" in el.attrib and "href" in el.attrib
        }
        order = []
        for el in opf.iter():
            if (el.tag.endswith("}itemref") or el.tag == "itemref") and el.attrib.get("idref") in manifest:
                href = unquote(manifest[el.attrib["idref"]]).split("#")[0]
                member = posixpath.normpath(posixpath.join(base, href))
                if member in names and member.lower().endswith((".xhtml", ".html", ".htm")):
                    order.append(member)
        if order:
            return order
    except (KeyError, StopIteration, ET.ParseError):
        pass
    return sorted(n for n in names if n.lower().endswith((".xhtml", ".html", ".htm")))
