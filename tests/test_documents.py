"""Tests for PDF / Word / EPUB / HTML ingestion. Every test file is built in the test itself,
using only the standard library (plus pypdf for the encrypted-PDF case)."""
import zipfile
from pathlib import Path

import pytest

from aletheia.curriculum import iter_text_files, read_text_file
from aletheia.documents import is_document, read_document


def _make_pdf(pages: list[list[str]], path: Path) -> Path:
    """Write a minimal valid PDF whose pages contain the given lines of text."""
    objects: list[bytes] = []

    def add(body: bytes) -> int:
        objects.append(body)
        return len(objects)

    font = add(b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>")
    page_ids = []
    pages_id_placeholder = len(objects) + 1 + 2 * len(pages)  # id the /Pages object will get
    for lines in pages:
        ops = ["BT", "/F1 12 Tf", "72 720 Td", "14 TL"]
        for line in lines:
            escaped = line.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
            ops.append(f"({escaped}) Tj T*")
        ops.append("ET")
        stream = "\n".join(ops).encode("latin-1")
        content = add(b"<< /Length %d >>\nstream\n" % len(stream) + stream + b"\nendstream")
        page_ids.append(
            add(
                f"<< /Type /Page /Parent {pages_id_placeholder} 0 R /MediaBox [0 0 612 792] "
                f"/Resources << /Font << /F1 {font} 0 R >> >> /Contents {content} 0 R >>".encode()
            )
        )
    kids = " ".join(f"{i} 0 R" for i in page_ids)
    pages_id = add(f"<< /Type /Pages /Kids [{kids}] /Count {len(page_ids)} >>".encode())
    assert pages_id == pages_id_placeholder
    catalog = add(f"<< /Type /Catalog /Pages {pages_id} 0 R >>".encode())

    out = bytearray(b"%PDF-1.4\n")
    offsets = []
    for index, body in enumerate(objects, start=1):
        offsets.append(len(out))
        out += f"{index} 0 obj\n".encode() + body + b"\nendobj\n"
    xref = len(out)
    out += f"xref\n0 {len(objects) + 1}\n".encode() + b"0000000000 65535 f \n"
    for offset in offsets:
        out += f"{offset:010d} 00000 n \n".encode()
    out += f"trailer\n<< /Size {len(objects) + 1} /Root {catalog} 0 R >>\nstartxref\n{xref}\n%%EOF\n".encode()
    path.write_bytes(bytes(out))
    return path


def test_pdf_text_is_extracted_and_running_headers_and_page_numbers_are_removed(tmp_path):
    topics = ["rivers", "mountains", "forests", "deserts", "oceans", "glaciers"]
    pages = []
    for number, topic in enumerate(topics, start=1):
        pages.append([
            "ACME Annual Report",
            f"This page discusses {topic} in considerable detail.",
            "The next sentence is wrapped with a hyphen-",
            "ated word and continues on a second line.",
            f"A closing remark that only concerns {topic} and nothing else.",
            f"{number}",
        ])
    path = _make_pdf(pages, tmp_path / "report.pdf")
    text = read_document(path)
    assert "This page discusses rivers in considerable detail." in text
    assert "hyphenated word and continues on a second line" in text   # end-of-line hyphenation undone
    assert "ACME Annual Report" not in text                           # header repeated on every page
    assert "\n6\n" not in "\n" + text + "\n"                          # bare page numbers dropped


def test_body_text_that_repeats_across_pages_is_not_mistaken_for_a_header(tmp_path):
    pages = [
        ["Running Title", "Line one of the body.", "Line two of the body.", "Line three of the body.",
         "Line four of the body.", "Line five of the body.", "Footer text"]
        for _ in range(6)
    ]
    text = read_document(_make_pdf(pages, tmp_path / "forms.pdf"))
    assert "Running Title" not in text and "Footer text" not in text      # edge lines that repeat
    for middle in ("Line two of the body.", "Line three of the body.", "Line four of the body."):
        assert text.count(middle) == 6                                    # body lines are all kept


def test_short_pdf_keeps_its_text(tmp_path):
    path = _make_pdf([["A short document about rivers and the weather over several seasons."]], tmp_path / "a.pdf")
    assert "rivers and the weather" in read_document(path)


def test_image_only_pdf_is_reported_not_silently_taught(tmp_path):
    path = _make_pdf([[]], tmp_path / "scan.pdf")
    with pytest.raises(ValueError, match="no extractable text"):
        read_document(path)


def test_corrupt_pdf_is_reported(tmp_path):
    path = tmp_path / "broken.pdf"
    path.write_bytes(b"%PDF-1.4\nthis is not really a pdf")
    with pytest.raises(ValueError):
        read_document(path)


def test_encrypted_pdf_needs_a_password(tmp_path):
    pypdf = pytest.importorskip("pypdf")
    source = _make_pdf([["Confidential text about rivers and the weather over many seasons."]], tmp_path / "plain.pdf")
    writer = pypdf.PdfWriter(clone_from=str(source))
    try:
        writer.encrypt("secret", algorithm="RC4-128")
    except Exception:
        pytest.skip("this pypdf install cannot write encrypted files")
    locked = tmp_path / "locked.pdf"
    with open(locked, "wb") as handle:
        writer.write(handle)
    with pytest.raises(ValueError, match="password"):
        read_document(locked)


def _make_docx(paragraphs: list[str], path: Path) -> Path:
    body = "".join(f"<w:p><w:r><w:t>{p}</w:t></w:r></w:p>" for p in paragraphs)
    xml = (
        '<?xml version="1.0" encoding="UTF-8"?>'
        '<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">'
        f"<w:body>{body}</w:body></w:document>"
    )
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("word/document.xml", xml)
    return path


def test_docx_paragraphs_are_extracted(tmp_path):
    path = _make_docx(["First paragraph about science and travel.", "Second paragraph about weather."], tmp_path / "a.docx")
    text = read_document(path)
    assert text.split("\n\n") == ["First paragraph about science and travel.", "Second paragraph about weather."]


def test_broken_docx_is_reported(tmp_path):
    path = tmp_path / "bad.docx"
    path.write_bytes(b"not a zip file")
    with pytest.raises(ValueError):
        read_document(path)


def test_html_keeps_visible_text_and_drops_scripts_and_navigation(tmp_path):
    path = tmp_path / "page.html"
    path.write_text(
        "<html><head><style>p{color:red}</style><script>var secret=1;</script></head><body>"
        "<nav><p>Home | About | Contact</p></nav>"
        "<h1>Rivers</h1><p>A river is a <em>large</em> natural stream of fresh water.</p>"
        "<ul><li>It flows to the sea.</li></ul></body></html>",
        encoding="utf-8",
    )
    text = read_document(path)
    assert "A river is a large natural stream of fresh water." in text
    assert "It flows to the sea." in text
    assert "secret" not in text and "color:red" not in text and "Home | About" not in text


def test_epub_chapters_follow_the_reading_order(tmp_path):
    path = tmp_path / "book.epub"
    chapter = lambda title, body: f"<html><body><h1>{title}</h1><p>{body}</p></body></html>"
    container = (
        '<?xml version="1.0"?><container xmlns="urn:oasis:names:tc:opendocument:xmlns:container">'
        '<rootfiles><rootfile full-path="OEBPS/content.opf" media-type="application/oebps-package+xml"/></rootfiles></container>'
    )
    opf = (
        '<?xml version="1.0"?><package xmlns="http://www.idpf.org/2007/opf" version="3.0"><manifest>'
        '<item id="a" href="b.xhtml" media-type="application/xhtml+xml"/>'
        '<item id="b" href="a.xhtml" media-type="application/xhtml+xml"/></manifest>'
        '<spine><itemref idref="a"/><itemref idref="b"/></spine></package>'
    )
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("META-INF/container.xml", container)
        archive.writestr("OEBPS/content.opf", opf)
        archive.writestr("OEBPS/a.xhtml", chapter("Second", "This chapter is listed second in the reading order."))
        archive.writestr("OEBPS/b.xhtml", chapter("First", "This chapter is listed first in the reading order."))
    text = read_document(path)
    assert text.index("listed first") < text.index("listed second")



def test_numeric_body_lines_are_kept_when_not_at_page_edge(tmp_path):
    pages = [
        [
            "Running Title",
            "The numbered section below is part of the body text.",
            "1",
            "This sentence explains the numbered item in context.",
            "Footer text",
        ]
        for _ in range(6)
    ]
    text = read_document(_make_pdf(pages, tmp_path / "numbered.pdf"))
    assert text.split().count("1") == 6


def test_directory_skip_callback_exposes_document_reason(tmp_path):
    root = tmp_path / "curriculum"
    root.mkdir()
    _make_pdf([[]], root / "scan.pdf")
    skipped = []
    lessons = list(iter_text_files(root, on_skip=lambda path, reason: skipped.append((path.name, reason))))
    assert lessons == []
    assert skipped and skipped[0][0] == "scan.pdf"
    assert "OCR" in skipped[0][1]

def test_plain_text_still_goes_through_the_old_reader(tmp_path):
    path = tmp_path / "notes.txt"
    path.write_text("Plain text lesson about rivers.", encoding="utf-8")
    assert not is_document(path)
    assert read_text_file(path).text == "Plain text lesson about rivers."


def test_directory_walk_reads_mixed_formats_and_reports_what_it_skipped(tmp_path):
    root = tmp_path / "curriculum"
    root.mkdir()
    (root / "a.txt").write_text("A plain lesson about rivers and weather.", encoding="utf-8")
    _make_pdf([["A pdf lesson about measurement and travel across many seasons."]], root / "b.pdf")
    _make_docx(["A Word lesson about science and communication."], root / "c.docx")
    _make_pdf([[]], root / "scan.pdf")
    (root / "photo.png").write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 200)
    skipped = []
    lessons = list(iter_text_files(root, on_skip=lambda path, reason: skipped.append((path.name, reason))))
    assert {lesson.title for lesson in lessons} == {"a.txt", "b.pdf", "c.docx"}
    assert {name for name, _ in skipped} == {"scan.pdf", "photo.png"}
    assert any("OCR" in reason for name, reason in skipped if name == "scan.pdf")
