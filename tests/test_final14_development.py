import io
import sys
import types
import zipfile
from pathlib import Path

import numpy as np
import pytest
from requests import Response

from aletheia.config import WebConfig, load_config
from aletheia.curriculum import Lesson, iter_parquet_lessons
from aletheia.documents import _decode
from aletheia.memory import ExperienceMemory, lexical_terms
from aletheia.web import Page, WebExplorer


def _small_config(tmp_path):
    cfg = load_config(Path("config.json"))
    cfg.runtime.device = "cpu"
    cfg.runtime.memory_limit = 3
    cfg.runtime.web_crawl_max_pages = 3
    cfg.runtime.web_crawl_max_depth = 1
    cfg.model.dim = 32
    cfg.model.layers = 1
    cfg.model.heads = 4
    cfg.model.ffn = 64
    cfg.model.recurrent_dim = 48
    cfg.model.context = 64
    cfg.model.dropout = 0.0
    return cfg


def test_utf8_even_length_is_not_misdecoded_as_utf16():
    text = "é漢" * 200
    raw = text.encode("utf-8")
    assert len(raw) % 2 == 0
    assert _decode(raw) == text


def test_unicode_lexical_terms_include_hindi_and_accented_words():
    terms = lexical_terms("हिन्दी भाषा café München world")
    assert "हिन्दी" in terms
    assert "भाषा" in terms
    assert "café" in terms
    assert "münchen" in terms


def test_memory_eviction_never_removes_lessons(tmp_path):
    memory = ExperienceMemory(tmp_path / "memory.sqlite3", limit=3, frontier_limit=10)
    try:
        lesson_id = memory.add(kind="lesson", source="book", title="important", url="", text="lesson that must survive")
        for i in range(8):
            memory.add(kind="external", source="web", title=f"obs{i}", url=str(i), text=f"observation {i}")
        lessons = memory.recent_lessons(10, validated_only=False)
        assert any(int(row["id"]) == lesson_id for row in lessons)
    finally:
        memory.close()


def test_same_domain_crawl_is_bounded_and_does_not_leave_domain(monkeypatch):
    web = WebExplorer(WebConfig([], [], ["http", "https"]), 1, 1000, "test", 0, 0)
    pages = {
        "https://example.org/": Page("https://example.org/", "root", "root", 200, ["https://example.org/a", "https://other.example/x"]),
        "https://example.org/a": Page("https://example.org/a", "a", "a", 200, ["https://example.org/b"]),
        "https://example.org/b": Page("https://example.org/b", "b", "b", 200, []),
    }
    monkeypatch.setattr(web, "open", lambda url: pages[url])
    try:
        result = web.crawl_same_domain("https://example.org/", max_pages=2, max_depth=3)
        assert [page.url for page in result] == ["https://example.org/", "https://example.org/a"]
    finally:
        web.close()


def test_http_html_uses_declared_meta_charset(monkeypatch):
    web = WebExplorer(WebConfig([], [], ["http", "https"]), 1, 1000, "test", 0, 0)
    response = Response()
    response.status_code = 200
    response.headers["Content-Type"] = "text/html"
    response._content = '<meta charset="utf-8"><p>é漢字</p>'.encode("utf-8")
    text = web._response_text(response, "text/html")
    assert "é漢字" in text
    web.close()


def test_remote_pdf_is_accepted_by_open(monkeypatch, tmp_path):
    pypdf = pytest.importorskip("pypdf")
    writer = pypdf.PdfWriter()
    writer.add_blank_page(width=200, height=200)
    buf = io.BytesIO()
    writer.write(buf)
    response = Response()
    response.status_code = 200
    response.url = "https://example.org/manual.pdf"
    response.headers["Content-Type"] = "application/pdf"
    response._content = buf.getvalue()
    web = WebExplorer(WebConfig([], [], ["http", "https"]), 1, 1000, "test", 0, 0)
    monkeypatch.setattr(web, "_get", lambda url: response)
    monkeypatch.setattr(web, "_robots", lambda url: True)
    monkeypatch.setattr(web, "_host_is_public", lambda host: True)
    try:
        # Blank PDF has no extractable text, which proves the request reached the PDF reader path.
        with pytest.raises(ValueError, match="no extractable text"):
            web.open("https://example.org/manual.pdf")
    finally:
        web.close()


def test_parquet_streaming_uses_text_column_without_materializing_all_rows(monkeypatch, tmp_path):
    # This test uses a tiny fake pyarrow module so the repository test suite can validate
    # the streaming contract even in offline CI where pyarrow itself is not installed.
    class FakeColumn:
        def to_pylist(self):
            return ["first row", "second row", None, "third row"]

    class FakeBatch:
        def column(self, index):
            assert index == 0
            return FakeColumn()

    class FakeParquet:
        schema_arrow = types.SimpleNamespace(
            names=["id", "text"],
            field=lambda self, name: types.SimpleNamespace(type="string"),
        )

        def iter_batches(self, **kwargs):
            assert kwargs["columns"] == ["text"]
            yield FakeBatch()

    fake_pq = types.SimpleNamespace(ParquetFile=lambda path: FakeParquet())
    fake_pyarrow = types.SimpleNamespace(parquet=fake_pq)
    monkeypatch.setitem(sys.modules, "pyarrow", fake_pyarrow)
    monkeypatch.setitem(sys.modules, "pyarrow.parquet", fake_pq)

    path = tmp_path / "data.parquet"
    path.write_bytes(b"not really parquet")
    lessons = list(iter_parquet_lessons(path, shard_max_chars=20, text_column="text"))
    combined = "\n".join(item.text for item in lessons)
    assert "first row" in combined
    assert "third row" in combined
