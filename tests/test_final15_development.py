from __future__ import annotations

import io
import types
from pathlib import Path

import numpy as np
import pytest
import torch
from requests import Response

from aletheia.agent import AletheiaAgent
from aletheia.config import WebConfig, load_config
from aletheia.curriculum import iter_parquet_lessons
from aletheia.memory import ExperienceMemory
from aletheia.perception import PerceptionSystem
from aletheia.web import Page, WebExplorer


def _tiny_config():
    cfg = load_config(Path("config.json"))
    cfg.runtime.device = "cpu"
    cfg.runtime.memory_limit = 40
    cfg.model.dim = 32
    cfg.model.layers = 1
    cfg.model.heads = 4
    cfg.model.ffn = 64
    cfg.model.recurrent_dim = 48
    cfg.model.context = 64
    cfg.model.dropout = 0.0
    cfg.runtime.train_batch_size = 1
    cfg.runtime.language_ready_min_tokens = 0
    cfg.runtime.language_ready_min_bpb_gain = -100.0
    cfg.runtime.language_ready_min_eval_bytes = 0
    cfg.runtime.language_ready_min_lessons = 1
    return cfg


def test_unicode_memory_search_falls_back_when_fts_has_no_hit(tmp_path):
    memory = ExperienceMemory(tmp_path / "memory.sqlite3", 20, 20)
    try:
        memory.add(kind="lesson", source="book", title="Hindi", url="", text="हिन्दी भाषा में संवाद और अनुभव")
        rows = memory.search("हिन्दी भाषा", 10)
        assert rows
        assert "हिन्दी" in rows[0]["text"]
    finally:
        memory.close()


def test_crawl_attempt_limit_tracks_failures_and_depth(monkeypatch):
    web = WebExplorer(WebConfig([], [], ["http", "https"]), 1, 1000, "test", 0, 0)
    pages = {
        "https://example.org/": Page("https://example.org/", "root", "root", 200, ["https://example.org/a", "https://example.org/b"]),
        "https://example.org/b": Page("https://example.org/b", "b", "b", 200, []),
    }
    attempts = []

    def fake_open(url):
        attempts.append(url)
        if url.endswith("/a"):
            raise RuntimeError("bad page")
        return pages[url]

    monkeypatch.setattr(web, "open", fake_open)
    try:
        result = web.crawl_same_domain("https://example.org/", max_pages=2, max_depth=2, max_attempts=3)
        assert [page.url for page in result] == ["https://example.org/", "https://example.org/b"]
        assert result[-1].depth == 1
        assert len(attempts) == 3
    finally:
        web.close()


def test_web_plain_text_without_charset_prefers_utf8():
    web = WebExplorer(WebConfig([], [], ["http", "https"]), 1, 1000, "test", 0, 0)
    response = Response()
    response.status_code = 200
    response.headers["Content-Type"] = "text/plain"
    response._content = "café हिन्दी".encode("utf-8")
    try:
        assert web._response_text(response, "text/plain") == "café हिन्दी"
    finally:
        web.close()


def test_web_download_limit_rejects_large_response(monkeypatch):
    web = WebExplorer(WebConfig([], [], ["http", "https"]), 1, 1000, "test", 0, 0, max_download_bytes=8)
    response = Response()
    response.status_code = 200
    response.url = "https://example.org/big"
    response.headers["Content-Length"] = "9"
    response.raw = io.BytesIO(b"012345678")
    monkeypatch.setattr(web._dns_adapter, "_resolve_public", lambda host: "93.184.216.34")
    monkeypatch.setattr(web, "_robots", lambda url: True)
    monkeypatch.setattr(web.session, "get", lambda *args, **kwargs: response)
    try:
        with pytest.raises(ValueError, match="download limit"):
            web._get("https://example.org/big")
    finally:
        web.close()


def test_parquet_iterator_honors_global_byte_budget_across_rows(monkeypatch, tmp_path):
    class FakeColumn:
        def to_pylist(self):
            return ["aaaa", "bbbb", "cccc", "dddd"]

    class FakeBatch:
        def column(self, index):
            assert index == 0
            return FakeColumn()

    class FakeSchema:
        names = ["text"]
        def field(self, name):
            return types.SimpleNamespace(type="string")

    class FakeParquet:
        schema_arrow = FakeSchema()
        def iter_batches(self, **kwargs):
            yield FakeBatch()

    fake_pq = types.SimpleNamespace(ParquetFile=lambda path: FakeParquet())
    monkeypatch.setitem(__import__("sys").modules, "pyarrow.parquet", fake_pq)
    monkeypatch.setitem(__import__("sys").modules, "pyarrow", types.SimpleNamespace(parquet=fake_pq))
    path = tmp_path / "dataset.parquet"
    path.write_bytes(b"stub")
    lessons = list(iter_parquet_lessons(path, shard_max_chars=20, max_source_bytes=9))
    total = sum(len(item.text.encode("utf-8")) for item in lessons)
    assert total <= 9
    assert total > 0


def test_identity_reindex_uses_stored_exemplars(tmp_path):
    system = PerceptionSystem(
        tmp_path, torch.device("cpu"), embedding_dim=8, batch_size=2, buffer_size=4,
        identity_max_exemplars=2, identity_reindex_interval=1,
    )
    image = np.zeros((64, 64, 3), dtype=np.uint8)
    image[..., 0] = 100
    first = system.observe_image(image, train=False)
    old = np.asarray(system.identities.entities[0]["centroid"], dtype=np.float32)
    system.identities.add_exemplar(first["entity_id"], image)
    system.vision.eval()
    system.identities.refresh(system.vision, torch.device("cpu"))
    new = np.asarray(system.identities.entities[0]["centroid"], dtype=np.float32)
    assert np.isfinite(new).all()
    assert old.shape == new.shape


def test_pretrain_uses_small_memory_metadata_not_full_shard_text(tmp_path, monkeypatch):
    import aletheia.agent as agent_module

    cfg = _tiny_config()
    cfg.runtime.pretrain_eval_every_shards = 999
    cfg.runtime.pretrain_epochs = 1
    agent = AletheiaAgent(tmp_path, cfg)
    lesson = agent_module.Lesson(title="big", source="synthetic", text="language " * 4000)
    monkeypatch.setattr(agent, "_parquet_files", lambda path: [])
    monkeypatch.setattr(agent, "_load_pretrain_shards", lambda path: ([lesson], {"cleaning": {}, "skipped": []}))
    monkeypatch.setattr(agent, "_encode_chunks", lambda text, max_chunks=0, context=None: [torch.zeros((1, 8), dtype=torch.long)] * 2)
    monkeypatch.setattr(agent, "_chunk_bpb", lambda chunks: 2.0)
    monkeypatch.setattr(agent, "_train_sequence_collection", lambda *args, **kwargs: 1.0)
    monkeypatch.setattr(agent, "evaluate_fixed_reference", lambda **kwargs: {"model_bpb": 2.0, "baseline_bpb": 3.0, "gain": 1.0, "reference_bytes": 80000})
    try:
        agent.pretrain(tmp_path, epochs=1, seed=1)
        row = agent.memory.recent_lessons(1, validated_only=False)[0]
        assert len(row["text"]) < 500
    finally:
        agent.close()


def test_companion_does_not_fake_speech_before_language_ready(tmp_path):
    cfg = _tiny_config()
    cfg.runtime.language_ready_min_tokens = 10**9
    agent = AletheiaAgent(tmp_path, cfg)
    try:
        result = agent.initiate_communication("visual novelty=0.2")
        assert result["initiated"] is False
        assert result["reason"] == "language_not_ready"
    finally:
        agent.close()


def test_speech_bridge_is_random_init_and_decodes_without_pretrained_weights(tmp_path):
    from aletheia.speech import ByteCTCSpeechModel, SpeechRecognizer
    model = ByteCTCSpeechModel(hidden=32)
    params = sum(p.numel() for p in model.parameters())
    assert params > 0
    recognizer = SpeechRecognizer(tmp_path, torch.device("cpu"), ready_min_steps=1)
    waveform = torch.randn(16000) * 0.01
    text = recognizer.transcribe(waveform)
    assert isinstance(text, str)
    assert recognizer.ready is False


def test_speech_manifest_reads_jsonl_relative_audio(tmp_path):
    import json
    from aletheia.speech import SpeechRecognizer
    manifest = tmp_path / "speech.jsonl"
    manifest.write_text(json.dumps({"audio": "sample.wav", "text": "hello"}) + "\n", encoding="utf-8")
    examples = SpeechRecognizer.read_manifest(manifest)
    assert len(examples) == 1
    assert examples[0].audio == (tmp_path / "sample.wav").resolve()
    assert examples[0].text == "hello"


def test_parquet_schedule_respects_global_byte_cap(tmp_path):
    cfg = _tiny_config()
    cfg.runtime.pretrain_max_source_bytes = 100
    from aletheia.agent import AletheiaAgent
    agent = AletheiaAgent(tmp_path, cfg)
    try:
        agent._parquet_text_uncompressed_bytes = lambda paths: 1000
        uncapped = agent._estimate_parquet_microbatches([tmp_path / "a.parquet"], 64, 4, 1, respect_cap=False)
        capped = agent._estimate_parquet_microbatches([tmp_path / "a.parquet"], 64, 4, 1, respect_cap=True)
        assert capped < uncapped
    finally:
        agent.close()


def test_web_crawl_supports_configurable_path_exclusions(monkeypatch):
    web = WebExplorer(WebConfig([], [], ["http", "https"]), 1, 1000, "test", 0, 0, crawl_exclude_patterns=[r"/login(?:$|/)"])
    pages = {
        "https://example.org/": Page("https://example.org/", "root", "root", 200, ["https://example.org/login", "https://example.org/content"]),
        "https://example.org/content": Page("https://example.org/content", "content", "ok", 200, []),
    }
    monkeypatch.setattr(web, "open", lambda url: pages[url])
    try:
        result = web.crawl_same_domain("https://example.org/", max_pages=3, max_depth=1, max_attempts=3)
        assert [page.url for page in result] == ["https://example.org/", "https://example.org/content"]
    finally:
        web.close()
