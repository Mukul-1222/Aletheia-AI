import json
import sqlite3
import wave
from pathlib import Path

import numpy as np
import torch

from aletheia.memory import ExperienceMemory
from aletheia.perception import PerceptualIdentityMemory
from aletheia.speech import SpeechRecognizer
from aletheia.web import WebExplorer


def test_memory_kind_cap_prevents_sensor_flooding(tmp_path):
    memory = ExperienceMemory(
        tmp_path / "memory.sqlite",
        limit=20,
        frontier_limit=20,
        kind_limits={"sensory_vision": 2},
        eviction_order=["sensory_vision"],
    )
    for index in range(6):
        memory.add(kind="sensory_vision", source="camera", title=str(index), url="", text=f"frame {index}", merge_existing=False)
    assert memory.document_count(kinds=("sensory_vision",)) == 2


def test_memory_trigram_fallback_avoids_table_scan_when_available(tmp_path):
    memory = ExperienceMemory(tmp_path / "memory.sqlite", limit=100, frontier_limit=20)
    if not memory._trigram_enabled:
        return
    for index in range(20):
        memory.add(kind="external", source="test", title=str(index), url=str(index), text=f"alpha unique passage {index}", merge_existing=False)
    rows = memory.search("que pas", limit=5)
    assert rows
    assert any("unique passage" in row["text"] for row in rows)


def test_identity_exemplars_are_deduplicated_and_alias_enrolment_is_pinned(tmp_path):
    identity = PerceptualIdentityMemory(tmp_path / "identity.json", threshold=0.5, max_exemplars=3)
    image = np.full((64, 64, 3), 100, dtype=np.uint8)
    identity.observe(torch.ones(8), alias="Mukul")
    entity_id = identity.entities[0]["entity_id"]
    assert identity.add_exemplar(entity_id, image, pinned=True, min_distance=0.02) is True
    assert identity.add_exemplar(entity_id, image, pinned=False, min_distance=0.02) is False
    assert identity.entities[0]["exemplars"][0]["pinned"] is True


def test_pretty_printed_json_manifest_is_supported(tmp_path):
    manifest = tmp_path / "speech.json"
    manifest.write_text(json.dumps([{"audio": "a.wav", "text": "hello"}, {"audio": "b.wav", "text": "world"}], indent=2), encoding="utf-8")
    rows = SpeechRecognizer.read_manifest(manifest)
    assert len(rows) == 2
    assert rows[0].audio == (tmp_path / "a.wav").resolve()


def test_speech_ready_requires_holdout_cer(tmp_path):
    recognizer = SpeechRecognizer(
        tmp_path,
        torch.device("cpu"),
        ready_min_steps=1,
        ready_max_cer=0.20,
    )
    recognizer.steps = 10
    recognizer.last_loss = 1.0
    recognizer.last_cer = 0.20
    # A low CER from too few examples is not enough to unlock the speech path.
    assert recognizer.ready is False
    recognizer.last_eval_examples = recognizer.ready_min_eval_examples - 1
    assert recognizer.ready is False
    recognizer.last_eval_examples = recognizer.ready_min_eval_examples
    # Metrics alone are insufficient: readiness requires source/speaker group isolation.
    assert recognizer.ready is False
    recognizer.last_split_strategy = "grouped_metadata"
    assert recognizer.ready is True


def test_corrupt_speech_checkpoint_is_quarantined(tmp_path):
    data = tmp_path / "data"
    data.mkdir(parents=True)
    checkpoint = data / "speech_ctc.pt"
    checkpoint.write_bytes(b"not a torch checkpoint")
    SpeechRecognizer(tmp_path, torch.device("cpu"), ready_min_steps=1, quarantine_corrupt=True)
    assert not checkpoint.exists()
    assert list(data.glob("speech_ctc_corrupt_*.pt"))


def test_web_malformed_content_length_is_ignored(monkeypatch):
    import requests

    from aletheia.config import load_config
    cfg = load_config(Path(__file__).parents[1] / "config.json")
    explorer = WebExplorer(cfg.web, 1, 1000, "AletheiaTest", 0, 0, 0)

    response = requests.Response()
    response.status_code = 200
    response.url = "https://example.com"
    response.headers["Content-Length"] = "not-a-number"
    response.headers["Content-Type"] = "text/plain"
    response._content = b"hello"
    response._content_consumed = True

    monkeypatch.setattr(explorer._dns_adapter, "_resolve_public", lambda host: "93.184.216.34")
    monkeypatch.setattr(explorer._dns_adapter, "pin_host", lambda host, ip: None)
    monkeypatch.setattr(explorer.session, "get", lambda *args, **kwargs: response)

    result = explorer._get("https://example.com")
    assert result.content == b"hello"
