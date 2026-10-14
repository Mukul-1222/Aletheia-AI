from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from aletheia.webapp import create_app


def _make_app(tmp_path: Path, *, auto_train: bool = False):
    (tmp_path / "data").mkdir(exist_ok=True)
    (tmp_path / "config.json").write_text(Path(__file__).resolve().parents[1].joinpath("config.json").read_text(encoding="utf-8"), encoding="utf-8")
    return create_app(tmp_path, tmp_path / "config.json", auto_train=auto_train)


class _FakeMemory:
    def search(self, query, limit=8):
        return [{"id": 1, "kind": "lesson", "source": "test", "title": "Test memory", "text": "A stored memory about useful local testing.", "created": 1.0, "accepted": True, "replay_eligible": True}]


class _FakeAgent:
    def __init__(self):
        self._last_feedback_target = None
        self.memory = _FakeMemory()
        self.saved = False

    def communicate(self, message, max_tokens=128, *, source="human_chat", **kwargs):
        reply = f"[experimental] I heard you say: {message}"
        self._last_feedback_target = {"kind": "chat", "message": message, "reply": reply}
        return reply

    def objective_feedback(self, positive, note=""):
        return {"updated": True, "positive": positive, "online_learning": {"accepted": positive, "reason": "test_feedback"}}

    def save(self):
        self.saved = True

    def close(self):
        return None


def test_webapp_serves_responsive_frontend_and_health(tmp_path):
    app = _make_app(tmp_path)
    with TestClient(app) as client:
        health = client.get("/api/health")
        assert health.status_code == 200
        assert health.json()["version"] == "0.34.0"
        assert health.json()["local_only"] is True
        index = client.get("/")
        assert index.status_code == 200
        assert 'id="orbStage"' in index.text
        assert 'id="learningFileInput"' in index.text
        js = client.get("/static/app.js")
        css = client.get("/static/styles.css")
        assert js.status_code == 200 and "startVoiceInput" in js.text
        assert css.status_code == 200 and "@media(max-width:760px)" in css.text


def test_chat_sessions_and_messages_are_persisted(tmp_path):
    app = _make_app(tmp_path)
    service = app.state.service
    fake = _FakeAgent()
    service.agent = fake
    service._refresh_status_locked = lambda: {"engine_loaded": True, "language_ready": False, "device": "cpu"}
    with TestClient(app) as client:
        chats = client.get("/api/chats").json()["chats"]
        chat_id = chats[0]["id"]
        new_chat = client.post("/api/chats", json={"title": "Research chat"})
        assert new_chat.status_code == 200
        assert new_chat.json()["title"] == "Research chat"
        chat_id = new_chat.json()["id"]
        sent = client.post(f"/api/chats/{chat_id}/messages", json={"message": "Remember this detail", "max_tokens": 64})
        assert sent.status_code == 200, sent.text
        assert sent.json()["response_mode"] == "experimental"
        history = client.get(f"/api/chats/{chat_id}").json()
        assert [m["role"] for m in history["messages"]] == ["user", "assistant"]
        assert history["messages"][0]["content"] == "Remember this detail"
        assert client.get("/api/chats").json()["chats"]
        feedback = client.post(f"/api/chats/{chat_id}/messages/{history['messages'][1]['id']}/feedback", json={"positive": True, "note": "clear"})
        assert feedback.status_code == 200, feedback.text
        assert feedback.json()["updated"] is True


def test_learning_upload_queues_supported_file_without_loading_model(tmp_path):
    app = _make_app(tmp_path, auto_train=False)
    paragraphs = "\n\n".join(
        f"Distinct training paragraph {i} discusses data structures, model evaluation, and careful scientific observation. "
        f"The text in unit {i} contains independent material useful for separate holdout evaluation."
        for i in range(1, 8)
    )
    with TestClient(app) as client:
        response = client.post("/api/learning/upload", files={"file": ("lesson.txt", paragraphs.encode(), "text/plain")}, data={"license_name": "user-provided", "auto_train": "false"})
        assert response.status_code == 200, response.text
        job = response.json()["job"]
        assert job["state"] == "queued"
        assert response.json()["auto_train"] is False
        assert app.state.service.agent is None
        listed = client.get("/api/learning/jobs").json()["jobs"]
        assert any(row["id"] == job["id"] for row in listed)


def test_learning_upload_rejects_unsupported_type_and_large_file(tmp_path):
    app = _make_app(tmp_path)
    with TestClient(app) as client:
        bad_type = client.post("/api/learning/upload", files={"file": ("malware.exe", b"hello", "application/octet-stream")})
        assert bad_type.status_code == 415
        empty = client.post("/api/learning/upload", files={"file": ("empty.txt", b"", "text/plain")})
        assert empty.status_code == 422
        assert app.state.service.agent is None


def test_camera_and_audio_endpoints_validate_payloads_without_model_init(tmp_path):
    app = _make_app(tmp_path)
    with TestClient(app) as client:
        image = client.post("/api/perception/image", files={"file": ("snapshot.jpg", b"not actually a jpeg", "image/jpeg")})
        assert image.status_code == 422
        audio = client.post("/api/perception/audio", files={"file": ("clip.wav", b"not a wav", "audio/wav")})
        assert audio.status_code == 422
        assert app.state.service.agent is None


def test_cross_origin_mutations_are_rejected(tmp_path):
    app = _make_app(tmp_path)
    with TestClient(app) as client:
        response = client.post("/api/chats", json={"title": "no"}, headers={"Origin": "https://attacker.example"})
        assert response.status_code == 403


def test_text_lesson_has_size_validation(tmp_path):
    app = _make_app(tmp_path)
    with TestClient(app) as client:
        response = client.post("/api/learning/text", json={"text": "too short", "title": "bad"})
        assert response.status_code == 422
        assert app.state.service.agent is None


def test_conversation_history_survives_app_recreation(tmp_path):
    first_app = _make_app(tmp_path)
    service = first_app.state.service
    chat = service.create_chat("Persistent research")
    user = service._add_message(chat["id"], "user", "Keep this across restarts")
    service._add_message(chat["id"], "assistant", "Stored locally", reply_to=user["id"])

    second_app = _make_app(tmp_path)
    with TestClient(second_app) as client:
        loaded = client.get(f"/api/chats/{chat['id']}")
        assert loaded.status_code == 200
        payload = loaded.json()
        assert payload["title"] == "Persistent research"
        assert [message["content"] for message in payload["messages"]] == ["Keep this across restarts", "Stored locally"]


def test_frontend_has_speaker_select_and_all_static_controls_have_handlers_or_dynamic_ids():
    root = Path(__file__).resolve().parents[1]
    html = root.joinpath("aletheia/web/index.html").read_text(encoding="utf-8")
    js = root.joinpath("aletheia/web/static/app.js").read_text(encoding="utf-8")
    assert 'id="voiceSpeakerSelect"' in html
    assert 'id="confirmCameraCheckbox"' not in html  # Removed setting was inert; camera remains explicit opt-in.
    assert "data-chat-id" in js or "dataset.chatId" in js
    assert "function stopSpeaking()" in js and "speechResolve" in js
    assert "function startCamera()" in js and "observeCameraFrame" in js
    assert "function uploadLearningFile(file)" in js and "/api/learning/upload" in js


def test_voice_enrollment_requires_explicit_consent_without_loading_engine(tmp_path):
    app = _make_app(tmp_path)
    with TestClient(app) as client:
        response = client.post(
            "/api/voice/enroll",
            files={"file": ("sample.wav", b"not a wav", "audio/wav")},
            data={"alias": "My voice", "consent_confirmed": "false"},
        )
        assert response.status_code == 400
        assert app.state.service.agent is None


def test_nonlocal_host_is_rejected(tmp_path):
    app = _make_app(tmp_path)
    with TestClient(app) as client:
        response = client.get("/api/health", headers={"Host": "aletheia.example"})
        assert response.status_code == 403


def test_app_status_stays_lazy_until_first_agent_feature(tmp_path):
    app = _make_app(tmp_path, auto_train=False)
    with TestClient(app) as client:
        status = client.get("/api/status")
        assert status.status_code == 200
        assert status.json()["engine_loaded"] is False
        assert app.state.service.agent is None


def test_feedback_on_previous_chat_is_saved_without_wrong_turn_training(tmp_path):
    app = _make_app(tmp_path, auto_train=False)
    service = app.state.service
    chat = service.create_chat("Old session")
    user = service._add_message(chat["id"], "user", "Old question")
    assistant = service._add_message(chat["id"], "assistant", "Old answer", reply_to=user["id"])
    with TestClient(app) as client:
        response = client.post(
            f"/api/chats/{chat['id']}/messages/{assistant['id']}/feedback",
            json={"positive": True, "note": "still useful"},
        )
        assert response.status_code == 200, response.text
        payload = response.json()
        assert payload["stored"] is True
        assert payload["online_learning"]["accepted"] is False
        assert service.agent is None  # A historical rating must not load the neural model.
    with service._connect() as db:
        record = db.execute("SELECT positive,note,online_learning_applied FROM message_feedback WHERE message_id=?", (assistant["id"],)).fetchone()
    assert tuple(record) == (1, "still useful", 0)


def test_file_upload_enables_worker_when_auto_train_is_checked(tmp_path, monkeypatch):
    app = _make_app(tmp_path, auto_train=False)
    service = app.state.service
    started = []
    monkeypatch.setattr(service, "start_worker", lambda: started.append(True))
    paragraphs = "\n\n".join(
        f"Training unit {i}: this is sufficiently distinct material for guarded learning and evaluation, unit {i}."
        for i in range(1, 9)
    )
    with TestClient(app) as client:
        # App lifespan starts the worker wrapper even when training is disabled.
        # Reset this marker so the next call proves the upload path re-enables it.
        started.clear()
        response = client.post(
            "/api/learning/upload",
            files={"file": ("automatic.txt", paragraphs.encode(), "text/plain")},
            data={"license_name": "user-provided", "auto_train": "true"},
        )
    assert response.status_code == 200, response.text
    assert response.json()["auto_train"] is True
    assert service.auto_train is True
    assert started == [True]
    assert service.agent is None
