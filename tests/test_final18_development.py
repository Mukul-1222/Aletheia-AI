import json

import torch

from aletheia.config import ModelConfig, load_config
from aletheia.model import AletheiaModel
from aletheia.speech import SpeakerIdentityMemory, StreamingSpeechSession


def test_response_loss_only_targets_response_and_backpropagates():
    cfg = ModelConfig(dim=48, layers=1, heads=4, ffn=96, recurrent_dim=64, context=96, dropout=0.0)
    model = AletheiaModel(cfg, ["chat"])
    loss = model.response_loss("Turn A:\nhello\nTurn B:\n", "hello there", device=torch.device("cpu"))
    assert torch.isfinite(loss)
    loss.backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.parameters())


def test_streaming_speech_session_emits_partial_and_final_without_pretrained_runtime():
    class FakeRecognizer:
        ready = True

        def transcribe(self, waveform, *, sample_rate=16000):
            return "hello world" if waveform.numel() >= 3200 else "hello"

    session = StreamingSpeechSession(
        FakeRecognizer(),
        sample_rate=16000,
        window_seconds=2.0,
        update_seconds=0.25,
        vad_threshold=0.01,
        end_silence_chunks=2,
        min_speech_seconds=0.1,
    )
    speech = torch.full((24000,), 0.05)
    update = session.push(speech[:4000])
    assert update["speech_active"]
    assert update["partial"] == "hello world"
    update = session.push(speech[4000:8000])
    assert update["speech_active"]
    silence = torch.zeros(800)
    update = session.push(silence)
    assert update["final"] == ""
    update = session.push(silence)
    assert update["final"] == "hello world"
    assert update["audio"] is not None


def test_speaker_identity_enrollment_is_explicit_and_persistent(tmp_path):
    path = tmp_path / "speaker_memory.json"
    memory = SpeakerIdentityMemory(path, threshold=0.80, max_exemplars=4)
    first = torch.tensor([1.0, 0.0, 0.0])
    second = torch.tensor([0.0, 1.0, 0.0])
    enrolled = memory.enroll(first, "primary_user")
    assert enrolled["new"] is True
    observed = memory.identify(first)
    assert observed["known"] is True
    assert observed["speaker_id"] == enrolled["speaker_id"]
    other = memory.observe(second, update=True)
    assert other["speaker_id"] != enrolled["speaker_id"]
    memory.flush()

    restored = SpeakerIdentityMemory(path, threshold=0.80, max_exemplars=4)
    assert restored.identify(first)["speaker_id"] == enrolled["speaker_id"]
    assert restored.summary()["profiles"] == 2


def test_runtime_exposes_final18_communication_controls():
    cfg = load_config()
    assert cfg.runtime.stream_sample_rate >= 8000
    assert cfg.runtime.stream_window_seconds > 0
    assert cfg.runtime.stream_update_seconds > 0
    assert 0.0 < cfg.runtime.speaker_threshold < 1.0
    assert cfg.runtime.speaker_max_exemplars > 0
    assert cfg.runtime.dialogue_online_steps > 0
    assert 0.0 < cfg.runtime.dialogue_online_learning_rate_scale <= 1.0


def test_speaker_memory_file_contains_no_audio_payload(tmp_path):
    path = tmp_path / "speaker_memory.json"
    memory = SpeakerIdentityMemory(path)
    memory.enroll(torch.tensor([1.0, 0.0, 0.0]), "user")
    memory.flush()
    payload = json.loads(path.read_text(encoding="utf-8"))
    serialized = json.dumps(payload)
    assert "wav" not in serialized.lower()
    assert "base64" not in serialized.lower()
