import json
import math
import wave
from pathlib import Path

import torch

from aletheia.config import load_config
from aletheia.speech import SpeakerIdentityMemory
from aletheia.tts import (
    LearnedVoiceSynthesizer,
    VoiceTTSModel,
    griffin_lim,
    mel_to_linear,
    waveform_to_mel,
)


def _write_wav(path: Path, frequency: float = 220.0, seconds: float = 0.20, rate: int = 16000) -> None:
    n = max(1, int(rate * seconds))
    t = torch.arange(n, dtype=torch.float32) / rate
    audio = 0.15 * torch.sin(2.0 * math.pi * frequency * t)
    pcm = (audio * 32767.0).round().to(torch.int16).numpy().tobytes()
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(rate)
        handle.writeframes(pcm)


def test_tts_forward_backward_and_attention_normalization():
    model = VoiceTTSModel(
        speaker_dim=16,
        text_dim=16,
        encoder_hidden=24,
        decoder_hidden=32,
        attention_dim=16,
        n_mels=16,
    )
    tokens = torch.tensor([[1, 2, 3, 257]], dtype=torch.long)
    speaker = torch.randn(1, 16)
    target = torch.randn(1, 16, 12)
    predicted, stop, attention = model(tokens, speaker, target)
    assert predicted.shape == target.shape
    assert stop.shape == (1, 12)
    assert attention.shape == (1, 12, tokens.shape[1])
    assert torch.allclose(attention.sum(dim=-1), torch.ones_like(attention[:, :, 0]), atol=1e-5)
    loss = (predicted - target).pow(2).mean() + stop.pow(2).mean()
    loss.backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.parameters())


def test_mel_roundtrip_and_griffin_lim_are_finite():
    waveform = torch.randn(3200)
    mel = waveform_to_mel(waveform, n_mels=16)
    linear = mel_to_linear(mel, n_mels=16)
    rebuilt = griffin_lim(linear, iterations=2)
    assert mel.shape[0] == 16
    assert linear.shape[0] == 201
    assert rebuilt.numel() > 0
    assert torch.isfinite(rebuilt).all()


def test_tts_manifest_supports_json_and_speaker_reference(tmp_path):
    wav = tmp_path / "sample.wav"
    ref = tmp_path / "reference.wav"
    _write_wav(wav, 220.0)
    _write_wav(ref, 330.0)
    manifest = tmp_path / "tts.json"
    manifest.write_text(
        json.dumps([{"audio": "sample.wav", "text": "hello", "speaker_audio": "reference.wav", "speaker": "voice_a"}]),
        encoding="utf-8",
    )
    rows = LearnedVoiceSynthesizer.read_manifest(manifest)
    assert len(rows) == 1
    assert rows[0].audio == wav.resolve()
    assert rows[0].speaker_audio == ref.resolve()
    assert rows[0].speaker == "voice_a"


def test_tts_training_streams_from_disk_and_persists_checkpoint(tmp_path):
    wav1 = tmp_path / "one.wav"
    wav2 = tmp_path / "two.wav"
    _write_wav(wav1, 220.0)
    _write_wav(wav2, 330.0)
    manifest = tmp_path / "train.jsonl"
    manifest.write_text(
        "\n".join([
            json.dumps({"audio": "one.wav", "text": "hello", "speaker": "speaker-one"}),
            json.dumps({"audio": "two.wav", "text": "world", "speaker": "speaker-two"}),
        ]),
        encoding="utf-8",
    )

    from aletheia.perception import AudioEncoder

    device = torch.device("cpu")
    speaker_encoder = AudioEncoder(embedding_dim=256)
    synthesizer = LearnedVoiceSynthesizer(
        tmp_path,
        device,
        speaker_encoder=speaker_encoder,
        learning_rate=1e-3,
        holdout_fraction=0.5,
        ready_min_steps=1,
        ready_max_loss=100.0,
        n_mels=16,
        text_dim=16,
        encoder_hidden=24,
        decoder_hidden=32,
        attention_dim=16,
        griffin_lim_iterations=2,
        max_synthesis_frames=16,
    )
    result = synthesizer.train_manifest(manifest, epochs=1)
    assert result["steps"] >= 1
    assert synthesizer.checkpoint_path.exists()
    assert synthesizer.last_holdout_loss is not None
    assert synthesizer.ready

    restored = LearnedVoiceSynthesizer(
        tmp_path,
        device,
        speaker_encoder=speaker_encoder,
        learning_rate=1e-3,
        holdout_fraction=0.5,
        ready_min_steps=1,
        ready_max_loss=100.0,
        n_mels=16,
        text_dim=16,
        encoder_hidden=24,
        decoder_hidden=32,
        attention_dim=16,
        griffin_lim_iterations=2,
        max_synthesis_frames=16,
    )
    assert restored.steps == synthesizer.steps
    assert restored.ready


def test_tts_synthesis_conditioned_on_reference_voice(tmp_path):
    wav = tmp_path / "voice.wav"
    _write_wav(wav, 260.0)
    from aletheia.perception import AudioEncoder

    synthesizer = LearnedVoiceSynthesizer(
        tmp_path,
        torch.device("cpu"),
        speaker_encoder=AudioEncoder(embedding_dim=256),
        n_mels=16,
        text_dim=16,
        encoder_hidden=24,
        decoder_hidden=32,
        attention_dim=16,
        griffin_lim_iterations=2,
        max_synthesis_frames=8,
    )
    loaded = synthesizer._load_example(
        type("Example", (), {"audio": wav, "speaker_audio": None, "text": "hello"})()
    )
    assert loaded is not None
    output = synthesizer.synthesize_from_audio("hello", torch.randn(3200), sample_rate=16000, max_frames=8)
    assert output.numel() > 0
    assert torch.isfinite(output).all()


def test_speaker_memory_can_supply_voice_conditioning(tmp_path):
    memory = SpeakerIdentityMemory(tmp_path / "speakers.json")
    vector = torch.tensor([1.0, 0.0, 0.0])
    enrolled = memory.enroll(vector, "primary")
    embedding = memory.embedding_for("primary")
    assert embedding is not None
    assert embedding.shape == vector.shape
    assert torch.allclose(embedding, vector)
    assert memory.embedding_for(enrolled["speaker_id"]) is not None


def test_runtime_exposes_final22_voice_controls():
    cfg = load_config()
    runtime = cfg.runtime
    assert runtime.tts_sample_rate >= 8000
    assert runtime.tts_n_mels >= 16
    assert runtime.tts_ready_min_steps > 0
    assert runtime.tts_encoder_hidden > 0
    assert runtime.tts_decoder_hidden > 0
    assert runtime.tts_griffin_lim_iterations > 0
