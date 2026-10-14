from __future__ import annotations

import numpy as np
import pytest
import torch


def test_vision_image_preprocessing_and_embedding_smoke():
    from aletheia.perception import VisionEncoder

    model = VisionEncoder(embedding_dim=32).eval()
    rgb = np.zeros((48, 64, 3), dtype=np.uint8)
    rgb[:, :, 0] = 127
    prepared = model.prepare_image(rgb, size=64)
    assert tuple(prepared.shape) == (1, 3, 64, 64)
    assert float(prepared.min()) >= 0.0 and float(prepared.max()) <= 1.0
    with torch.inference_mode():
        embedding = model(prepared)
    assert tuple(embedding.shape) == (1, 32)
    assert torch.isfinite(embedding).all()
    assert torch.linalg.vector_norm(embedding, dim=-1).item() == pytest.approx(1.0, abs=1e-5)


def test_vision_preprocessing_rejects_invalid_channel_layout():
    from aletheia.perception import VisionEncoder

    with pytest.raises(ValueError, match="HxWx3/4"):
        VisionEncoder.prepare_image(np.zeros((16, 16, 2), dtype=np.uint8))


def test_audio_encoder_and_byte_ctc_asr_forward_smoke():
    from aletheia.perception import AudioEncoder
    from aletheia.speech import ByteCTCSpeechModel

    torch.manual_seed(21)
    waveform = torch.sin(torch.arange(3200, dtype=torch.float32) * 0.025)
    audio_model = AudioEncoder(embedding_dim=32).eval()
    speech_model = ByteCTCSpeechModel(hidden=24).eval()
    with torch.inference_mode():
        embedding = audio_model(waveform)
        log_probs = speech_model(waveform.unsqueeze(0))
    assert tuple(embedding.shape) == (1, 32)
    assert torch.isfinite(embedding).all()
    assert log_probs.ndim == 3 and log_probs.shape[0] == 1 and log_probs.shape[-1] == 257
    assert torch.isfinite(log_probs).all()
    # A forward pass is only an integration smoke test, not evidence of transcription quality.


def test_learned_tts_forward_and_synthesis_shapes():
    from aletheia.tts import VoiceTTSModel

    torch.manual_seed(22)
    model = VoiceTTSModel(speaker_dim=16, text_dim=12, encoder_hidden=16,
                          decoder_hidden=20, attention_dim=10, n_mels=8).eval()
    tokens = torch.tensor([[65, 66, 67]], dtype=torch.long)
    speaker = torch.randn(1, 16)
    target_mel = torch.randn(1, 8, 5)
    with torch.inference_mode():
        predicted, stops, attention = model(tokens, speaker, target_mel, teacher_forcing=1.0)
        generated, generated_attention = model.synthesize(tokens, speaker, max_frames=6, min_frames=6)
    assert tuple(predicted.shape) == (1, 8, 5)
    assert tuple(stops.shape) == (1, 5)
    assert tuple(attention.shape) == (1, 5, 3)
    assert generated.shape[0] == 1 and generated.shape[1] == 8
    assert generated.shape[-1] >= 1 and generated.shape[-1] <= 6
    assert generated_attention.shape[0] == 1
    assert torch.isfinite(predicted).all() and torch.isfinite(generated).all()
    # Randomly initialized synthesis is not considered ready or intelligible by this test.


def test_web_content_quality_flags_repetitive_boilerplate_without_claiming_truth():
    from aletheia.web import WebExplorer

    noisy = " ".join(["Click here to continue navigation."] * 120)
    result = WebExplorer.assess_content_quality(noisy)
    assert result["status"] == "low_quality"
    assert "very_low_token_diversity" in result["flags"] or "repeated_sentence_content" in result["flags"]
    assert "do not validate factual correctness" in result["limitations"]

    prose = (
        "Rivers change gradually as water carries sediment through the landscape. "
        "Researchers compare observations from different seasons to distinguish stable patterns "
        "from temporary changes caused by rainfall, temperature, and human activity. "
        "Geologists measure water depth and velocity at several locations along the channel. "
        "Their records are compared with rainfall gauges, satellite images, and earlier field notes. "
        "The combined evidence helps explain erosion, deposition, and changes in nearby habitats. "
        "Separate teams repeat measurements to check whether the same pattern appears again. "
    )
    healthy = WebExplorer.assess_content_quality(prose)
    assert healthy["status"] == "usable"
    assert healthy["character_count"] > 80

    residual = WebExplorer.assess_content_quality(
        "A verified paragraph remains after the fixed evaluation overlap is removed."
    )
    assert residual["status"] == "short_but_usable"
    assert "short_text" in residual["flags"]


def test_web_extraction_removes_hidden_and_navigation_regions():
    from aletheia.web import WebExplorer

    html = """
    <html><body>
      <nav>top menu navigation options</nav>
      <div aria-hidden="true">screen reader hidden duplicate</div>
      <div role="navigation">secondary navigation links</div>
      <main><p>The observation log records a distinct measured result.</p>
      <p>The observation log records a distinct measured result.</p>
      <p>A second paragraph provides a different result.</p></main>
      <script>should_not_appear()</script>
    </body></html>
    """
    text = WebExplorer.extract_text(html)
    assert "top menu" not in text
    assert "hidden duplicate" not in text
    assert "should_not_appear" not in text
    assert text.count("The observation log records a distinct measured result.") == 1
    assert "A second paragraph provides a different result." in text


def test_tts_jsonl_preserves_source_and_speaker_groups_for_holdout(tmp_path):
    import json
    from aletheia.tts import LearnedVoiceSynthesizer
    from aletheia.data_splits import split_examples

    manifest = tmp_path / "tts.jsonl"
    rows = [
        {"audio": "a.wav", "text": "first phrase", "speaker_id": "speaker-a", "source_id": "recording-a"},
        {"audio": "b.wav", "text": "second phrase", "speaker_id": "speaker-b", "source_id": "recording-b"},
        {"audio": "c.wav", "text": "third phrase", "speaker_id": "speaker-c", "source_id": "recording-c"},
        {"audio": "d.wav", "text": "fourth phrase", "speaker_id": "speaker-d", "source_id": "recording-d"},
    ]
    manifest.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")
    examples = LearnedVoiceSynthesizer.read_manifest(manifest)
    assert [row.group_id for row in examples] == ["speaker-a", "speaker-b", "speaker-c", "speaker-d"]
    split = split_examples(examples, holdout_fraction=0.5, seed=11, group_key=lambda row: row.group_id)
    assert split.strategy == "grouped_metadata"
    assert not ({row.group_id for row in split.train} & {row.group_id for row in split.holdout})


def test_tts_readiness_rejects_example_level_holdout(tmp_path):
    from aletheia.perception import AudioEncoder
    from aletheia.tts import LearnedVoiceSynthesizer

    synthesizer = LearnedVoiceSynthesizer(
        tmp_path,
        torch.device("cpu"),
        speaker_encoder=AudioEncoder(embedding_dim=256),
        ready_min_steps=1,
        ready_max_loss=10.0,
        ready_min_eval_examples=1,
        n_mels=8,
        text_dim=8,
        encoder_hidden=8,
        decoder_hidden=8,
        attention_dim=8,
        griffin_lim_iterations=1,
        max_synthesis_frames=4,
    )
    synthesizer.steps = 2
    synthesizer.last_holdout_loss = 0.5
    synthesizer.last_eval_examples = 2
    synthesizer.last_split_strategy = "example_level_no_usable_group_metadata"
    assert synthesizer.ready is False
    synthesizer.last_split_strategy = "grouped_metadata"
    assert synthesizer.ready is True
