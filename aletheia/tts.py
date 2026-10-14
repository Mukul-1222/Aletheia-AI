from __future__ import annotations

import csv
import json
import math
import random
import time
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F

from .perception import AudioEncoder
from .data_splits import split_examples, stable_manifest_seed


def _stable_manifest_audio_key(audio: Path, manifest_path: Path) -> str:
    """Identify audio relative to its manifest so machine-local paths do not alter splits."""
    audio_path = Path(audio).expanduser().resolve()
    manifest_dir = Path(manifest_path).expanduser().resolve().parent
    try:
        return audio_path.relative_to(manifest_dir).as_posix()
    except ValueError:
        # Outside-manifest assets still get a stable name without embedding an absolute
        # home/workspace path into the split seed.
        return audio_path.name


@dataclass(frozen=True)
class TTSExample:
    audio: Path
    text: str
    speaker_audio: Path | None = None
    speaker: str = ""
    group_id: str = ""


def _encode_text(text: str, device: torch.device | None = None) -> torch.Tensor:
    raw = str(text).encode("utf-8", errors="replace")
    # 0 = padding, bytes map to 1..256, 257 = EOS.
    values = [int(value) + 1 for value in raw] + [257]
    return torch.tensor(values, dtype=torch.long, device=device)


def _read_wav(path: Path) -> tuple[torch.Tensor, int]:
    waveform, rate = AudioEncoder.load_wav(path)
    return waveform.float().reshape(-1), int(rate)


def _resample(waveform: torch.Tensor, source_rate: int, target_rate: int) -> torch.Tensor:
    value = waveform.float().reshape(-1)
    if int(source_rate) == int(target_rate):
        return value
    target = max(1, int(round(value.numel() * target_rate / max(1, source_rate))))
    return F.interpolate(value.view(1, 1, -1), size=target, mode="linear", align_corners=False).view(-1)


def build_mel_filterbank(
    *,
    sample_rate: int = 16000,
    n_fft: int = 400,
    n_mels: int = 80,
    f_min: float = 50.0,
    f_max: float | None = None,
    device: torch.device | None = None,
) -> torch.Tensor:
    """Create a triangular mel filterbank without external audio packages."""
    if f_max is None:
        f_max = sample_rate / 2.0
    if not (0.0 <= f_min < f_max <= sample_rate / 2.0):
        raise ValueError("invalid mel frequency bounds")

    def hz_to_mel(hz: torch.Tensor) -> torch.Tensor:
        return 2595.0 * torch.log10(1.0 + hz / 700.0)

    def mel_to_hz(mel: torch.Tensor) -> torch.Tensor:
        return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)

    points = torch.linspace(
        float(hz_to_mel(torch.tensor(f_min))),
        float(hz_to_mel(torch.tensor(f_max))),
        n_mels + 2,
        device=device,
    )
    hz = mel_to_hz(points)
    bins = torch.floor((n_fft + 1) * hz / sample_rate).long()
    bins = bins.clamp(0, n_fft // 2)

    fb = torch.zeros(n_mels, n_fft // 2 + 1, device=device)
    for index in range(n_mels):
        left, center, right = int(bins[index]), int(bins[index + 1]), int(bins[index + 2])
        if center > left:
            fb[index, left:center] = torch.linspace(
                0.0, 1.0, center - left, device=device, dtype=fb.dtype
            )
        if right > center:
            fb[index, center:right] = torch.linspace(
                1.0, 0.0, right - center, device=device, dtype=fb.dtype
            )
    fb = fb / fb.sum(dim=1, keepdim=True).clamp_min(1e-6)
    return fb


def waveform_to_mel(
    waveform: torch.Tensor,
    *,
    sample_rate: int = 16000,
    n_fft: int = 400,
    hop_length: int = 160,
    n_mels: int = 80,
    mel_filterbank: torch.Tensor | None = None,
) -> torch.Tensor:
    value = waveform.float().reshape(-1)
    if value.numel() < n_fft:
        value = F.pad(value, (0, n_fft - value.numel()))
    window = torch.hann_window(n_fft, device=value.device)
    spectrum = torch.stft(
        value,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=n_fft,
        window=window,
        center=True,
        return_complex=True,
    ).abs()
    if mel_filterbank is None:
        mel_filterbank = build_mel_filterbank(
            sample_rate=sample_rate,
            n_fft=n_fft,
            n_mels=n_mels,
            device=value.device,
        )
    mel = torch.matmul(mel_filterbank, spectrum)
    return torch.log1p(mel)


def mel_to_linear(
    mel: torch.Tensor,
    *,
    sample_rate: int = 16000,
    n_fft: int = 400,
    n_mels: int = 80,
    mel_filterbank: torch.Tensor | None = None,
) -> torch.Tensor:
    value = mel.float()
    if mel_filterbank is None:
        mel_filterbank = build_mel_filterbank(
            sample_rate=sample_rate,
            n_fft=n_fft,
            n_mels=n_mels,
            device=value.device,
        )
    inverse = torch.linalg.pinv(mel_filterbank)
    return inverse.matmul(torch.expm1(value).clamp_min(0.0)).clamp_min(0.0)


def griffin_lim(
    magnitude: torch.Tensor,
    *,
    n_fft: int = 400,
    hop_length: int = 160,
    iterations: int = 32,
) -> torch.Tensor:
    """Reconstruct a waveform from a non-negative linear magnitude spectrogram."""
    if magnitude.ndim != 2:
        raise ValueError("expected [frequency, frames] magnitude")
    magnitude = magnitude.float().clamp_min(0.0)
    window = torch.hann_window(n_fft, device=magnitude.device)
    phase = torch.exp(2j * math.pi * torch.rand_like(magnitude))
    complex_spec = magnitude * phase
    waveform = torch.istft(
        complex_spec,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=n_fft,
        window=window,
        center=True,
    )
    for _ in range(max(1, int(iterations))):
        rebuilt = torch.stft(
            waveform,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=n_fft,
            window=window,
            center=True,
            return_complex=True,
        )
        rebuilt = rebuilt / rebuilt.abs().clamp_min(1e-5)
        waveform = torch.istft(
            magnitude * rebuilt,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=n_fft,
            window=window,
            center=True,
        )
    return waveform.clamp(-1.0, 1.0)


class VoiceTTSModel(nn.Module):
    """Small learned text-to-acoustic model, randomly initialized.

    It predicts log-mel frames with an autoregressive attention decoder. Speaker conditioning
    comes from an embedding produced by Aletheia's own from-scratch audio representation.
    There are no pretrained TTS, vocoder, speaker, or language weights.
    """

    def __init__(
        self,
        *,
        speaker_dim: int = 256,
        text_dim: int = 128,
        encoder_hidden: int = 192,
        decoder_hidden: int = 256,
        attention_dim: int = 128,
        n_mels: int = 80,
    ):
        super().__init__()
        self.speaker_dim = int(speaker_dim)
        self.text_dim = int(text_dim)
        self.encoder_hidden = int(encoder_hidden)
        self.decoder_hidden = int(decoder_hidden)
        self.attention_dim = int(attention_dim)
        self.n_mels = int(n_mels)

        self.text_embedding = nn.Embedding(258, self.text_dim, padding_idx=0)
        self.text_encoder = nn.GRU(
            self.text_dim,
            self.encoder_hidden,
            batch_first=True,
            bidirectional=True,
        )
        self.speaker_projection = nn.Sequential(
            nn.Linear(self.speaker_dim, self.text_dim),
            nn.Tanh(),
        )
        encoder_dim = self.encoder_hidden * 2
        self.key_projection = nn.Linear(encoder_dim, self.attention_dim, bias=False)
        self.query_projection = nn.Linear(self.decoder_hidden, self.attention_dim, bias=False)
        self.attention_score = nn.Linear(self.attention_dim, 1, bias=False)
        self.decoder = nn.GRUCell(
            self.n_mels + encoder_dim + self.text_dim,
            self.decoder_hidden,
        )
        self.mel_projection = nn.Linear(self.decoder_hidden + encoder_dim + self.text_dim, self.n_mels)
        self.stop_projection = nn.Linear(self.decoder_hidden + encoder_dim + self.text_dim, 1)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(self.text_embedding.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.text_embedding.weight[0])
        for module in self.modules():
            if isinstance(module, nn.GRU):
                for name, parameter in module.named_parameters():
                    if "weight_ih" in name:
                        nn.init.xavier_uniform_(parameter)
                    elif "weight_hh" in name:
                        nn.init.orthogonal_(parameter)
                    elif "bias" in name:
                        nn.init.zeros_(parameter)
            elif isinstance(module, nn.GRUCell):
                for name, parameter in module.named_parameters():
                    if "weight_ih" in name:
                        nn.init.xavier_uniform_(parameter)
                    elif "weight_hh" in name:
                        nn.init.orthogonal_(parameter)
                    elif "bias" in name:
                        nn.init.zeros_(parameter)
            elif isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def encode_text(self, tokens: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        embedded = self.text_embedding(tokens)
        encoded, _ = self.text_encoder(embedded)
        mask = tokens.ne(0)
        return encoded, mask

    def _attention(
        self,
        hidden: torch.Tensor,
        encoded: torch.Tensor,
        mask: torch.Tensor,
        keys: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        query = self.query_projection(hidden).unsqueeze(1)
        scores = self.attention_score(torch.tanh(keys + query)).squeeze(-1)
        scores = scores.masked_fill(~mask, -1e4)
        weights = torch.softmax(scores, dim=-1)
        context = torch.bmm(weights.unsqueeze(1), encoded).squeeze(1)
        return context, weights

    def forward(
        self,
        tokens: torch.Tensor,
        speaker_embedding: torch.Tensor,
        target_mel: torch.Tensor,
        *,
        teacher_forcing: float = 1.0,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if tokens.ndim != 2 or speaker_embedding.ndim != 2 or target_mel.ndim != 3:
            raise ValueError("expected tokens [B,T], speaker [B,D], mel [B,M,F]")
        encoded, mask = self.encode_text(tokens)
        keys = self.key_projection(encoded)
        speaker = self.speaker_projection(F.normalize(speaker_embedding.float(), dim=-1))
        batch = tokens.size(0)
        frames = target_mel.size(-1)
        hidden = target_mel.new_zeros(batch, self.decoder_hidden)
        previous = target_mel.new_zeros(batch, self.n_mels)
        outputs: list[torch.Tensor] = []
        stops: list[torch.Tensor] = []
        attentions: list[torch.Tensor] = []
        for frame in range(frames):
            context, weights = self._attention(hidden, encoded, mask, keys)
            decoder_input = torch.cat([previous, context, speaker], dim=-1)
            hidden = self.decoder(decoder_input, hidden)
            combined = torch.cat([hidden, context, speaker], dim=-1)
            mel = self.mel_projection(combined)
            stop = self.stop_projection(combined).squeeze(-1)
            outputs.append(mel)
            stops.append(stop)
            attentions.append(weights)
            use_teacher = teacher_forcing >= 1.0 or (teacher_forcing > 0.0 and random.random() < teacher_forcing)
            previous = target_mel[:, :, frame] if use_teacher else mel.detach()
        return (
            torch.stack(outputs, dim=-1),
            torch.stack(stops, dim=-1),
            torch.stack(attentions, dim=1),
        )

    @torch.no_grad()
    def synthesize(
        self,
        tokens: torch.Tensor,
        speaker_embedding: torch.Tensor,
        *,
        max_frames: int = 240,
        stop_threshold: float = 0.55,
        min_frames: int = 12,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        self.eval()
        encoded, mask = self.encode_text(tokens)
        keys = self.key_projection(encoded)
        speaker = self.speaker_projection(F.normalize(speaker_embedding.float(), dim=-1))
        batch = tokens.size(0)
        hidden = encoded.new_zeros(batch, self.decoder_hidden)
        previous = encoded.new_zeros(batch, self.n_mels)
        frames: list[torch.Tensor] = []
        stops: list[torch.Tensor] = []
        attentions: list[torch.Tensor] = []
        for _frame in range(max(1, int(max_frames))):
            context, weights = self._attention(hidden, encoded, mask, keys)
            decoder_input = torch.cat([previous, context, speaker], dim=-1)
            hidden = self.decoder(decoder_input, hidden)
            combined = torch.cat([hidden, context, speaker], dim=-1)
            mel = self.mel_projection(combined)
            stop_probability = torch.sigmoid(self.stop_projection(combined).squeeze(-1))
            frames.append(mel)
            stops.append(stop_probability)
            attentions.append(weights)
            previous = mel
            if len(frames) >= max(1, min_frames) and bool((stop_probability > stop_threshold).all()):
                break
        return (
            torch.stack(frames, dim=-1),
            torch.stack(attentions, dim=1),
        )


class LearnedVoiceSynthesizer:
    """Train and persist a from-scratch speaker-conditioned voice synthesizer."""

    def __init__(
        self,
        root: Path,
        device: torch.device,
        *,
        speaker_encoder: nn.Module,
        learning_rate: float = 2e-4,
        holdout_fraction: float = 0.2,
        ready_min_steps: int = 25,
        ready_max_loss: float = 1.0,
        ready_min_eval_examples: int = 1,
        sample_rate: int = 16000,
        n_fft: int = 400,
        hop_length: int = 160,
        n_mels: int = 80,
        speaker_dim: int = 256,
        text_dim: int = 128,
        encoder_hidden: int = 192,
        decoder_hidden: int = 256,
        attention_dim: int = 128,
        griffin_lim_iterations: int = 32,
        max_synthesis_frames: int = 240,
    ):
        self.root = Path(root)
        self.device = device
        self.sample_rate = int(sample_rate)
        self.n_fft = int(n_fft)
        self.hop_length = int(hop_length)
        self.n_mels = int(n_mels)
        self.holdout_fraction = min(0.8, max(0.0, float(holdout_fraction)))
        self.ready_min_steps = max(1, int(ready_min_steps))
        self.ready_max_loss = max(0.0, float(ready_max_loss))
        self.ready_min_eval_examples = max(1, int(ready_min_eval_examples))
        self.griffin_lim_iterations = max(1, int(griffin_lim_iterations))
        self.max_synthesis_frames = max(16, int(max_synthesis_frames))
        self.mel_filterbank = build_mel_filterbank(
            sample_rate=self.sample_rate,
            n_fft=self.n_fft,
            n_mels=self.n_mels,
            device=self.device,
        )
        self.model = VoiceTTSModel(
            speaker_dim=speaker_dim,
            text_dim=text_dim,
            encoder_hidden=encoder_hidden,
            decoder_hidden=decoder_hidden,
            attention_dim=attention_dim,
            n_mels=self.n_mels,
        ).to(device)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=float(learning_rate), weight_decay=1e-4)
        self.speaker_encoder = speaker_encoder
        self.speaker_encoder.eval()
        for parameter in self.speaker_encoder.parameters():
            parameter.requires_grad_(False)
        self.steps = 0
        self.epochs = 0
        self.last_loss: float | None = None
        self.last_holdout_loss: float | None = None
        self.last_eval_examples = 0
        self.last_split_strategy = "not_evaluated"
        self.last_holdout_evaluation_mode = "teacher_forced_acoustic_reconstruction"
        self.last_manifest: str = ""
        self._load()

    @property
    def checkpoint_path(self) -> Path:
        return self.root / "data" / "voice_tts.pt"

    @property
    def ready(self) -> bool:
        return (
            self.steps >= self.ready_min_steps
            and self.last_holdout_loss is not None
            and math.isfinite(float(self.last_holdout_loss))
            and float(self.last_holdout_loss) <= self.ready_max_loss
            and self.last_eval_examples >= self.ready_min_eval_examples
            and self.last_split_strategy == "grouped_metadata"
        )

    @staticmethod
    def read_manifest(path: str | Path) -> list[TTSExample]:
        path = Path(path).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        examples: list[TTSExample] = []
        suffix = path.suffix.lower()
        if suffix == ".json":
            payload = json.loads(path.read_text(encoding="utf-8"))
            items = payload if isinstance(payload, list) else [payload]
            for item in items:
                if isinstance(item, dict):
                    audio = item.get("audio") or item.get("wav") or item.get("path")
                    text = item.get("text") or item.get("transcript") or item.get("sentence")
                    speaker_audio = item.get("speaker_audio") or item.get("reference_audio")
                    if audio and text is not None:
                        examples.append(
                            TTSExample(
                                (path.parent / str(audio)).resolve(),
                                str(text),
                                (path.parent / str(speaker_audio)).resolve() if speaker_audio else None,
                                str(item.get("speaker", "")),
                                str(item.get("group_id") or item.get("speaker_id") or item.get("speaker") or item.get("session_id") or item.get("recording_id") or item.get("source_id") or ""),
                            )
                        )
            return examples
        if suffix == ".jsonl":
            for line in path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                item = json.loads(line)
                if not isinstance(item, dict):
                    continue
                audio = item.get("audio") or item.get("wav") or item.get("path")
                text = item.get("text") or item.get("transcript") or item.get("sentence")
                speaker_audio = item.get("speaker_audio") or item.get("reference_audio")
                if audio and text is not None:
                    examples.append(
                        TTSExample(
                            (path.parent / str(audio)).resolve(),
                            str(text),
                            (path.parent / str(speaker_audio)).resolve() if speaker_audio else None,
                            str(item.get("speaker", "")),
                            str(item.get("group_id") or item.get("speaker_id") or item.get("speaker") or item.get("session_id") or item.get("recording_id") or item.get("source_id") or ""),
                        )
                    )
            return examples
        if suffix == ".csv":
            with path.open("r", encoding="utf-8", newline="") as handle:
                for item in csv.DictReader(handle):
                    audio = item.get("audio") or item.get("wav") or item.get("path")
                    text = item.get("text") or item.get("transcript") or item.get("sentence")
                    speaker_audio = item.get("speaker_audio") or item.get("reference_audio")
                    if audio and text is not None:
                        examples.append(
                            TTSExample(
                                (path.parent / str(audio)).resolve(),
                                str(text),
                                (path.parent / str(speaker_audio)).resolve() if speaker_audio else None,
                                str(item.get("speaker", "")),
                                str(item.get("group_id") or item.get("speaker_id") or item.get("speaker") or item.get("session_id") or item.get("recording_id") or item.get("source_id") or ""),
                            )
                        )
            return examples
        raise ValueError("TTS manifest must be .json, .jsonl, or .csv")

    def _speaker_embedding(self, waveform: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            return self.speaker_encoder(waveform.to(self.device).reshape(-1).float())[0].detach()

    def _load_example(self, item: TTSExample) -> tuple[torch.Tensor, torch.Tensor] | None:
        try:
            waveform, rate = _read_wav(item.audio)
            waveform = _resample(waveform, rate, self.sample_rate).clamp(-1.0, 1.0)
            if waveform.numel() < self.n_fft:
                return None
            mel = waveform_to_mel(
                waveform,
                sample_rate=self.sample_rate,
                n_fft=self.n_fft,
                hop_length=self.hop_length,
                n_mels=self.n_mels,
                mel_filterbank=self.mel_filterbank,
            )
            speaker_waveform = waveform
            if item.speaker_audio is not None and item.speaker_audio.exists():
                speaker_waveform, speaker_rate = _read_wav(item.speaker_audio)
                speaker_waveform = _resample(speaker_waveform, speaker_rate, self.sample_rate)
            speaker = self._speaker_embedding(speaker_waveform).squeeze(0)
            tokens = _encode_text(item.text, self.device)
            return tokens.unsqueeze(0), speaker.unsqueeze(0), mel.unsqueeze(0)
        except (OSError, ValueError, wave.Error):
            return None

    def _loss(self, predicted: torch.Tensor, predicted_stop: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        mel_loss = F.smooth_l1_loss(predicted, target)
        frames = target.size(-1)
        stop_target = torch.zeros_like(predicted_stop)
        stop_target[:, max(0, frames - 1) :] = 1.0
        stop_loss = F.binary_cross_entropy_with_logits(predicted_stop, stop_target)
        return mel_loss + 0.1 * stop_loss

    def _eval_examples(self, examples: list[TTSExample], limit: int = 32) -> float | None:
        if not examples:
            self.last_eval_examples = 0
            return None
        values: list[float] = []
        self.model.eval()
        with torch.no_grad():
            for item in examples[:max(1, int(limit))]:
                loaded = self._load_example(item)
                if loaded is None:
                    continue
                tokens, speaker, target = loaded
                predicted, predicted_stop, _ = self.model(
                    tokens, speaker, target, teacher_forcing=1.0
                )
                values.append(float(self._loss(predicted, predicted_stop, target).detach().cpu()))
        self.last_eval_examples = len(values)
        return float(sum(values) / len(values)) if values else None

    def train_manifest(self, manifest: str | Path, *, epochs: int = 1, max_examples: int = 0) -> dict:
        manifest_path = Path(manifest).expanduser().resolve()
        self.last_manifest = str(manifest_path)
        examples = self.read_manifest(manifest_path)
        if max_examples > 0:
            examples = examples[: int(max_examples)]
        if not examples:
            raise ValueError("TTS manifest contained no usable examples")
        # Keep the same evaluation split across repeated learning calls to prevent
        # previous holdout examples from becoming training examples in a later call.
        seed = stable_manifest_seed(
            [( _stable_manifest_audio_key(item.audio, manifest_path), item.text, item.speaker, item.group_id) for item in examples],
            manifest=manifest_path.name,
        )
        split_result = split_examples(
            examples,
            holdout_fraction=self.holdout_fraction,
            seed=seed,
            group_key=lambda item: item.group_id,
        )
        train_examples = split_result.train
        holdout = split_result.holdout
        self.last_split_strategy = split_result.strategy
        rng = random.Random(seed + self.epochs)
        if not train_examples:
            raise ValueError("TTS manifest needs at least one training example after holdout split")
        self.model.train()
        losses: list[float] = []
        skipped = 0
        for _epoch in range(max(1, int(epochs))):
            order = list(train_examples)
            rng.shuffle(order)
            for item in order:
                loaded = self._load_example(item)
                if loaded is None:
                    skipped += 1
                    continue
                tokens, speaker, target = loaded
                predicted, predicted_stop, _ = self.model(tokens, speaker, target, teacher_forcing=1.0)
                loss = self._loss(predicted, predicted_stop, target)
                if not torch.isfinite(loss):
                    skipped += 1
                    continue
                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()
                self.steps += 1
                self.last_loss = float(loss.detach().cpu())
                losses.append(self.last_loss)
            self.epochs += 1
            self.last_holdout_loss = self._eval_examples(holdout)
            self.save()
        return {
            "examples": len(examples),
            "train_examples": len(train_examples),
            "holdout_examples": len(holdout),
            "holdout_evaluated_examples": self.last_eval_examples,
            "split_strategy": self.last_split_strategy,
            "holdout_evaluation_mode": self.last_holdout_evaluation_mode,
            "epochs": max(1, int(epochs)),
            "steps": self.steps,
            "loss": float(sum(losses[-16:]) / max(1, len(losses[-16:]))) if losses else self.last_loss,
            "holdout_loss": self.last_holdout_loss,
            "training_ready": self.ready,
            "ready": self.ready,
            "readiness_note": ("teacher-forced acoustic reconstruction is not a perceptual intelligibility test"),
            "skipped": skipped,
        }

    def evaluate_manifest(self, manifest: str | Path, *, max_examples: int = 64) -> float | None:
        import hashlib
        manifest_path = Path(manifest).expanduser().resolve()
        examples = self.read_manifest(manifest_path)
        if not examples:
            return None
        seed = stable_manifest_seed(
            [( _stable_manifest_audio_key(item.audio, manifest_path), item.text, item.speaker, item.group_id) for item in examples],
            manifest=manifest_path.name,
        )
        split_result = split_examples(examples, holdout_fraction=self.holdout_fraction, seed=seed, group_key=lambda item: item.group_id)
        self.last_split_strategy = split_result.strategy
        return self._eval_examples(split_result.holdout, limit=max(1, int(max_examples)))

    def save(self) -> None:
        path = self.checkpoint_path
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        torch.save(
            {
                "model": self.model.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "steps": self.steps,
                "epochs": self.epochs,
                "last_loss": self.last_loss,
                "last_holdout_loss": self.last_holdout_loss,
                "last_eval_examples": self.last_eval_examples,
                "last_split_strategy": self.last_split_strategy,
                "last_holdout_evaluation_mode": self.last_holdout_evaluation_mode,
                "last_manifest": self.last_manifest,
            },
            tmp,
        )
        tmp.replace(path)

    def _load(self) -> None:
        path = self.checkpoint_path
        if not path.exists():
            return
        try:
            payload = torch.load(path, map_location=self.device, weights_only=True)
            self.model.load_state_dict(payload["model"])
            if "optimizer" in payload:
                self.optimizer.load_state_dict(payload["optimizer"])
            self.steps = int(payload.get("steps", 0))
            self.epochs = int(payload.get("epochs", 0))
            self.last_loss = payload.get("last_loss")
            self.last_holdout_loss = payload.get("last_holdout_loss")
            self.last_eval_examples = int(payload.get("last_eval_examples", 0))
            self.last_split_strategy = str(payload.get("last_split_strategy", "legacy_checkpoint") or "legacy_checkpoint")
            self.last_holdout_evaluation_mode = str(payload.get("last_holdout_evaluation_mode", "teacher_forced_acoustic_reconstruction") or "teacher_forced_acoustic_reconstruction")
            self.last_manifest = str(payload.get("last_manifest", "") or "")
        except Exception:
            if path.exists():
                stamp = time.strftime("%Y%m%d_%H%M%S")
                try:
                    path.replace(path.with_name(f"voice_tts_corrupt_{stamp}.pt"))
                except OSError:
                    pass
            self.steps = 0
            self.epochs = 0
            self.last_loss = None
            self.last_holdout_loss = None
            self.last_eval_examples = 0
            self.last_split_strategy = "not_evaluated"
            self.last_holdout_evaluation_mode = "teacher_forced_acoustic_reconstruction"
            self.last_manifest = ""

    @torch.no_grad()
    def synthesize_from_embedding(
        self,
        text: str,
        speaker_embedding: torch.Tensor,
        *,
        max_frames: int | None = None,
        stop_threshold: float = 0.55,
    ) -> torch.Tensor:
        tokens = _encode_text(text, self.device).unsqueeze(0)
        embedding = speaker_embedding.to(self.device).reshape(1, -1)
        mel, _ = self.model.synthesize(
            tokens,
            embedding,
            max_frames=max_frames or self.max_synthesis_frames,
            stop_threshold=stop_threshold,
        )
        linear = mel_to_linear(
            mel[0],
            sample_rate=self.sample_rate,
            n_fft=self.n_fft,
            n_mels=self.n_mels,
            mel_filterbank=self.mel_filterbank,
        )
        return griffin_lim(
            linear,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            iterations=self.griffin_lim_iterations,
        )

    @torch.no_grad()
    def synthesize_from_audio(
        self,
        text: str,
        waveform: torch.Tensor,
        *,
        sample_rate: int = 16000,
        max_frames: int | None = None,
        stop_threshold: float = 0.55,
    ) -> torch.Tensor:
        value = _resample(waveform.reshape(-1), sample_rate, self.sample_rate)
        speaker = self._speaker_embedding(value)
        return self.synthesize_from_embedding(
            text,
            speaker,
            max_frames=max_frames,
            stop_threshold=stop_threshold,
        )

    def write_wav(self, waveform: torch.Tensor, path: str | Path) -> Path:
        destination = Path(path).expanduser().resolve()
        destination.parent.mkdir(parents=True, exist_ok=True)
        value = waveform.detach().float().cpu().clamp(-1.0, 1.0)
        pcm = (value * 32767.0).round().to(torch.int16).numpy().tobytes()
        with wave.open(str(destination), "wb") as handle:
            handle.setnchannels(1)
            handle.setsampwidth(2)
            handle.setframerate(self.sample_rate)
            handle.writeframes(pcm)
        return destination
