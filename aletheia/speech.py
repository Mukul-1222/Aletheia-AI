from __future__ import annotations

import csv
import json
import math
import random
import time
import wave
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F

from .perception import AudioEncoder
from .data_splits import split_examples, stable_manifest_seed


_BLANK = 256
_VOCAB_SIZE = 257


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
class SpeechExample:
    audio: Path
    text: str
    group_id: str = ""


def _encode_target(text: str) -> torch.Tensor:
    raw = text.encode("utf-8", errors="replace")
    if not raw:
        raw = b" "
    return torch.tensor(list(raw), dtype=torch.long)


def _decode_bytes(tokens: Iterable[int]) -> str:
    values = bytes(int(token) for token in tokens if 0 <= int(token) < 256)
    return values.decode("utf-8", errors="replace").strip()


class ByteCTCSpeechModel(nn.Module):
    """Small randomly initialized speech-to-byte CTC model.

    It intentionally contains no pretrained acoustic or language weights. The model learns
    directly from paired WAV + transcript examples supplied by the user.
    """

    def __init__(self, hidden: int = 192):
        super().__init__()
        self.hidden = int(hidden)
        self.frontend = nn.Sequential(
            nn.Conv1d(201, 128, kernel_size=5, stride=2, padding=2),
            nn.GELU(),
            nn.Conv1d(128, self.hidden, kernel_size=5, stride=1, padding=2),
            nn.GELU(),
        )
        self.encoder = nn.GRU(self.hidden, self.hidden, num_layers=2, batch_first=True, bidirectional=True, dropout=0.0)
        self.output = nn.Linear(self.hidden * 2, _VOCAB_SIZE)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv1d):
                nn.init.kaiming_normal_(module.weight, nonlinearity="linear")
                nn.init.zeros_(module.bias)
            elif isinstance(module, nn.GRU):
                for name, parameter in module.named_parameters():
                    if "weight_ih" in name:
                        nn.init.xavier_uniform_(parameter)
                    elif "weight_hh" in name:
                        nn.init.orthogonal_(parameter)
                    elif "bias" in name:
                        nn.init.zeros_(parameter)
            elif isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)

    @staticmethod
    def _spectrogram(waveform: torch.Tensor) -> torch.Tensor:
        waveform = waveform.float().reshape(-1)
        if waveform.numel() < 400:
            waveform = F.pad(waveform, (0, 400 - waveform.numel()))
        window = torch.hann_window(400, device=waveform.device)
        spec = torch.stft(
            waveform,
            n_fft=400,
            hop_length=160,
            win_length=400,
            window=window,
            return_complex=True,
            center=True,
        )
        return torch.log1p(spec.abs())

    @classmethod
    def features(cls, waveform: torch.Tensor) -> torch.Tensor:
        if waveform.ndim == 1:
            return cls._spectrogram(waveform).unsqueeze(0)
        if waveform.ndim != 2:
            raise ValueError("expected [samples] or [batch,samples]")
        lengths = [cls._spectrogram(item) for item in waveform]
        max_frames = max(item.shape[-1] for item in lengths)
        padded = []
        for item in lengths:
            padded.append(F.pad(item, (0, max_frames - item.shape[-1])))
        return torch.stack(padded, dim=0)

    @staticmethod
    def output_lengths(sample_lengths: torch.Tensor) -> torch.Tensor:
        # _spectrogram pads very short clips to 400 samples. With center=True this
        # yields floor(effective_length / 160) + 1 frames for the supported lengths.
        effective = sample_lengths.long().clamp_min(400)
        frames = torch.div(effective, 160, rounding_mode="floor") + 1
        return torch.div(frames + 1, 2, rounding_mode="floor").clamp_min(1)

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        features = self.features(waveform)
        encoded = self.frontend(features)
        encoded = encoded.transpose(1, 2)
        encoded, _ = self.encoder(encoded)
        return self.output(encoded).log_softmax(dim=-1)

    @torch.no_grad()
    def greedy_decode(self, waveform: torch.Tensor) -> list[str]:
        logits = self(waveform).argmax(dim=-1).cpu()
        decoded: list[str] = []
        for row in logits:
            previous = _BLANK
            tokens: list[int] = []
            for token in row.tolist():
                if token != _BLANK and token != previous:
                    tokens.append(token)
                previous = token
            decoded.append(_decode_bytes(tokens))
        return decoded


class SpeechRecognizer:
    """Optional from-scratch ASR bridge for paired audio/text learning."""

    def __init__(self, root: Path, device: torch.device, *, learning_rate: float = 2e-4, ready_min_steps: int = 100, ready_max_cer: float = 0.35, ready_min_eval_examples: int = 1, holdout_fraction: float = 0.20, batch_size: int = 4, eval_max_examples: int = 256, augmentation: bool = True, quarantine_corrupt: bool = True):
        self.root = Path(root)
        self.device = device
        self.model = ByteCTCSpeechModel().to(device)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=float(learning_rate), weight_decay=1e-3)
        self.loss_fn = nn.CTCLoss(blank=_BLANK, zero_infinity=True)
        self.steps = 0
        self.epochs = 0
        self.last_loss: float | None = None
        self.last_cer: float | None = None
        self.last_eval_examples = 0
        self.last_split_strategy = "not_evaluated"
        self.last_manifest: str = ""
        self.ready_min_steps = max(1, int(ready_min_steps))
        self.ready_min_eval_examples = max(1, int(ready_min_eval_examples))
        self.ready_max_cer = max(0.0, float(ready_max_cer))
        self.holdout_fraction = min(0.8, max(0.0, float(holdout_fraction)))
        self.batch_size = max(1, int(batch_size))
        self.eval_max_examples = max(1, int(eval_max_examples))
        self.augmentation = bool(augmentation)
        self.quarantine_corrupt = bool(quarantine_corrupt)
        self._load()

    @property
    def checkpoint_path(self) -> Path:
        return self.root / "data" / "speech_ctc.pt"

    @property
    def ready(self) -> bool:
        return (
            self.steps >= self.ready_min_steps
            and self.last_loss is not None
            and math.isfinite(float(self.last_loss))
            and self.last_cer is not None
            and math.isfinite(float(self.last_cer))
            and self.last_eval_examples >= self.ready_min_eval_examples
            and float(self.last_cer) <= self.ready_max_cer
            and self.last_split_strategy == "grouped_metadata"
        )

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
            self.last_cer = payload.get("last_cer")
            self.last_eval_examples = int(payload.get("last_eval_examples", 0))
            self.last_split_strategy = str(payload.get("last_split_strategy", "legacy_checkpoint") or "legacy_checkpoint")
            self.last_manifest = str(payload.get("last_manifest", "") or "")
        except Exception:
            # Preserve the damaged checkpoint for inspection rather than silently erasing it.
            if self.quarantine_corrupt and path.exists():
                stamp = time.strftime("%Y%m%d_%H%M%S")
                try:
                    path.replace(path.with_name(f"speech_ctc_corrupt_{stamp}.pt"))
                except OSError:
                    pass
            self.steps = 0
            self.epochs = 0
            self.last_loss = None
            self.last_cer = None
            self.last_eval_examples = 0
            self.last_split_strategy = "not_evaluated"
            self.last_manifest = ""

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
                "last_cer": self.last_cer,
                "last_eval_examples": self.last_eval_examples,
                "last_split_strategy": self.last_split_strategy,
                "last_manifest": self.last_manifest,
            },
            tmp,
        )
        tmp.replace(path)

    @staticmethod
    def read_manifest(path: str | Path) -> list[SpeechExample]:
        path = Path(path).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        examples: list[SpeechExample] = []
        if path.suffix.lower() == ".json":
            payload = json.loads(path.read_text(encoding="utf-8"))
            items = payload if isinstance(payload, list) else [payload]
            for item in items:
                if not isinstance(item, dict):
                    continue
                audio = item.get("audio") or item.get("wav") or item.get("path")
                text = item.get("text") or item.get("transcript") or item.get("sentence")
                if audio and text is not None:
                    examples.append(SpeechExample((path.parent / str(audio)).resolve(), str(text), str(item.get("group_id") or item.get("speaker_id") or item.get("speaker") or item.get("session_id") or item.get("recording_id") or item.get("source_id") or "")))
            return examples
        if path.suffix.lower() == ".jsonl":
            lines = path.read_text(encoding="utf-8").splitlines()
            items = [json.loads(line) for line in lines if line.strip()]
            for item in items:
                audio = item.get("audio") or item.get("wav") or item.get("path")
                text = item.get("text") or item.get("transcript") or item.get("sentence")
                if audio and text is not None:
                    examples.append(SpeechExample((path.parent / str(audio)).resolve(), str(text), str(item.get("group_id") or item.get("speaker_id") or item.get("speaker") or item.get("session_id") or item.get("recording_id") or item.get("source_id") or "")))
            return examples
        if path.suffix.lower() == ".csv":
            with path.open("r", encoding="utf-8", newline="") as handle:
                for row in csv.DictReader(handle):
                    audio = row.get("audio") or row.get("wav") or row.get("path")
                    text = row.get("text") or row.get("transcript") or row.get("sentence")
                    if audio and text is not None:
                        examples.append(SpeechExample((path.parent / str(audio)).resolve(), str(text), str(row.get("group_id") or row.get("speaker_id") or row.get("speaker") or row.get("session_id") or row.get("recording_id") or row.get("source_id") or "")))
            return examples
        raise ValueError("speech manifest must be .jsonl, .json, or .csv")

    @staticmethod
    def _load_example(item: SpeechExample, *, augment: bool = False) -> tuple[torch.Tensor, torch.Tensor] | None:
        try:
            waveform, rate = AudioEncoder.load_wav(item.audio)
            if rate != 16000:
                target = max(1, int(round(waveform.numel() * 16000 / rate)))
                waveform = F.interpolate(waveform.view(1, 1, -1), size=target, mode="linear", align_corners=False).view(-1)
            waveform = waveform.float()
            if waveform.numel() < 256:
                return None
            if augment:
                gain = random.uniform(0.85, 1.15)
                waveform = waveform * gain
                if waveform.numel() > 800:
                    shift = random.randint(-min(800, waveform.numel() // 8), min(800, waveform.numel() // 8))
                    waveform = torch.roll(waveform, shift)
                noise_scale = random.uniform(0.0, 0.008)
                if noise_scale:
                    waveform = waveform + torch.randn_like(waveform) * noise_scale
                if random.random() < 0.35:
                    start = random.randint(0, max(0, waveform.numel() - 200))
                    width = random.randint(40, min(200, waveform.numel() - start)) if waveform.numel() > 200 else 0
                    if width:
                        waveform[start:start + width] *= random.uniform(0.0, 0.25)
            return waveform.clamp(-1.0, 1.0), _encode_target(item.text)
        except (OSError, ValueError, wave.Error):
            return None

    @staticmethod
    def _levenshtein(reference: str, hypothesis: str) -> int:
        previous = list(range(len(hypothesis) + 1))
        for i, ref_char in enumerate(reference, 1):
            current = [i]
            for j, hyp_char in enumerate(hypothesis, 1):
                current.append(min(current[-1] + 1, previous[j] + 1, previous[j - 1] + (ref_char != hyp_char)))
            previous = current
        return previous[-1]

    @classmethod
    def _cer(cls, references: list[str], hypotheses: list[str]) -> float:
        edits = sum(cls._levenshtein(ref, hyp) for ref, hyp in zip(references, hypotheses))
        chars = sum(max(1, len(ref)) for ref in references)
        return float(edits) / float(max(1, chars))

    def _evaluate_holdout(self, examples: list[SpeechExample]) -> float | None:
        if not examples:
            self.last_eval_examples = 0
            return None
        subset = examples[: self.eval_max_examples]
        self.model.eval()
        refs: list[str] = []
        hyps: list[str] = []
        with torch.no_grad():
            for item in subset:
                pair = self._load_example(item, augment=False)
                if pair is None:
                    continue
                waveform, target = pair
                hypothesis = self.model.greedy_decode(waveform.to(self.device).unsqueeze(0))[0]
                refs.append(item.text.strip())
                hyps.append(hypothesis.strip())
        self.last_eval_examples = len(refs)
        if not refs:
            return None
        return self._cer(refs, hyps)

    @staticmethod
    def _audio_frames(path: Path) -> int:
        try:
            with wave.open(str(path), "rb") as handle:
                return int(handle.getnframes())
        except (OSError, wave.Error):
            return 0

    @staticmethod
    def _feature_augment(features: torch.Tensor) -> torch.Tensor:
        """Small SpecAugment-style masks applied only to training features."""
        out = features.clone()
        if out.size(-1) >= 8 and random.random() < 0.5:
            width = random.randint(1, max(1, out.size(-1) // 12))
            start = random.randint(0, max(0, out.size(-1) - width))
            out[:, start:start + width] = 0.0
        if out.size(-2) >= 8 and random.random() < 0.5:
            width = random.randint(1, max(1, out.size(-2) // 12))
            start = random.randint(0, max(0, out.size(-2) - width))
            out[:, start:start + width] = out.mean(dim=(-2, -1), keepdim=True)
        return out

    def train_manifest(self, manifest: str | Path, *, epochs: int = 1, max_examples: int = 0) -> dict:
        manifest_path = Path(manifest).expanduser().resolve()
        self.last_manifest = str(manifest_path)
        examples = self.read_manifest(manifest_path)
        if max_examples > 0:
            examples = examples[: int(max_examples)]
        if not examples:
            raise ValueError("speech manifest contained no examples")
        # The same manifest must retain the same holdout across repeated training calls;
        # otherwise earlier holdout utterances can silently drift into training.
        seed = stable_manifest_seed([( _stable_manifest_audio_key(x.audio, manifest_path), x.text, x.group_id) for x in examples], manifest=manifest_path.name)
        split_result = split_examples(
            examples,
            holdout_fraction=self.holdout_fraction,
            seed=seed,
            group_key=lambda item: item.group_id,
        )
        train_examples = split_result.train
        holdout_examples = split_result.holdout
        self.last_split_strategy = split_result.strategy
        rng = random.Random(seed + self.epochs)
        if not train_examples:
            raise ValueError("speech manifest needs at least one training example after holdout split")

        self.model.train()
        losses: list[float] = []
        skipped = 0
        for _epoch_index in range(max(1, int(epochs))):
            # Bucket by inexpensive WAV header duration, then shuffle whole batches. This
            # keeps padding bounded without loading the corpus into RAM.
            ordered = sorted(range(len(train_examples)), key=lambda i: self._audio_frames(train_examples[i].audio))
            batches = [ordered[start:start + self.batch_size] for start in range(0, len(ordered), self.batch_size)]
            rng.shuffle(batches)
            for batch_indices in batches:
                batch_items = [train_examples[i] for i in batch_indices]
                pairs = []
                for item in batch_items:
                    pair = self._load_example(item, augment=self.augmentation)
                    if pair is not None:
                        pairs.append((pair, item.text))
                if not pairs:
                    continue
                waveforms = [pair[0] for pair in pairs]
                targets = [pair[1] for pair in pairs]
                feature_list = [self.model.features(waveform)[0] for waveform in waveforms]
                max_frames = max(item.shape[-1] for item in feature_list)
                features = torch.stack([F.pad(item, (0, max_frames - item.shape[-1])) for item in feature_list], dim=0)
                if self.augmentation:
                    features = self._feature_augment(features)
                features = features.to(self.device)
                encoded = self.model.frontend(features)
                encoded = encoded.transpose(1, 2)
                encoded, _ = self.model.encoder(encoded)
                log_probs = self.model.output(encoded).log_softmax(dim=-1).transpose(0, 1)
                input_lengths = torch.tensor([((item.shape[-1] + 1) // 2) for item in feature_list], device=self.device, dtype=torch.long)
                target_lengths = torch.tensor([target.numel() for target in targets], device=self.device, dtype=torch.long)
                keep = input_lengths > target_lengths
                if not bool(keep.any()):
                    skipped += len(targets)
                    continue
                if not bool(keep.all()):
                    skipped += int((~keep).sum().item())
                    indices = torch.nonzero(keep, as_tuple=False).view(-1).tolist()
                    log_probs = log_probs[:, indices, :]
                    targets = [targets[i] for i in indices]
                    input_lengths = input_lengths[keep]
                    target_lengths = target_lengths[keep]
                target_tensor = torch.cat(targets).to(self.device)
                loss = self.loss_fn(log_probs, target_tensor, input_lengths, target_lengths)
                if not torch.isfinite(loss):
                    skipped += len(targets)
                    continue
                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5.0)
                self.optimizer.step()
                self.steps += 1
                self.last_loss = float(loss.detach().cpu())
                losses.append(self.last_loss)
            self.epochs += 1
            self.last_cer = self._evaluate_holdout(holdout_examples)
            self.save()
        return {
            "examples": len(examples),
            "train_examples": len(train_examples),
            "holdout_examples": len(holdout_examples),
            "holdout_evaluated_examples": self.last_eval_examples,
            "split_strategy": self.last_split_strategy,
            "epochs": max(1, int(epochs)),
            "steps": self.steps,
            "loss": float(sum(losses[-16:]) / max(1, len(losses[-16:]))) if losses else self.last_loss,
            "cer": self.last_cer,
            "ready": self.ready,
            "skipped": skipped,
        }

    def evaluate_manifest(self, manifest: str | Path, *, max_examples: int = 256) -> float | None:
        import hashlib
        manifest_path = Path(manifest).expanduser().resolve()
        examples = self.read_manifest(manifest_path)
        if not examples:
            return None
        seed = stable_manifest_seed([( _stable_manifest_audio_key(x.audio, manifest_path), x.text, x.group_id) for x in examples], manifest=manifest_path.name)
        split_result = split_examples(examples, holdout_fraction=self.holdout_fraction, seed=seed, group_key=lambda item: item.group_id)
        holdout = split_result.holdout
        self.last_split_strategy = split_result.strategy
        previous = self.eval_max_examples
        self.eval_max_examples = max(1, int(max_examples))
        try:
            return self._evaluate_holdout(holdout)
        finally:
            self.eval_max_examples = previous

    def transcribe(self, waveform: torch.Tensor, *, sample_rate: int = 16000) -> str:
        if sample_rate != 16000:
            target = max(1, int(round(waveform.numel() * 16000 / sample_rate)))
            waveform = F.interpolate(waveform.float().view(1, 1, -1), size=target, mode="linear", align_corners=False).view(-1)
        self.model.eval()
        return self.model.greedy_decode(waveform.to(self.device).unsqueeze(0))[0]


class SpeakerIdentityMemory:
    """Persistent voice-identity prototypes learned from Aletheia's own audio embeddings.

    No speaker labels or pretrained classifier are built into the model. A profile is an
    observed embedding cluster that can optionally acquire an operator-supplied alias during
    enrollment. Prototypes are updated only on confident matches or explicit enrollment.
    """

    def __init__(self, path: Path, *, threshold: float = 0.78, max_exemplars: int = 8):
        self.path = Path(path)
        self.threshold = float(threshold)
        self.max_exemplars = max(1, int(max_exemplars))
        self.profiles: list[dict] = []
        self._dirty = False
        self._load()

    @staticmethod
    def _cosine(a: list[float] | torch.Tensor, b: list[float] | torch.Tensor) -> float:
        left = torch.as_tensor(a, dtype=torch.float32).reshape(-1)
        right = torch.as_tensor(b, dtype=torch.float32).reshape(-1)
        denom = float(left.norm().item() * right.norm().item())
        return float(torch.dot(left, right).item() / denom) if denom else 0.0

    @staticmethod
    def _normalize(vector: torch.Tensor) -> list[float]:
        value = vector.detach().float().cpu().reshape(-1)
        norm = float(value.norm().item())
        if norm:
            value = value / norm
        return value.tolist()

    def _load(self) -> None:
        if not self.path.exists():
            return
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
            profiles = payload.get("profiles", []) if isinstance(payload, dict) else []
            self.profiles = [dict(item) for item in profiles if isinstance(item, dict) and item.get("speaker_id")]
        except (OSError, ValueError, TypeError):
            self.profiles = []

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps({"profiles": self.profiles}, ensure_ascii=False, indent=2), encoding="utf-8")
        tmp.replace(self.path)
        self._dirty = False

    def flush(self) -> None:
        if self._dirty:
            self._save()

    def _next_id(self) -> str:
        used = {str(item.get("speaker_id")) for item in self.profiles}
        index = 1
        while f"speaker_{index:04d}" in used:
            index += 1
        return f"speaker_{index:04d}"

    def _best(self, embedding: torch.Tensor) -> tuple[dict | None, float]:
        vector = self._normalize(embedding)
        best: dict | None = None
        best_score = -1.0
        for profile in self.profiles:
            centroid = profile.get("centroid")
            if not centroid:
                continue
            score = self._cosine(vector, centroid)
            if score > best_score:
                best = profile
                best_score = score
        return best, best_score

    def observe(self, embedding: torch.Tensor, *, alias: str = "", update: bool = True) -> dict:
        vector = self._normalize(embedding)
        best, score = self._best(embedding)
        explicit = bool(str(alias).strip())
        matched = best is not None and (score >= self.threshold or explicit)
        created = best is None or not matched
        if created:
            speaker_id = self._next_id()
            profile = {
                "speaker_id": speaker_id,
                "centroid": vector,
                "count": 1,
                "aliases": [str(alias).strip()] if explicit else [],
                "exemplars": [vector],
                "updated": time.time(),
            }
            self.profiles.append(profile)
            best = profile
            score = 1.0
        else:
            speaker_id = str(best["speaker_id"])
            if update:
                count = max(1, int(best.get("count", 1)))
                centroid = torch.as_tensor(best["centroid"], dtype=torch.float32)
                incoming = torch.as_tensor(vector, dtype=torch.float32)
                centroid = (centroid * count + incoming) / float(count + 1)
                centroid = F.normalize(centroid, dim=0)
                best["centroid"] = centroid.tolist()
                best["count"] = count + 1
                if explicit and str(alias).strip() not in best.setdefault("aliases", []):
                    best["aliases"].append(str(alias).strip())
                exemplars = list(best.setdefault("exemplars", []))
                if not exemplars or max(self._cosine(vector, item) for item in exemplars) < 0.97:
                    exemplars.append(vector)
                    best["exemplars"] = exemplars[-self.max_exemplars:]
                best["updated"] = time.time()
        if update:
            self._dirty = True
        return {
            "speaker_id": str(best["speaker_id"]),
            "score": float(score),
            "new": bool(created),
            "alias": str(alias).strip(),
            "aliases": list(best.get("aliases", [])),
            "count": int(best.get("count", 1)),
        }

    def enroll(self, embedding: torch.Tensor, alias: str) -> dict:
        alias_value = str(alias).strip()
        if not alias_value:
            raise ValueError("speaker enrollment requires a non-empty alias")
        existing = next((item for item in self.profiles if alias_value in item.get("aliases", [])), None)
        vector = self._normalize(embedding)
        if existing is None:
            profile = {
                "speaker_id": self._next_id(),
                "centroid": vector,
                "count": 1,
                "aliases": [alias_value],
                "exemplars": [vector],
                "updated": time.time(),
            }
            self.profiles.append(profile)
            self._dirty = True
            return {"speaker_id": profile["speaker_id"], "score": 1.0, "new": True, "alias": alias_value, "aliases": [alias_value], "count": 1, "enrolled": True}
        result = self.observe(embedding, alias=alias_value, update=True)
        result["speaker_id"] = str(existing["speaker_id"])
        result["enrolled"] = True
        self._dirty = True
        return result

    def identify(self, embedding: torch.Tensor) -> dict:
        best, score = self._best(embedding)
        if best is None or score < self.threshold:
            return {"speaker_id": None, "score": float(max(score, 0.0)), "known": False, "aliases": []}
        return {
            "speaker_id": str(best["speaker_id"]),
            "score": float(score),
            "known": True,
            "aliases": list(best.get("aliases", [])),
        }

    def link(self, speaker_id: str, entity_id: str) -> bool:
        for profile in self.profiles:
            if str(profile.get("speaker_id")) == str(speaker_id):
                profile["entity_ids"] = sorted(set(profile.get("entity_ids", [])) | {str(entity_id)})
                profile["updated"] = time.time()
                self._dirty = True
                return True
        return False

    def embedding_for(self, identifier: str) -> torch.Tensor | None:
        """Return a stored normalized voice prototype by speaker id or alias."""
        key = str(identifier).strip()
        if not key:
            return None
        for profile in self.profiles:
            aliases = [str(item) for item in profile.get("aliases", [])]
            if key == str(profile.get("speaker_id")) or key in aliases:
                centroid = profile.get("centroid")
                if centroid:
                    return F.normalize(torch.as_tensor(centroid, dtype=torch.float32), dim=0)
        return None

    def summary(self) -> dict:
        return {
            "profiles": len(self.profiles),
            "aliases": sum(len(item.get("aliases", [])) for item in self.profiles),
            "known_profiles": sum(bool(item.get("aliases")) for item in self.profiles),
        }


class StreamingSpeechSession:
    """Incremental utterance detector/decoder around the from-scratch CTC recognizer.

    The recognizer itself remains bidirectional over the current audio window. The session
    supplies streaming behavior at the interaction layer: VAD starts/stops utterances, recent
    audio is retained, hypotheses are refreshed periodically, and only a stable text prefix is
    exposed as the partial transcript. No pretrained streaming ASR is used.
    """

    def __init__(
        self,
        recognizer: SpeechRecognizer,
        *,
        sample_rate: int = 16000,
        window_seconds: float = 3.0,
        update_seconds: float = 0.75,
        vad_threshold: float = 0.02,
        end_silence_chunks: int = 4,
        min_speech_seconds: float = 0.25,
        max_utterance_seconds: float = 20.0,
    ):
        self.recognizer = recognizer
        self.sample_rate = max(8000, int(sample_rate))
        self.window_samples = max(512, int(round(float(window_seconds) * self.sample_rate)))
        self.update_samples = max(256, int(round(float(update_seconds) * self.sample_rate)))
        self.vad_threshold = max(1e-5, float(vad_threshold))
        self.end_silence_chunks = max(1, int(end_silence_chunks))
        self.min_speech_samples = max(1, int(round(float(min_speech_seconds) * self.sample_rate)))
        self.max_utterance_samples = max(self.window_samples, int(round(float(max_utterance_seconds) * self.sample_rate)))
        self.buffer = torch.empty(0, dtype=torch.float32)
        self.pending = torch.empty(0, dtype=torch.float32)
        self.speech_active = False
        self.silence_chunks = 0
        self.last_partial = ""
        self.last_stable = ""
        self._last_decode_samples = 0
        self._hypotheses: list[str] = []

    @staticmethod
    def _common_prefix(left: str, right: str) -> str:
        length = min(len(left), len(right))
        index = 0
        while index < length and left[index] == right[index]:
            index += 1
        prefix = left[:index]
        boundary = max(prefix.rfind(" "), prefix.rfind("\n"), prefix.rfind("\t"))
        return prefix[:boundary + 1].strip()

    @staticmethod
    def _rms(waveform: torch.Tensor) -> float:
        value = waveform.float().reshape(-1)
        return float(value.pow(2).mean().sqrt().item()) if value.numel() else 0.0

    def _resample(self, waveform: torch.Tensor, sample_rate: int) -> torch.Tensor:
        value = waveform.float().reshape(-1)
        if int(sample_rate) == self.sample_rate:
            return value
        target = max(1, int(round(value.numel() * self.sample_rate / max(1, int(sample_rate)))))
        return F.interpolate(value.view(1, 1, -1), size=target, mode="linear", align_corners=False).view(-1)

    def _decode_partial(self) -> str:
        if not self.recognizer.ready or self.buffer.numel() < self.min_speech_samples:
            return ""
        window = self.buffer[-self.window_samples:]
        hypothesis = self.recognizer.transcribe(window, sample_rate=self.sample_rate).strip()
        if hypothesis:
            self._hypotheses.append(hypothesis)
            self._hypotheses = self._hypotheses[-3:]
        self.last_partial = hypothesis
        if len(self._hypotheses) >= 2:
            stable = self._common_prefix(self._hypotheses[-2], self._hypotheses[-1])
            if stable:
                self.last_stable = stable
        return hypothesis

    def _reset(self) -> None:
        self.buffer = torch.empty(0, dtype=torch.float32)
        self.pending = torch.empty(0, dtype=torch.float32)
        self.speech_active = False
        self.silence_chunks = 0
        self.last_partial = ""
        self.last_stable = ""
        self._last_decode_samples = 0
        self._hypotheses.clear()

    def push(self, audio: torch.Tensor, *, sample_rate: int | None = None) -> dict:
        waveform = self._resample(audio, int(sample_rate or self.sample_rate))
        rms = self._rms(waveform)
        active = rms >= self.vad_threshold
        final_text = ""
        final_audio = None

        if active:
            if not self.speech_active:
                self.speech_active = True
                self.silence_chunks = 0
            self.silence_chunks = 0
            self.buffer = torch.cat([self.buffer, waveform.cpu()])[-self.max_utterance_samples:]
            if self.buffer.numel() - self._last_decode_samples >= self.update_samples:
                self._last_decode_samples = int(self.buffer.numel())
                self._decode_partial()
        elif self.speech_active:
            self.silence_chunks += 1
            if self.silence_chunks >= self.end_silence_chunks:
                final_audio = self.buffer.clone()
                if final_audio.numel() >= self.min_speech_samples and self.recognizer.ready:
                    final_text = self.recognizer.transcribe(final_audio, sample_rate=self.sample_rate).strip()
                self._reset()

        return {
            "speech_active": bool(self.speech_active),
            "rms": rms,
            "partial": self.last_partial,
            "stable": self.last_stable,
            "final": final_text,
            "audio": final_audio,
        }

    def flush(self) -> dict:
        if not self.speech_active:
            return {"speech_active": False, "rms": 0.0, "partial": "", "stable": "", "final": "", "audio": None}
        audio = self.buffer.clone()
        text = self.recognizer.transcribe(audio, sample_rate=self.sample_rate).strip() if self.recognizer.ready and audio.numel() >= self.min_speech_samples else ""
        self._reset()
        return {"speech_active": False, "rms": 0.0, "partial": "", "stable": "", "final": text, "audio": audio}
