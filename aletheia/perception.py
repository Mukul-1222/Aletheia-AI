from __future__ import annotations

import hashlib
import base64
import io
import json
import math
import time
import wave
from collections import deque
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class VisionEncoder(nn.Module):
    """Randomly initialized visual encoder; learned by self-supervised experience."""

    def __init__(self, embedding_dim: int = 256):
        super().__init__()
        self.embedding_dim = int(embedding_dim)
        self.backbone = nn.Sequential(
            nn.Conv2d(3, 32, 5, stride=2, padding=2), nn.GELU(),
            nn.Conv2d(32, 64, 5, stride=2, padding=2), nn.GELU(),
            nn.Conv2d(64, 96, 3, stride=2, padding=1), nn.GELU(),
            nn.Conv2d(96, 128, 3, stride=2, padding=1), nn.GELU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.projector = nn.Sequential(nn.Flatten(), nn.Linear(128, embedding_dim), nn.LayerNorm(embedding_dim))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, nonlinearity="linear")
                nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.normal_(module.bias, mean=0.0, std=0.02)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def feature_map(self, images: torch.Tensor) -> torch.Tensor:
        """Return the spatial feature field before global pooling."""
        x = images
        for module in self.backbone:
            if isinstance(module, nn.AdaptiveAvgPool2d):
                break
            x = module(x)
        return x

    def encode_raw(self, images: torch.Tensor) -> torch.Tensor:
        return self.projector(self.backbone(images))

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.encode_raw(images), dim=-1)

    @staticmethod
    def prepare_image(value: Any, size: int = 128) -> torch.Tensor:
        if isinstance(value, (str, Path)):
            value = VisionEncoder.load_image(value)
        if isinstance(value, np.ndarray):
            image = value
            if image.ndim == 2:
                image = np.repeat(image[..., None], 3, axis=-1)
            if image.ndim != 3 or image.shape[-1] not in (3, 4):
                raise ValueError("expected an HxWx3/4 image array")
            if image.shape[-1] == 4:
                image = image[..., :3]
            image = image.astype(np.float32)
            if image.max() > 1.5:
                image /= 255.0
            tensor = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0)
        elif torch.is_tensor(value):
            tensor = value.detach().float()
            if tensor.ndim == 3:
                tensor = tensor.unsqueeze(0)
            if tensor.ndim != 4:
                raise ValueError("expected an image tensor with shape [C,H,W] or [B,C,H,W]")
            if tensor.size(1) == 1:
                tensor = tensor.repeat(1, 3, 1, 1)
            if tensor.size(1) != 3:
                raise ValueError("expected three image channels")
            if float(tensor.max()) > 1.5:
                tensor = tensor / 255.0
        else:
            raise TypeError("image must be a path, numpy array, or torch tensor")
        return F.interpolate(tensor, (size, size), mode="bilinear", align_corners=False).clamp(0.0, 1.0)

    @staticmethod
    def load_image(path: str | Path) -> np.ndarray:
        path = Path(path)
        try:
            from PIL import Image
            return np.asarray(Image.open(path).convert("RGB"))
        except ImportError:
            pass
        try:
            import cv2
        except ImportError as exc:
            raise RuntimeError("install pillow or opencv-python to load images") from exc
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(path)
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


class VisionObjectDiscovery(nn.Module):
    """From-scratch slot-style spatial decomposition over visual feature maps.

    The module is deliberately class-free: it does not know what an object is. It learns
    a small set of latent slots that jointly reconstruct the visual feature field. Spatial
    masks, latent slot vectors, and geometry are then exposed to the temporal tracker.
    """

    def __init__(
        self,
        feature_dim: int = 128,
        slot_dim: int = 128,
        num_slots: int = 6,
        iterations: int = 3,
        output_dim: int = 256,
        max_objects: int | None = None,
        min_mass: float = 0.02,
    ):
        super().__init__()
        self.feature_dim = int(feature_dim)
        self.slot_dim = int(slot_dim)
        self.num_slots = max(1, int(num_slots))
        self.iterations = max(1, int(iterations))
        self.output_dim = int(output_dim)
        self.max_objects = max(1, int(max_objects or num_slots))
        self.min_mass = max(0.0, float(min_mass))
        self.feature_norm = nn.LayerNorm(self.feature_dim)
        self.feature_proj = nn.Linear(self.feature_dim, self.slot_dim)
        self.position_proj = nn.Linear(2, self.slot_dim)
        self.query_norm = nn.LayerNorm(self.slot_dim)
        self.key_norm = nn.LayerNorm(self.slot_dim)
        self.value_norm = nn.LayerNorm(self.slot_dim)
        self.to_q = nn.Linear(self.slot_dim, self.slot_dim, bias=False)
        self.to_k = nn.Linear(self.slot_dim, self.slot_dim, bias=False)
        self.to_v = nn.Linear(self.slot_dim, self.slot_dim, bias=False)
        self.gru = nn.GRUCell(self.slot_dim, self.slot_dim)
        self.slot_mlp = nn.Sequential(
            nn.LayerNorm(self.slot_dim),
            nn.Linear(self.slot_dim, self.slot_dim * 2),
            nn.GELU(),
            nn.Linear(self.slot_dim * 2, self.slot_dim),
        )
        self.slot_seed = nn.Parameter(torch.randn(1, self.num_slots, self.slot_dim) * 0.05)
        self.slot_to_feature = nn.Linear(self.slot_dim, self.feature_dim)
        self.scene_projector = nn.Sequential(
            nn.LayerNorm(self.slot_dim),
            nn.Linear(self.slot_dim, self.output_dim),
            nn.LayerNorm(self.output_dim),
        )
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        nn.init.normal_(self.slot_seed, mean=0.0, std=0.05)

    @staticmethod
    def _coordinates(height: int, width: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        y = torch.linspace(-1.0, 1.0, height, device=device, dtype=dtype)
        x = torch.linspace(-1.0, 1.0, width, device=device, dtype=dtype)
        yy, xx = torch.meshgrid(y, x, indexing="ij")
        return torch.stack((xx, yy), dim=-1).reshape(1, height * width, 2)

    def _attend(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch, channels, height, width = features.shape
        tokens = features.flatten(2).transpose(1, 2)
        tokens = self.feature_norm(tokens)
        tokens = self.feature_proj(tokens)
        tokens = tokens + self.position_proj(self._coordinates(height, width, features.device, features.dtype))
        keys = self.to_k(self.key_norm(tokens))
        values = self.to_v(self.value_norm(tokens))
        slots = self.slot_seed.expand(batch, -1, -1).contiguous()
        scale = float(self.slot_dim) ** -0.5
        attention = None
        for _ in range(self.iterations):
            queries = self.to_q(self.query_norm(slots))
            logits = torch.einsum("bkd,bnd->bkn", queries, keys) * scale
            attention = torch.softmax(logits, dim=1)
            updates = torch.einsum("bkn,bnd->bkd", attention, values)
            flat_slots = slots.reshape(batch * self.num_slots, self.slot_dim)
            flat_updates = updates.reshape(batch * self.num_slots, self.slot_dim)
            slots = self.gru(flat_updates, flat_slots).reshape(batch, self.num_slots, self.slot_dim)
            slots = slots + self.slot_mlp(slots)
        assert attention is not None
        return slots, attention

    @staticmethod
    def _boxes(attention: torch.Tensor, height: int, width: int) -> torch.Tensor:
        # Geometry is derived from the learned spatial mask; no semantic detector is involved.
        batch, slots, _ = attention.shape
        coords = torch.stack(
            torch.meshgrid(
                torch.arange(width, device=attention.device, dtype=attention.dtype),
                torch.arange(height, device=attention.device, dtype=attention.dtype),
                indexing="xy",
            ), dim=-1,
        ).reshape(-1, 2)
        weights = attention / attention.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        center = torch.einsum("bkn,nc->bkc", weights, coords)
        delta = coords[None, None] - center[..., None, :]
        variance = torch.einsum("bkn,bknc->bkc", weights, delta.square())
        extent = 2.5 * torch.sqrt(variance + 1e-4)
        half = extent.clamp_min(1.0)
        mins = (center - half).clamp_min(0.0)
        maxs = (center + half).clamp(max=torch.tensor([width - 1, height - 1], device=attention.device, dtype=attention.dtype))
        # xyxy normalized to [0,1].
        return torch.stack((mins[..., 0] / max(1, width - 1), mins[..., 1] / max(1, height - 1),
                            maxs[..., 0] / max(1, width - 1), maxs[..., 1] / max(1, height - 1)), dim=-1)

    def forward(self, features: torch.Tensor) -> dict[str, torch.Tensor]:
        slots, attention = self._attend(features)
        batch, _, height, width = features.shape
        slot_features = self.slot_to_feature(slots)
        recon = torch.einsum("bkn,bkf->bnf", attention, slot_features)
        target = features.flatten(2).transpose(1, 2).detach()
        scene_weights = attention.mean(dim=-1)
        scene = torch.einsum("bk,bkd->bd", scene_weights, slots)
        scene_embedding = F.normalize(self.scene_projector(scene), dim=-1)
        mass = attention.mean(dim=-1)
        entropy = -(attention.clamp_min(1e-8) * attention.clamp_min(1e-8).log()).sum(dim=-1)
        entropy_norm = entropy / math.log(max(2, self.num_slots))
        concentration = (1.0 - entropy_norm).clamp(0.0, 1.0)
        scores = mass * (0.5 + 0.5 * concentration)
        boxes = self._boxes(attention, height, width)
        return {
            "slots": slots,
            "attention": attention,
            "slot_features": slot_features,
            "reconstruction": recon,
            "target": target,
            "scene_embedding": scene_embedding,
            "mass": mass,
            "concentration": concentration,
            "scores": scores,
            "boxes": boxes,
            "height": torch.tensor(height, device=features.device),
            "width": torch.tensor(width, device=features.device),
        }

    def training_loss(self, features: torch.Tensor) -> tuple[torch.Tensor, dict[str, float]]:
        result = self.forward(features)
        reconstruction = F.smooth_l1_loss(result["reconstruction"], result["target"])
        slots = F.normalize(result["slots"], dim=-1)
        gram = torch.matmul(slots, slots.transpose(1, 2))
        eye = torch.eye(self.num_slots, device=features.device, dtype=features.dtype).unsqueeze(0)
        diversity = ((gram - eye) ** 2).mean()
        usage = result["mass"].clamp_min(1e-6)
        usage_balance = F.mse_loss(usage, torch.full_like(usage, 1.0 / self.num_slots))
        loss = reconstruction + 0.25 * diversity + 0.05 * usage_balance
        return loss, {
            "reconstruction": float(reconstruction.detach().cpu()),
            "diversity": float(diversity.detach().cpu()),
            "usage_balance": float(usage_balance.detach().cpu()),
        }

    @torch.no_grad()
    def discover(self, features: torch.Tensor) -> dict[str, torch.Tensor]:
        self.eval()
        result = self.forward(features)
        slots = result["slots"]
        scores = result["scores"]
        batch = features.size(0)
        objects = []
        for b in range(batch):
            order = torch.argsort(scores[b], descending=True)
            selected = [int(idx) for idx in order.tolist() if float(scores[b, idx]) >= self.min_mass][:self.max_objects]
            objects.append(selected)
        result["selected_indices"] = objects  # metadata only; tensors remain authoritative above.
        return result


class VisualObjectTracker:
    """Temporal association over learned object embeddings and geometry."""

    def __init__(self, max_tracks: int = 32, match_threshold: float = 0.55, max_age: int = 6, ema: float = 0.25):
        self.max_tracks = max(1, int(max_tracks))
        self.match_threshold = float(match_threshold)
        self.max_age = max(0, int(max_age))
        self.ema = min(1.0, max(0.0, float(ema)))
        self.tracks: dict[str, dict[str, Any]] = {}
        self._next_id = 1

    @staticmethod
    def _cosine(a: np.ndarray, b: np.ndarray) -> float:
        denom = float(np.linalg.norm(a) * np.linalg.norm(b))
        return float(np.dot(a, b) / denom) if denom else 0.0

    @staticmethod
    def _iou(a: np.ndarray, b: np.ndarray) -> float:
        ax1, ay1, ax2, ay2 = a.tolist()
        bx1, by1, bx2, by2 = b.tolist()
        ix1, iy1 = max(ax1, bx1), max(ay1, by1)
        ix2, iy2 = min(ax2, bx2), min(ay2, by2)
        inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
        area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
        area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
        union = area_a + area_b - inter
        return float(inter / union) if union > 1e-8 else 0.0

    def reset(self) -> None:
        self.tracks.clear()
        self._next_id = 1

    def update(self, objects: list[dict[str, Any]], *, timestamp: float | None = None) -> list[dict[str, Any]]:
        now = float(timestamp if timestamp is not None else time.time())
        candidates: list[tuple[float, str, int]] = []
        for track_id, track in self.tracks.items():
            old = np.asarray(track["embedding"], dtype=np.float32)
            old_box = np.asarray(track["bbox"], dtype=np.float32)
            for index, obj in enumerate(objects):
                embedding = np.asarray(obj["embedding"], dtype=np.float32)
                box = np.asarray(obj["bbox"], dtype=np.float32)
                cosine = self._cosine(old, embedding)
                iou = self._iou(old_box, box)
                score = 0.75 * cosine + 0.25 * iou
                candidates.append((score, track_id, index))
        candidates.sort(reverse=True)
        matched_tracks: set[str] = set()
        matched_objects: set[int] = set()
        assignments: dict[int, str] = {}
        for score, track_id, index in candidates:
            if score < self.match_threshold or track_id in matched_tracks or index in matched_objects:
                continue
            matched_tracks.add(track_id)
            matched_objects.add(index)
            assignments[index] = track_id

        for track_id, track in list(self.tracks.items()):
            if track_id not in matched_tracks:
                track["age"] = int(track.get("age", 0)) + 1
                if track["age"] > self.max_age:
                    del self.tracks[track_id]

        results: list[dict[str, Any]] = []
        for index, obj in enumerate(objects):
            embedding = np.asarray(obj["embedding"], dtype=np.float32)
            box = np.asarray(obj["bbox"], dtype=np.float32)
            track_id = assignments.get(index)
            matched = track_id is not None
            if track_id is None:
                if len(self.tracks) >= self.max_tracks:
                    oldest = min(self.tracks.items(), key=lambda item: item[1].get("last_seen", 0.0))[0]
                    self.tracks.pop(oldest, None)
                track_id = f"track_{self._next_id:06d}"
                self._next_id += 1
                self.tracks[track_id] = {
                    "embedding": embedding.tolist(), "bbox": box.tolist(), "age": 0,
                    "hits": 1, "last_seen": now,
                }
            else:
                track = self.tracks[track_id]
                old_embedding = np.asarray(track["embedding"], dtype=np.float32)
                blended = (1.0 - self.ema) * old_embedding + self.ema * embedding
                norm = np.linalg.norm(blended)
                if norm:
                    blended /= norm
                track["embedding"] = blended.tolist()
                track["bbox"] = box.tolist()
                track["age"] = 0
                track["hits"] = int(track.get("hits", 1)) + 1
                track["last_seen"] = now
            results.append({**obj, "track_id": track_id, "track_age": int(self.tracks[track_id].get("age", 0)), "track_hits": int(self.tracks[track_id].get("hits", 1)), "matched": matched})
        return results


class VisualEntityMemory:
    """Persistent learned visual entity representations with no semantic class labels."""

    def __init__(self, path: Path, threshold: float = 0.84, ema: float = 0.15, max_tracks_per_entity: int = 32, max_entities: int = 5000):
        self.path = path
        self.threshold = float(threshold)
        self.ema = min(1.0, max(0.0, float(ema)))
        self.max_tracks_per_entity = max(1, int(max_tracks_per_entity))
        self.max_entities = max(1, int(max_entities))
        self.entities: list[dict[str, Any]] = []
        self._dirty = False
        self._load()

    def _load(self) -> None:
        if not self.path.exists():
            return
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
            self.entities = list(payload.get("entities", []))
        except (OSError, ValueError, TypeError):
            self.entities = []

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps({"entities": self.entities}, ensure_ascii=False, indent=2), encoding="utf-8")
        tmp.replace(self.path)
        self._dirty = False

    def flush(self) -> None:
        if self._dirty:
            self._save()

    @staticmethod
    def _cosine(a: np.ndarray, b: np.ndarray) -> float:
        denom = float(np.linalg.norm(a) * np.linalg.norm(b))
        return float(np.dot(a, b) / denom) if denom else 0.0

    def observe(self, embedding: torch.Tensor | np.ndarray, *, track_id: str = "", bbox: Iterable[float] | None = None, alias: str = "") -> dict[str, Any]:
        vector = np.asarray(embedding.detach().float().cpu().numpy() if torch.is_tensor(embedding) else embedding, dtype=np.float32).reshape(-1)
        best, best_score = None, -1.0
        for entity in self.entities:
            score = self._cosine(vector, np.asarray(entity.get("centroid", []), dtype=np.float32))
            if score > best_score:
                best, best_score = entity, score
        created = best is None or best_score < self.threshold
        now = time.time()
        if created:
            entity_id = f"ventity_{len(self.entities) + 1:06d}"
            entity = {
                "entity_id": entity_id, "centroid": vector.tolist(), "count": 1,
                "aliases": [alias] if alias else [], "tracks": [track_id] if track_id else [],
                "last_bbox": list(map(float, bbox)) if bbox is not None else None, "last_seen": now,
            }
            self.entities.append(entity)
            if len(self.entities) > self.max_entities:
                candidates = [index for index, item in enumerate(self.entities) if item.get("entity_id") != entity_id]
                ranked = sorted(
                    candidates,
                    key=lambda index: (int(self.entities[index].get("count", 0)), float(self.entities[index].get("last_seen", 0.0))),
                )
                remove = ranked[: max(1, len(self.entities) - self.max_entities)]
                for index in sorted(remove, reverse=True):
                    self.entities.pop(index)
        else:
            entity_id = str(best["entity_id"])
            old = np.asarray(best["centroid"], dtype=np.float32)
            blended = (1.0 - self.ema) * old + self.ema * vector
            norm = np.linalg.norm(blended)
            if norm:
                blended /= norm
            best["centroid"] = blended.tolist()
            best["count"] = int(best.get("count", 0)) + 1
            if alias and alias not in best.setdefault("aliases", []):
                best["aliases"].append(alias)
            if track_id:
                tracks = best.setdefault("tracks", [])
                if track_id not in tracks:
                    tracks.append(track_id)
                    if len(tracks) > self.max_tracks_per_entity:
                        del tracks[:-self.max_tracks_per_entity]
            best["last_bbox"] = list(map(float, bbox)) if bbox is not None else best.get("last_bbox")
            best["last_seen"] = now
            entity = best
        self._dirty = True
        return {"entity_id": str(entity["entity_id"]), "score": float(best_score if best is not None else 1.0), "new": bool(created), "count": int(entity.get("count", 1)), "aliases": list(entity.get("aliases", [])), "track_id": track_id}

    def recognize(self, embedding: torch.Tensor | np.ndarray) -> dict[str, Any] | None:
        vector = np.asarray(embedding.detach().float().cpu().numpy() if torch.is_tensor(embedding) else embedding, dtype=np.float32).reshape(-1)
        best, best_score = None, -1.0
        for entity in self.entities:
            score = self._cosine(vector, np.asarray(entity.get("centroid", []), dtype=np.float32))
            if score > best_score:
                best, best_score = entity, score
        if best is None:
            return None
        return {"entity_id": str(best["entity_id"]), "score": float(best_score), "recognized": bool(best_score >= self.threshold), "aliases": list(best.get("aliases", [])), "count": int(best.get("count", 0))}

    def summary(self) -> dict[str, Any]:
        return {"entities": len(self.entities), "observations": int(sum(int(e.get("count", 0)) for e in self.entities))}


class AudioEncoder(nn.Module):
    """Randomly initialized waveform representation; trained self-supervised."""

    def __init__(self, embedding_dim: int = 256, sample_rate: int = 16000):
        super().__init__()
        self.embedding_dim = int(embedding_dim)
        self.sample_rate = int(sample_rate)
        self.backbone = nn.Sequential(
            nn.Conv2d(1, 32, 5, stride=2, padding=2), nn.GELU(),
            nn.Conv2d(32, 64, 5, stride=2, padding=2), nn.GELU(),
            nn.Conv2d(64, 96, 3, stride=2, padding=1), nn.GELU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.projector = nn.Sequential(nn.Flatten(), nn.Linear(96, embedding_dim), nn.LayerNorm(embedding_dim))
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, nonlinearity="linear")
                nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.normal_(module.bias, mean=0.0, std=0.02)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def _spectrogram(self, waveform: torch.Tensor) -> torch.Tensor:
        waveform = waveform.float().reshape(-1)
        if waveform.numel() < 256:
            waveform = F.pad(waveform, (0, 256 - waveform.numel()))
        window = torch.hann_window(256, device=waveform.device)
        spec = torch.stft(waveform, n_fft=256, hop_length=128, win_length=256, window=window, return_complex=True)
        return torch.log1p(spec.abs()).unsqueeze(0).unsqueeze(0)

    def encode_raw(self, waveform: torch.Tensor) -> torch.Tensor:
        if waveform.ndim == 1:
            waveform = waveform.unsqueeze(0)
        if waveform.ndim == 2:
            specs = [self._spectrogram(item)[0] for item in waveform]
            spec = torch.stack(specs, dim=0)
        else:
            spec = self._spectrogram(waveform.reshape(-1))
        return self.projector(self.backbone(spec))

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.encode_raw(waveform), dim=-1)

    @staticmethod
    def load_wav(path: str | Path) -> tuple[torch.Tensor, int]:
        with wave.open(str(path), "rb") as handle:
            rate = handle.getframerate()
            channels = handle.getnchannels()
            width = handle.getsampwidth()
            frames = handle.readframes(handle.getnframes())
        if width == 1:
            values = torch.frombuffer(bytearray(frames), dtype=torch.uint8).float()
            values = (values - 128.0) / 128.0
        elif width == 2:
            values = torch.frombuffer(bytearray(frames), dtype=torch.int16).float() / 32768.0
        elif width == 3:
            raw = np.frombuffer(frames, dtype=np.uint8)
            if raw.size % 3:
                raw = raw[: raw.size - (raw.size % 3)]
            packed = raw.reshape(-1, 3).astype(np.int32)
            values_np = packed[:, 0] | (packed[:, 1] << 8) | (packed[:, 2] << 16)
            values_np = np.where(values_np & 0x800000, values_np - 0x1000000, values_np)
            values = torch.from_numpy(values_np.astype(np.float32)) / 8388608.0
        elif width == 4:
            values = torch.frombuffer(bytearray(frames), dtype=torch.int32).float() / 2147483648.0
        else:
            raise ValueError(f"unsupported PCM WAV sample width: {width} bytes")
        if channels > 1:
            values = values[: values.numel() - (values.numel() % channels)]
            values = values.view(-1, channels).mean(dim=1)
        return values.clamp(-1.0, 1.0), rate


class PerceptualIdentityMemory:
    """Unsupervised persistent entity clusters over learned sensory embeddings."""

    def __init__(self, path: Path, threshold: float = 0.92, max_exemplars: int = 4):
        self.path = path
        self.threshold = float(threshold)
        self.max_exemplars = max(1, int(max_exemplars))
        self.entities: list[dict] = []
        self._dirty = False
        self._load()

    def _load(self) -> None:
        if not self.path.exists():
            return
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
            self.entities = list(payload.get("entities", []))
            for entity in self.entities:
                normalized = []
                for exemplar in entity.get("exemplars", []):
                    if isinstance(exemplar, str):
                        normalized.append({"data": exemplar, "pinned": False, "created": 0.0})
                    elif isinstance(exemplar, dict) and exemplar.get("data"):
                        normalized.append({"data": str(exemplar["data"]), "pinned": bool(exemplar.get("pinned", False)), "created": float(exemplar.get("created", 0.0))})
                entity["exemplars"] = normalized
        except (OSError, ValueError, TypeError):
            self.entities = []

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps({"entities": self.entities}, ensure_ascii=False, indent=2), encoding="utf-8")
        tmp.replace(self.path)
        self._dirty = False

    def flush(self) -> None:
        if self._dirty:
            self._save()

    @staticmethod
    def _cosine(a: np.ndarray, b: np.ndarray) -> float:
        denom = float(np.linalg.norm(a) * np.linalg.norm(b))
        return float(np.dot(a, b) / denom) if denom else 0.0

    def observe(self, embedding: torch.Tensor, *, alias: str = "") -> dict:
        vector = embedding.detach().float().cpu().numpy().reshape(-1)
        best_id, best_score = None, -1.0
        for entity in self.entities:
            centroid = np.asarray(entity["centroid"], dtype=np.float32)
            score = self._cosine(vector, centroid)
            if score > best_score:
                best_score = score
                best_id = entity["entity_id"]
        created = best_id is None or best_score < self.threshold
        if created:
            entity_id = f"entity_{len(self.entities) + 1:04d}"
            self.entities.append({"entity_id": entity_id, "centroid": vector.tolist(), "count": 1, "aliases": [alias] if alias else [], "exemplars": [], "updated": time.time()})
        else:
            entity = next(item for item in self.entities if item["entity_id"] == best_id)
            count = max(1, int(entity.get("count", 1)))
            centroid = np.asarray(entity["centroid"], dtype=np.float32)
            centroid = (centroid * count + vector) / float(count + 1)
            norm = np.linalg.norm(centroid)
            if norm:
                centroid /= norm
            entity["centroid"] = centroid.tolist()
            entity["count"] = count + 1
            if alias and alias not in entity.setdefault("aliases", []):
                entity["aliases"].append(alias)
            entity["updated"] = time.time()
            entity_id = entity["entity_id"]
        self._dirty = True
        return {"entity_id": entity_id, "score": float(best_score if best_id else 1.0), "new": created, "alias": alias, "aliases": [x for x in next(item for item in self.entities if item["entity_id"] == entity_id).get("aliases", [])]}

    @staticmethod
    def _encode_exemplar(image: Any) -> str | None:
        try:
            from PIL import Image
            if isinstance(image, (str, Path)):
                pil = Image.open(image).convert("RGB")
            elif isinstance(image, np.ndarray):
                array = image
                if array.ndim == 2:
                    array = np.repeat(array[..., None], 3, axis=-1)
                if array.shape[-1] == 4:
                    array = array[..., :3]
                pil = Image.fromarray(array.astype(np.uint8), mode="RGB")
            elif torch.is_tensor(image):
                tensor = image.detach().cpu().float()
                if tensor.ndim == 4:
                    tensor = tensor[0]
                if tensor.ndim == 3 and tensor.shape[0] in (3, 4):
                    tensor = tensor[:3].permute(1, 2, 0)
                array = (tensor.clamp(0, 1).numpy() * 255.0).round().astype(np.uint8)
                pil = Image.fromarray(array, mode="RGB")
            else:
                return None
            pil.thumbnail((256, 256))
            buf = io.BytesIO()
            pil.save(buf, format="JPEG", quality=80, optimize=True)
            return base64.b64encode(buf.getvalue()).decode("ascii")
        except Exception:
            return None

    @staticmethod
    def _decode_exemplar(encoded: str) -> np.ndarray | None:
        try:
            from PIL import Image
            image = Image.open(io.BytesIO(base64.b64decode(encoded))).convert("RGB")
            return np.asarray(image)
        except Exception:
            return None

    @staticmethod
    def _exemplar_distance(a: np.ndarray, b: np.ndarray) -> float:
        from PIL import Image
        left = Image.fromarray(a).resize((32, 32))
        right = Image.fromarray(b).resize((32, 32))
        la = np.asarray(left, dtype=np.float32) / 255.0
        rb = np.asarray(right, dtype=np.float32) / 255.0
        return float(np.abs(la - rb).mean())

    def add_exemplar(self, entity_id: str, image: Any, *, pinned: bool = False, min_distance: float = 0.02) -> bool:
        encoded = self._encode_exemplar(image)
        if not encoded:
            return False
        entity = next((item for item in self.entities if item.get("entity_id") == entity_id), None)
        if entity is None:
            return False
        examples = entity.setdefault("exemplars", [])
        candidate = self._decode_exemplar(encoded)
        if candidate is None:
            return False
        for item in examples:
            old_encoded = item.get("data") if isinstance(item, dict) else item
            old = self._decode_exemplar(str(old_encoded)) if old_encoded else None
            if old is not None and self._exemplar_distance(candidate, old) < float(min_distance):
                if pinned and isinstance(item, dict) and not item.get("pinned"):
                    item["pinned"] = True
                    self._dirty = True
                return False
        examples.append({"data": encoded, "pinned": bool(pinned), "created": time.time()})
        if len(examples) > self.max_exemplars:
            pinned_items = [x for x in examples if isinstance(x, dict) and x.get("pinned")]
            unpinned = [x for x in examples if not (isinstance(x, dict) and x.get("pinned"))]
            keep_unpinned = max(0, self.max_exemplars - len(pinned_items))
            examples[:] = pinned_items + unpinned[-keep_unpinned:]
        self._dirty = True
        return True

    def refresh(self, encoder: nn.Module, device: torch.device) -> None:
        """Recompute centroids from stored visual exemplars after encoder training.

        This prevents identity drift when the learned visual representation changes.
        """
        changed = False
        for entity in self.entities:
            vectors = []
            for item in entity.get("exemplars", []):
                encoded = item.get("data") if isinstance(item, dict) else item
                array = self._decode_exemplar(str(encoded)) if encoded else None
                if array is None:
                    continue
                try:
                    tensor = VisionEncoder.prepare_image(array).to(device)
                    with torch.no_grad():
                        vectors.append(encoder(tensor)[0].detach().float().cpu().numpy())
                except Exception:
                    continue
            if vectors:
                centroid = np.mean(np.stack(vectors, axis=0), axis=0)
                norm = np.linalg.norm(centroid)
                if norm:
                    centroid /= norm
                entity["centroid"] = centroid.tolist()
                entity["updated"] = time.time()
                changed = True
        if changed:
            self._dirty = True

    def recognize(self, embedding: torch.Tensor) -> dict | None:
        vector = embedding.detach().float().cpu().numpy().reshape(-1)
        best, score = None, -1.0
        for entity in self.entities:
            current = self._cosine(vector, np.asarray(entity["centroid"], dtype=np.float32))
            if current > score:
                best, score = entity, current
        if best is None:
            return None
        return {"entity_id": best["entity_id"], "score": float(score), "recognized": bool(score >= self.threshold), "aliases": list(best.get("aliases", [])), "count": int(best.get("count", 0))}


class PerceptionSystem:
    """Self-supervised visual/audio learning without pretrained encoders."""

    def __init__(
        self,
        root: Path,
        device: torch.device,
        embedding_dim: int = 256,
        *,
        batch_size: int = 32,
        buffer_size: int = 256,
        learning_rate: float = 2e-4,
        min_visual_delta: float = 0.02,
        identity_threshold: float = 0.92,
        identity_max_exemplars: int = 4,
        identity_reindex_interval: int = 8,
        object_slots: int = 6,
        object_slot_dim: int = 128,
        object_iterations: int = 3,
        object_max: int = 6,
        object_min_mass: float = 0.02,
        tracker_max_tracks: int = 32,
        tracker_match_threshold: float = 0.55,
        tracker_max_age: int = 6,
        tracker_ema: float = 0.25,
        entity_threshold: float = 0.84,
        entity_ema: float = 0.15,
        entity_max_entities: int = 5000,
    ):
        self.root = root
        self.device = device
        self.vision = VisionEncoder(embedding_dim).to(device)
        self.audio = AudioEncoder(embedding_dim).to(device)
        self.identities = PerceptualIdentityMemory(
            root / "data" / "identity_memory.json",
            threshold=float(identity_threshold),
            max_exemplars=int(identity_max_exemplars),
        )
        self.object_discovery = VisionObjectDiscovery(
            feature_dim=128, slot_dim=int(object_slot_dim), num_slots=int(object_slots),
            iterations=int(object_iterations), output_dim=embedding_dim, max_objects=int(object_max),
            min_mass=float(object_min_mass),
        ).to(device)
        self.tracker = VisualObjectTracker(
            max_tracks=int(tracker_max_tracks), match_threshold=float(tracker_match_threshold),
            max_age=int(tracker_max_age), ema=float(tracker_ema),
        )
        self.visual_entities = VisualEntityMemory(
            root / "data" / "visual_entity_memory.json",
            threshold=float(entity_threshold), ema=float(entity_ema), max_entities=int(entity_max_entities),
        )
        self.vision_optimizer = torch.optim.AdamW(
            list(self.vision.parameters()) + list(self.object_discovery.parameters()),
            lr=float(learning_rate), weight_decay=1e-3,
        )
        self.audio_optimizer = torch.optim.AdamW(self.audio.parameters(), lr=float(learning_rate), weight_decay=1e-3)
        self.vision_steps = 0
        self.audio_steps = 0
        self.vision_batch_size = max(32, int(batch_size))
        self.audio_batch_size = max(32, int(batch_size))
        self.buffer_size = max(self.vision_batch_size, int(buffer_size))
        self._vision_buffer: deque[torch.Tensor] = deque(maxlen=self.buffer_size)
        self._audio_buffer: deque[torch.Tensor] = deque(maxlen=self.buffer_size)
        self._vision_keys: set[str] = set()
        self._audio_keys: set[str] = set()
        self._buffers_dirty = False
        self._buffer_persist_interval = 32
        self.min_visual_delta = max(0.0, float(min_visual_delta))
        self.identity_reindex_interval = max(1, int(identity_reindex_interval))
        self._observations_since_buffer_save = 0
        self.last_scene_embedding: torch.Tensor | None = None
        self.last_visual_objects: list[dict[str, Any]] = []
        self._load_buffers()

    @staticmethod
    def _vision_key(image: torch.Tensor) -> str:
        q = F.interpolate(image.unsqueeze(0), (32, 32), mode="bilinear", align_corners=False)[0]
        q = (q.clamp(0, 1) * 255.0).round().to(torch.uint8).cpu().numpy()
        return hashlib.sha256(q.tobytes()).hexdigest()

    def _vision_is_diverse(self, image: torch.Tensor) -> bool:
        if not self._vision_buffer:
            return True
        candidate = F.interpolate(image.unsqueeze(0), (16, 16), mode="bilinear", align_corners=False)[0]
        threshold = float(getattr(self, "min_visual_delta", 0.02))
        for existing in self._vision_buffer:
            other = existing.float() / 255.0
            other = F.interpolate(other.unsqueeze(0), (16, 16), mode="bilinear", align_corners=False)[0]
            channel_delta = (candidate - other).abs().mean(dim=(1, 2)).max()
            if float(channel_delta) < threshold:
                return False
        return True

    @staticmethod
    def _audio_key(waveform: torch.Tensor) -> str:
        x = waveform.float().reshape(-1)
        if x.numel() > 2048:
            x = F.interpolate(x.view(1, 1, -1), size=2048, mode="linear", align_corners=False).view(-1)
        q = (x.clamp(-1, 1) * 2047.0).round().to(torch.int16).cpu().numpy()
        return hashlib.sha256(q.tobytes()).hexdigest()

    def _load_buffers(self) -> None:
        path = self.root / "data" / "perception_buffers.pt"
        if not path.exists():
            return
        try:
            payload = torch.load(path, map_location="cpu", weights_only=True)
            visions = payload.get("vision", []) if isinstance(payload, dict) else []
            audios = payload.get("audio", []) if isinstance(payload, dict) else []
            for item in visions:
                tensor = torch.as_tensor(item, dtype=torch.uint8)
                if tuple(tensor.shape) == (3, 128, 128):
                    image = tensor.float() / 255.0
                    key = self._vision_key(image)
                    if key not in self._vision_keys and self._vision_is_diverse(image):
                        self._vision_buffer.append(tensor)
                        self._vision_keys.add(key)
            for item in audios:
                tensor = torch.as_tensor(item, dtype=torch.float32).reshape(-1)
                if tensor.numel() >= 256:
                    tensor = self._fixed_audio_window(tensor)
                    key = self._audio_key(tensor)
                    if key not in self._audio_keys:
                        self._audio_buffer.append(tensor)
                        self._audio_keys.add(key)
        except Exception:
            self._vision_buffer.clear()
            self._audio_buffer.clear()
            self._vision_keys.clear()
            self._audio_keys.clear()

    def save_buffers(self, *, force: bool = False) -> None:
        if not force and not self._buffers_dirty:
            self.identities.flush()
            self.visual_entities.flush()
            return
        path = self.root / "data" / "perception_buffers.pt"
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "vision": [item.contiguous().cpu() for item in self._vision_buffer],
            "audio": [item.float().cpu() for item in self._audio_buffer],
        }
        tmp = path.with_suffix(".tmp")
        torch.save(payload, tmp)
        tmp.replace(path)
        self._buffers_dirty = False
        self._observations_since_buffer_save = 0
        self.identities.flush()
        self.visual_entities.flush()

    def flush(self) -> None:
        self.save_buffers(force=True)

    def _add_vision_buffer(self, image: torch.Tensor) -> bool:
        key = self._vision_key(image)
        if key in self._vision_keys or not self._vision_is_diverse(image):
            return False
        if len(self._vision_buffer) >= self.buffer_size:
            old = self._vision_buffer.popleft()
            self._vision_keys.discard(self._vision_key(old.float() / 255.0))
        self._vision_buffer.append((image.clamp(0, 1) * 255.0).round().to(torch.uint8).cpu())
        self._vision_keys.add(key)
        self._buffers_dirty = True
        self._observations_since_buffer_save += 1
        return True

    def _add_audio_buffer(self, waveform: torch.Tensor) -> bool:
        waveform = self._fixed_audio_window(waveform).float().cpu()
        key = self._audio_key(waveform)
        if key in self._audio_keys:
            return False
        if len(self._audio_buffer) >= self.buffer_size:
            old = self._audio_buffer.popleft()
            self._audio_keys.discard(self._audio_key(old))
        self._audio_buffer.append(waveform)
        self._audio_keys.add(key)
        self._buffers_dirty = True
        self._observations_since_buffer_save += 1
        return True

    @staticmethod
    def _vision_augment(images: torch.Tensor) -> torch.Tensor:
        out = images.clone()
        if torch.rand(()) < 0.8:
            scale = float(torch.empty(1).uniform_(0.85, 1.0))
            size = max(8, int(out.shape[-1] * scale))
            top = int(torch.randint(0, out.shape[-2] - size + 1, (1,)))
            left = int(torch.randint(0, out.shape[-1] - size + 1, (1,)))
            out = out[..., top:top + size, left:left + size]
            out = F.interpolate(out, (128, 128), mode="bilinear", align_corners=False)
        gain = torch.empty(out.shape[0], 1, 1, 1, device=out.device).uniform_(0.8, 1.2)
        out = (out * gain).clamp(0.0, 1.0)
        noise = torch.randn_like(out) * 0.02
        return (out + noise).clamp(0.0, 1.0)

    @staticmethod
    def _covariance_loss(z: torch.Tensor) -> torch.Tensor:
        if z.size(0) < 2:
            return torch.zeros((), device=z.device, dtype=z.dtype)
        centered = z - z.mean(dim=0, keepdim=True)
        cov = centered.T @ centered / max(1, z.size(0) - 1)
        off_diag = cov - torch.diag_embed(torch.diagonal(cov))
        return off_diag.pow(2).mean()

    @classmethod
    def _ssl_loss(cls, z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
        """VICReg-style loss on unnormalized features to prevent representational collapse."""
        invariance = F.mse_loss(z1, z2)
        eps = 1e-4
        std1 = torch.sqrt(z1.var(dim=0, unbiased=False) + eps)
        std2 = torch.sqrt(z2.var(dim=0, unbiased=False) + eps)
        variance = F.relu(1.0 - std1).mean() + F.relu(1.0 - std2).mean()
        covariance = cls._covariance_loss(z1) + cls._covariance_loss(z2)
        return 25.0 * invariance + 25.0 * variance + covariance

    def train_visual_observations(self, images: Iterable[Any], *, steps: int = 8) -> dict:
        added = 0
        for item in images:
            try:
                prepared = VisionEncoder.prepare_image(item)[0]
                added += int(self._add_vision_buffer(prepared))
            except Exception:
                continue
        if len(self._vision_buffer) < self.vision_batch_size:
            return {
                "trained": False, "steps": self.vision_steps,
                "reason": "insufficient_distinct_images", "distinct_images": len(self._vision_buffer),
                "required_images": self.vision_batch_size, "added": added,
            }
        pool = list(self._vision_buffer)
        losses: list[float] = []
        discovery_losses: list[float] = []
        std_values: list[float] = []
        self.vision.train(); self.object_discovery.train()
        for _ in range(max(1, int(steps))):
            indices = torch.randperm(len(pool))[:self.vision_batch_size].tolist()
            batch = torch.stack([pool[i].float() / 255.0 for i in indices], dim=0).to(self.device)
            augmented = self._vision_augment(batch)
            z1 = self.vision.encode_raw(augmented)
            z2 = self.vision.encode_raw(self._vision_augment(batch))
            ssl = self._ssl_loss(z1, z2)
            features = self.vision.feature_map(augmented)
            discovery, parts = self.object_discovery.training_loss(features)
            loss = ssl + discovery
            self.vision_optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(list(self.vision.parameters()) + list(self.object_discovery.parameters()), 1.0)
            self.vision_optimizer.step()
            losses.append(float(ssl.detach().cpu()))
            discovery_losses.append(float(discovery.detach().cpu()))
            std_values.append(float(torch.cat([z1, z2], dim=0).detach().std(dim=0).mean().cpu()))
            self.vision_steps += 1
        self.vision.eval(); self.object_discovery.eval()
        self._buffers_dirty = True
        if self._observations_since_buffer_save >= self._buffer_persist_interval:
            self.save_buffers()
        return {
            "trained": True,
            "steps": self.vision_steps,
            "loss_before_after": [losses[0], losses[-1]],
            "discovery_loss_before_after": [discovery_losses[0], discovery_losses[-1]],
            "batch_size": self.vision_batch_size,
            "buffer_size": len(pool),
            "distinct_images": len(pool),
            "embedding_std_mean": std_values[-1],
            "added": added,
        }

    def _augment_audio(self, waveform: torch.Tensor) -> torch.Tensor:
        out = waveform.clone()
        if out.numel() > 256:
            max_shift = max(1, int(out.numel() * 0.03))
            shift = int(torch.randint(-max_shift, max_shift + 1, (1,)))
            out = torch.roll(out, shift, dims=0)
        out = out * float(torch.empty(1).uniform_(0.85, 1.15))
        out = out + torch.randn_like(out) * 0.002
        return out.clamp(-1.0, 1.0)

    @staticmethod
    def _fixed_audio_window(waveform: torch.Tensor, size: int = 16000) -> torch.Tensor:
        waveform = waveform.float().reshape(-1)
        if waveform.numel() >= size:
            return waveform[:size]
        return F.pad(waveform, (0, size - waveform.numel()))

    def train_audio_observations(self, waveforms: Iterable[torch.Tensor], *, steps: int = 8) -> dict:
        added=0
        for item in waveforms:
            if item is None or item.numel()==0: continue
            try: added += int(self._add_audio_buffer(self._fixed_audio_window(item)))
            except Exception: continue
        if len(self._audio_buffer) < self.audio_batch_size:
            return {"trained":False,"steps":self.audio_steps,"reason":"insufficient_distinct_audio","distinct_audio":len(self._audio_buffer),"required_audio":self.audio_batch_size,"added":added}
        pool=list(self._audio_buffer); losses=[]; std_values=[]; self.audio.train()
        for _ in range(max(1,int(steps))):
            indices=torch.randperm(len(pool))[:self.audio_batch_size].tolist()
            batch=torch.stack([pool[i] for i in indices],0).to(self.device)
            z1=self.audio.encode_raw(torch.stack([self._augment_audio(x) for x in batch],0)); z2=self.audio.encode_raw(torch.stack([self._augment_audio(x) for x in batch],0))
            loss=self._ssl_loss(z1,z2)
            self.audio_optimizer.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(self.audio.parameters(),1.0); self.audio_optimizer.step()
            losses.append(float(loss.detach().cpu())); std_values.append(float(torch.cat([z1,z2],0).detach().std(dim=0).mean().cpu())); self.audio_steps += 1
        self.audio.eval(); self._buffers_dirty=True
        if self._observations_since_buffer_save >= self._buffer_persist_interval: self.save_buffers()
        return {"trained":True,"steps":self.audio_steps,"loss_before_after":[losses[0],losses[-1]],"batch_size":self.audio_batch_size,"buffer_size":len(pool),"distinct_audio":len(pool),"embedding_std_mean":std_values[-1],"added":added}

    @torch.no_grad()
    def _discover_current_objects(self, tensor: torch.Tensor, *, alias: str = "") -> tuple[list[dict[str, Any]], torch.Tensor]:
        self.object_discovery.eval()
        features = self.vision.feature_map(tensor)
        result = self.object_discovery.discover(features)
        slot_embeddings = result["slots"]
        scores = result["scores"]
        boxes = result["boxes"]
        selections = result["selected_indices"][0]
        objects: list[dict[str, Any]] = []
        for slot_index in selections:
            embedding = F.normalize(slot_embeddings[0, slot_index], dim=-1)
            entity_score = float(scores[0, slot_index].cpu())
            bbox = [float(x) for x in boxes[0, slot_index].cpu().tolist()]
            objects.append({
                "slot_index": int(slot_index),
                "score": entity_score,
                "bbox": bbox,
                "embedding": embedding.detach().cpu().tolist(),
            })
        tracked = self.tracker.update(objects)
        for item in tracked:
            entity = self.visual_entities.observe(
                np.asarray(item["embedding"], dtype=np.float32),
                track_id=str(item["track_id"]), bbox=item["bbox"],
                alias=(alias if item is tracked[0] and alias else ""),
            )
            item.update({"entity_id": entity["entity_id"], "entity_score": entity["score"], "entity_new": entity["new"], "entity_count": entity["count"]})
            item.pop("embedding", None)
        self.last_scene_embedding = result["scene_embedding"][0].detach()
        self.last_visual_objects = tracked
        return tracked, self.last_scene_embedding

    def observe_image(self, image: Any, *, alias: str = "", train: bool = True) -> dict:
        if train:
            training_result = self.train_visual_observations([image], steps=1)
            if training_result.get("trained") and self.vision_steps % self.identity_reindex_interval == 0:
                self.identities.refresh(self.vision, self.device)
        tensor = VisionEncoder.prepare_image(image).to(self.device)
        self.vision.eval(); self.object_discovery.eval()
        embedding = self.vision(tensor)[0]
        observed = self.identities.observe(embedding, alias=alias)
        confident_match = (
            bool(observed.get("new")) or bool(str(alias).strip())
            or float(observed.get("score", 0.0)) >= self.identities.threshold + 0.03
        )
        if confident_match:
            self.identities.add_exemplar(
                observed["entity_id"], image,
                pinned=bool(str(alias).strip()),
                min_distance=max(0.01, float(self.min_visual_delta)),
            )
        tracked, scene_embedding = self._discover_current_objects(tensor, alias=alias)
        raw_hash = None
        if isinstance(image, (str, Path)):
            raw_hash = hashlib.sha256(Path(image).read_bytes()).hexdigest()
        return {
            "modality": "vision",
            "embedding_dim": int(scene_embedding.numel()),
            "content_hash": raw_hash,
            **observed,
            "objects": tracked,
            "object_count": len(tracked),
            "scene_embedding_dim": int(scene_embedding.numel()),
            "trained_steps": self.vision_steps,
        }

    @torch.no_grad()
    def recognize_image(self, image: Any) -> dict | None:
        self.vision.eval(); self.object_discovery.eval()
        tensor = VisionEncoder.prepare_image(image).to(self.device)
        embedding = self.vision(tensor)[0]
        return self.identities.recognize(embedding)

    def observe_audio(self, audio: torch.Tensor | str | Path, *, train: bool = True) -> dict:
        if isinstance(audio, (str, Path)):
            waveform, rate = AudioEncoder.load_wav(audio)
        else:
            waveform, rate = audio.float().reshape(-1).cpu(), self.audio.sample_rate
        if rate != self.audio.sample_rate:
            target = max(1, int(round(waveform.numel() * self.audio.sample_rate / rate)))
            waveform = F.interpolate(waveform.view(1, 1, -1), size=target, mode="linear", align_corners=False).view(-1)
            rate = self.audio.sample_rate
        if train:
            self.train_audio_observations([waveform], steps=1)
        self.audio.eval()
        embedding = self.audio(waveform.to(self.device))[0]
        return {"modality": "audio", "sample_rate": rate, "embedding_dim": int(embedding.numel()), "norm": float(embedding.norm().cpu()), "trained_steps": self.audio_steps}

    def camera_stream(self, *, device_index: int = 0, frames: int = 0, alias: str = "", show: bool = False) -> list[dict]:
        try:
            import cv2
        except ImportError as exc:
            raise RuntimeError("camera support requires opencv-python") from exc
        capture = cv2.VideoCapture(int(device_index))
        if not capture.isOpened():
            raise RuntimeError(f"unable to open camera device {device_index}")
        results = []
        training_buffer: list[np.ndarray] = []
        frame_budget = None if int(frames) <= 0 else int(frames)
        try:
            while frame_budget is None or frame_budget > 0:
                ok, frame = capture.read()
                if not ok:
                    break
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                training_buffer.append(rgb.copy())
                if len(training_buffer) >= self.vision_batch_size:
                    training_result = self.train_visual_observations(training_buffer, steps=1)
                    if training_result.get("trained"):
                        training_buffer.clear()
                    else:
                        training_buffer = training_buffer[-self.vision_batch_size:]
                result = self.observe_image(rgb, alias=alias, train=False)
                results.append(result)
                if show:
                    cv2.imshow("Aletheia", frame)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break
                if frame_budget is not None:
                    frame_budget -= 1
            if training_buffer:
                self.train_visual_observations(training_buffer, steps=1)
        finally:
            capture.release()
            self.save_buffers(force=True)
            if show:
                cv2.destroyAllWindows()
        return results
