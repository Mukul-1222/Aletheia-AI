import json

import numpy as np
import torch

from aletheia.config import load_config
from aletheia.perception import (
    PerceptionSystem,
    VisionEncoder,
    VisionObjectDiscovery,
    VisualEntityMemory,
    VisualObjectTracker,
)


def _features(batch=2):
    torch.manual_seed(7)
    return torch.randn(batch, 128, 8, 8)


def test_spatial_vision_features_preserve_geometry():
    encoder = VisionEncoder(embedding_dim=32)
    image = torch.rand(2, 3, 128, 128)
    spatial = encoder.feature_map(image)
    assert spatial.shape == (2, 128, 8, 8)
    assert torch.isfinite(spatial).all()


def test_object_discovery_returns_slots_masks_boxes_and_scene_embedding():
    module = VisionObjectDiscovery(feature_dim=128, slot_dim=32, num_slots=4, iterations=2, output_dim=32, max_objects=4)
    result = module(_features())
    assert result["slots"].shape == (2, 4, 32)
    assert result["attention"].shape == (2, 4, 64)
    assert result["boxes"].shape == (2, 4, 4)
    assert result["scene_embedding"].shape == (2, 32)
    assert torch.allclose(result["attention"].sum(dim=1), torch.ones(2, 64), atol=1e-5)
    assert torch.isfinite(result["scene_embedding"]).all()
    assert torch.all((result["boxes"] >= 0.0) & (result["boxes"] <= 1.0))


def test_object_discovery_training_loss_backpropagates():
    module = VisionObjectDiscovery(feature_dim=128, slot_dim=32, num_slots=4, iterations=2, output_dim=32)
    features = _features()
    loss, parts = module.training_loss(features)
    assert torch.isfinite(loss)
    assert set(parts) == {"reconstruction", "diversity", "usage_balance"}
    loss.backward()
    assert module.slot_seed.grad is not None
    assert module.to_q.weight.grad is not None


def test_tracker_preserves_identity_across_small_motion():
    tracker = VisualObjectTracker(max_tracks=8, match_threshold=0.50, max_age=2, ema=0.25)
    first = tracker.update([{"bbox": [0.1, 0.1, 0.4, 0.5], "embedding": [1.0, 0.0], "score": 0.5}])
    second = tracker.update([{"bbox": [0.12, 0.11, 0.42, 0.51], "embedding": [0.98, 0.02], "score": 0.5}])
    assert first[0]["track_id"] == second[0]["track_id"]
    assert second[0]["matched"] is True
    assert second[0]["track_hits"] >= 2


def test_tracker_expires_stale_tracks():
    tracker = VisualObjectTracker(max_tracks=8, match_threshold=0.50, max_age=1)
    first = tracker.update([{"bbox": [0.1, 0.1, 0.4, 0.5], "embedding": [1.0, 0.0], "score": 0.5}])
    old_id = first[0]["track_id"]
    tracker.update([])
    tracker.update([])
    assert old_id not in tracker.tracks


def test_visual_entity_memory_persists_learned_representation(tmp_path):
    path = tmp_path / "visual_entity_memory.json"
    memory = VisualEntityMemory(path, threshold=0.75, ema=0.2)
    first = memory.observe(np.array([1.0, 0.0, 0.0], dtype=np.float32), track_id="track_1", bbox=[0.1, 0.2, 0.3, 0.4])
    second = memory.observe(np.array([0.99, 0.01, 0.0], dtype=np.float32), track_id="track_2", bbox=[0.11, 0.2, 0.31, 0.41])
    assert first["entity_id"] == second["entity_id"]
    memory.flush()
    restored = VisualEntityMemory(path, threshold=0.75)
    recognized = restored.recognize(np.array([1.0, 0.0, 0.0], dtype=np.float32))
    assert recognized["entity_id"] == first["entity_id"]
    assert restored.summary()["entities"] == 1


def test_perception_observation_exposes_learned_objects_without_external_detector(tmp_path):
    perception = PerceptionSystem(
        tmp_path,
        torch.device("cpu"),
        embedding_dim=32,
        batch_size=2,
        buffer_size=8,
        object_slots=3,
        object_slot_dim=24,
        object_iterations=1,
        object_max=3,
    )
    image = torch.rand(3, 128, 128)
    result = perception.observe_image(image, train=False)
    assert result["object_count"] > 0
    assert len(result["objects"]) == result["object_count"]
    assert perception.last_scene_embedding is not None
    assert perception.last_scene_embedding.shape[-1] == 32
    assert all("entity_id" in obj and "track_id" in obj and "bbox" in obj for obj in result["objects"])
    perception.flush()


def test_final19_runtime_config_exists():
    cfg = load_config()
    assert cfg.runtime.vision_object_slots >= 1
    assert cfg.runtime.vision_object_slot_dim >= 8
    assert cfg.runtime.vision_tracker_max_tracks >= 1
    assert 0.0 < cfg.runtime.vision_tracker_match_threshold < 1.0
    assert 0.0 < cfg.runtime.vision_entity_threshold < 1.0


def test_visual_entity_file_contains_no_raw_images(tmp_path):
    path = tmp_path / "visual_entity_memory.json"
    memory = VisualEntityMemory(path)
    memory.observe(np.array([0.0, 1.0], dtype=np.float32), track_id="track_1")
    memory.flush()
    payload = json.loads(path.read_text(encoding="utf-8"))
    serialized = json.dumps(payload)
    assert "base64" not in serialized.lower()
    assert "jpeg" not in serialized.lower()
    assert "png" not in serialized.lower()


def test_visual_entity_memory_caps_population(tmp_path):
    path = tmp_path / "visual_entity_memory.json"
    memory = VisualEntityMemory(path, threshold=0.99, max_entities=2)
    for idx in range(4):
        vector = np.zeros(4, dtype=np.float32)
        vector[idx] = 1.0
        memory.observe(vector, track_id=f"track_{idx}")
    assert len(memory.entities) == 2
