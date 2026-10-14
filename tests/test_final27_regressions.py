from __future__ import annotations

import random
from pathlib import Path
import json

import pytest
from types import SimpleNamespace

import torch

from aletheia.agent import AletheiaAgent
from aletheia.config import load_config
from aletheia.curriculum import _read_bytes
from aletheia.eval_corpus import _read_exclusion_text
from aletheia.memory import ExperienceMemory
from aletheia.model import AletheiaModel
from aletheia.tokenizer import BASE_VOCAB_SIZE, LearnedBPETokenizer


def test_per_kind_retention_keeps_newest_rows(tmp_path: Path):
    memory = ExperienceMemory(tmp_path / "memory.sqlite3", limit=100, frontier_limit=10, kind_limits={"sensory_vision": 3})
    returned_ids = []
    for value in range(1, 6):
        returned_ids.append(memory.add(
            kind="sensory_vision", source="test", title=f"frame {value}", url="",
            text=f"frame payload {value}", merge_existing=False,
        ))
    rows = memory.db.execute(
        "SELECT id, title FROM memories WHERE kind='sensory_vision' ORDER BY id ASC"
    ).fetchall()
    assert [row["title"] for row in rows] == ["frame 3", "frame 4", "frame 5"]
    assert memory.db.execute("SELECT 1 FROM memories WHERE id=?", (returned_ids[-1],)).fetchone()
    memory.close()


def test_model_checkpoint_save_is_atomic_and_keeps_backup(tmp_path: Path):
    config = load_config(Path("config.json")).model
    config.dim = 32
    config.layers = 1
    config.heads = 4
    config.ffn = 64
    config.recurrent_dim = 48
    config.world_attention_heads = 4
    config.context = 32
    config.perception_dim = 16
    config.world_model_hidden = 32
    config.vocab_size = 259
    model = AletheiaModel(config, ["observe"])
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    path = tmp_path / "aletheia.pt"
    model.save(path, optimizer)
    with torch.no_grad():
        model.token_embedding.weight[7, 0].add_(1.0)
    model.save(path, optimizer)
    assert path.is_file() and path.stat().st_size > 0
    assert path.with_suffix(".pt.bak").is_file() and path.with_suffix(".pt.bak").stat().st_size > 0
    backup_payload = torch.load(path.with_suffix(".pt.bak"), map_location="cpu", weights_only=True)
    current_payload = torch.load(path, map_location="cpu", weights_only=True)
    assert backup_payload["state_dict"]["token_embedding.weight"][7, 0] != current_payload["state_dict"]["token_embedding.weight"][7, 0]


def test_token_chunks_only_mark_true_document_end(tmp_path: Path):
    agent = object.__new__(AletheiaAgent)
    agent.model = SimpleNamespace(cfg=SimpleNamespace(context=8))
    agent.device = torch.device("cpu")
    chunks = agent._encode_token_chunks(list(range(10)), context=8)
    assert len(chunks) == 2
    assert int(chunks[0][0, 0]) == 257
    assert int(chunks[0][0, -1]) != 258
    assert int(chunks[-1][0, -1]) == 258
    assert all(chunk.size(1) <= 8 for chunk in chunks)


def test_batched_trajectory_uses_last_nonpadding_hidden_state():
    cfg = load_config(Path("config.json")).model
    cfg.dim = 8
    cfg.layers = 1
    cfg.heads = 2
    cfg.ffn = 16
    cfg.recurrent_dim = 8
    cfg.world_attention_heads = 2
    cfg.context = 8
    cfg.perception_dim = 4
    cfg.world_model_hidden = 8
    cfg.vocab_size = 259
    model = AletheiaModel(cfg, ["observe"])
    captured = []

    def fake_encode(tokens, state=None):
        batch, width = tokens.shape
        positions = torch.arange(width, device=tokens.device, dtype=torch.float32).view(1, width, 1)
        hidden = positions.expand(batch, width, cfg.dim).contiguous()
        logits = torch.zeros(batch, width, cfg.vocab_size, device=tokens.device)
        return logits, hidden

    def fake_integrate(state, *, text_summary=None, **kwargs):
        captured.append(text_summary.detach().cpu().clone())
        return state, torch.zeros(state.size(0), 0, device=state.device)

    model.encode_sequence = fake_encode
    model.integrate_multimodal = fake_integrate
    model.predict_next_observation = lambda state, action, error: torch.zeros(1, cfg.dim, device=state.device)
    model.transition_uncertainty = lambda state: torch.zeros(state.size(0), device=state.device)
    model.world_prediction_loss = lambda prediction, target: torch.zeros((), device=target.device)
    model.update_state = lambda summary, state, action, error: state

    long = torch.tensor([[257, 10, 11, 258]], dtype=torch.long)
    short = torch.tensor([[257, 12, 258]], dtype=torch.long)
    model.trajectory_loss_batched([
        ([long], [None], [0.0]),
        ([short], [None], [0.0]),
    ])
    # The shorter row has valid positions 0..2 even though padding expands the batch to 4.
    assert captured
    assert captured[0][0, 0].item() == 3.0
    assert captured[0][1, 0].item() == 2.0


def _naive_bpe(tokenizer: LearnedBPETokenizer, raw: bytes) -> list[int]:
    ids = list(raw)
    while len(ids) > 1:
        best_index = None
        best_rank = None
        for index in range(len(ids) - 1):
            rank = tokenizer._merge_rank.get((ids[index], ids[index + 1]))
            if rank is not None and (best_rank is None or rank < best_rank):
                best_index, best_rank = index, rank
        if best_index is None:
            break
        ids[best_index:best_index + 2] = [BASE_VOCAB_SIZE + int(best_rank)]
    return ids


def test_heap_bpe_matches_reference_merger_and_roundtrips():
    tokenizer, _ = LearnedBPETokenizer.train(
        [" aletheia learns from experience", " learning learning language", "hello there hello"],
        vocab_size=300, min_frequency=2, max_training_bytes=10000,
    )
    for raw in [b"hello there", b" learning", b"abababab", "日本語 hello".encode("utf-8")]:
        assert tokenizer._bpe_ids(raw) == _naive_bpe(tokenizer, raw)
        assert tokenizer.decode(tokenizer.encode_bytes(raw)) == raw.decode("utf-8", errors="replace")


def test_legacy_cp1252_not_misread_as_utf16(tmp_path: Path):
    raw = "café".encode("cp1252")  # even byte length, so blind UTF-16 decoding can succeed incorrectly
    path = tmp_path / "legacy.txt"
    path.write_bytes(raw)
    assert _read_bytes(path) == "café"
    assert _read_exclusion_text(path) == "café"


def test_reset_storage_archives_all_active_data_items(tmp_path: Path):
    data = tmp_path / "data"
    data.mkdir()
    (data / "aletheia.pt").write_bytes(b"checkpoint")
    (data / "speaker_memory.json").write_text("{}", encoding="utf-8")
    (data / "objectives.json").write_text("{}", encoding="utf-8")
    (data / "archive").mkdir()
    archive = AletheiaAgent.reset_storage(tmp_path)
    assert archive is not None
    assert (archive / "aletheia.pt").read_bytes() == b"checkpoint"
    assert (archive / "speaker_memory.json").exists()
    assert (archive / "objectives.json").exists()
    assert not (data / "aletheia.pt").exists()
    assert (data / "archive").exists()


def test_prompt_budget_uses_tokenizer_not_character_count():
    from aletheia.tokenizer import ByteFallbackTokenizer

    agent = object.__new__(AletheiaAgent)
    agent.model = SimpleNamespace(cfg=SimpleNamespace(context=64))
    agent.tokenizer = ByteFallbackTokenizer()
    agent.config = SimpleNamespace(runtime=SimpleNamespace(chat_memory_results=2, chat_memory_chars=100))
    agent.objectives = SimpleNamespace(current=lambda: None)
    agent._recent_communication_context = lambda limit=4: "recent " * 100
    prompt = agent._prompt_with_context("what is this?", [], reply_reserve=16)
    assert len(agent.tokenizer.encode(prompt, bos=True, eos=False)) <= 48
    assert prompt.endswith("Turn B:\n")


def test_browser_guard_blocks_javascript_post_without_permission():
    from aletheia.browser import BrowserEnvironment

    class Request:
        url = "https://example.org/api/action"
        method = "POST"

    class Route:
        request = Request()
        aborted = False
        continued = False

        def abort(self):
            self.aborted = True

        def continue_(self):
            self.continued = True

    browser = object.__new__(BrowserEnvironment)
    browser.allow_form_submit = False
    browser.allow_authentication = False
    browser.allow_domains = set()
    browser.web = SimpleNamespace(_host_is_public=lambda host: True)
    route = Route()
    browser._guard_request(route)
    assert route.aborted and not route.continued


def _runtime_evolution_agent(score_fn):
    """Tiny stateful stand-in for deterministic runtime-evolution regressions."""
    import random

    model = torch.nn.Linear(1, 1, bias=False)
    with torch.no_grad():
        model.weight.fill_(2.0)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    runtime = SimpleNamespace(learning_rate=0.01, train_steps_per_observation=1)
    memory = SimpleNamespace(lesson_count=lambda: 1)
    agent = SimpleNamespace(
        config=SimpleNamespace(runtime=runtime),
        model=model,
        optimizer=optimizer,
        memory=memory,
        last_eval_bpb=0.05,  # deliberately stale and unrealistically optimistic
    )
    starts = []

    def train(steps):
        starts.append((float(runtime.learning_rate), float(model.weight.item()), steps))
        with torch.no_grad():
            model.weight.add_(float(runtime.learning_rate))

    def evaluate():
        score = float(score_fn(float(runtime.learning_rate)))
        agent.last_eval_bpb = score
        return {"available": True, "model_bpb": score}

    agent._train_recent_trajectory = train
    agent.evaluate_fixed_reference = evaluate
    return agent, starts


def test_runtime_evolution_uses_same_compute_control_not_stale_score(tmp_path: Path):
    from aletheia.evolution import EvolutionLab

    # Every alternate LR is worse than the current setting. The stale score of
    # 0.05 would have made them all look like improvements in the old algorithm.
    agent, starts = _runtime_evolution_agent(lambda lr: 1.0 + abs(lr - 0.01) * 100.0)
    result = EvolutionLab(tmp_path).evolve_runtime_parameter(agent, "learning_rate")
    assert result.changed is False
    assert result.before == result.after == 0.01
    assert result.score_before == result.score_after == 1.0
    assert result.reason == "no_candidate_improved_on_all_paired_seeds"
    assert len(starts) == 15  # five arms, each on three paired seeds
    assert all(start[1] == 2.0 and start[2] == 1 for start in starts)
    assert float(agent.model.weight.item()) == 2.0


def test_runtime_evolution_can_promote_a_measured_improvement(tmp_path: Path):
    from aletheia.evolution import EvolutionLab

    agent, starts = _runtime_evolution_agent(
        lambda lr: 0.8 if abs(lr - 0.0075) < 1e-9 else 1.0 + abs(lr - 0.01) * 100.0
    )
    result = EvolutionLab(tmp_path).evolve_runtime_parameter(agent, "learning_rate")
    assert result.changed is True
    assert result.after == 0.0075
    assert result.score_before == 1.0
    assert result.score_after == pytest.approx(0.8)
    assert result.reason == "candidate_improved_all_paired_seeds"
    assert len(starts) == 15
    assert float(agent.model.weight.item()) == 2.0


def test_source_edit_prompt_is_token_bounded_and_keeps_instructions(tmp_path: Path):
    from aletheia.evolution import EvolutionLab
    from aletheia.tokenizer import ByteFallbackTokenizer

    package = tmp_path / "aletheia"
    package.mkdir()
    (tmp_path / "config.json").write_text(json.dumps({"runtime": {
        "evolution_self_edit_allowlist": ["aletheia/learning_adapter.py", "aletheia/experimental.py"]
    }}), encoding="utf-8")
    (package / "learning_adapter.py").write_text(
        'payload = "' + ("x" * 4000) + '"\n', encoding="utf-8"
    )
    captured = []

    def generate(prompt, **kwargs):
        captured.append(prompt)
        return "not a unified diff"

    agent = SimpleNamespace(
        config=SimpleNamespace(runtime=SimpleNamespace(
            evolution_source_context_chars=4200,
            chat_max_response_tokens=256,
            chat_top_k=16,
        )),
        model=SimpleNamespace(cfg=SimpleNamespace(context=512), generate=generate),
        tokenizer=ByteFallbackTokenizer(),
        objectives=SimpleNamespace(current=lambda: None),
        device=torch.device("cpu"),
        state=torch.zeros(1),
    )
    result = EvolutionLab(tmp_path).propose_source_change(
        agent, "aletheia/learning_adapter.py", "Improve this function without external models."
    )
    assert result["reason"] == "self_model_must_generate_unified_diff" or result["reason"] == "self_model_generated_too_little_code"
    assert captured
    encoded = len(agent.tokenizer.encode(captured[0], bos=False, eos=False)) + 1
    assert encoded <= 512 - min(max(64, 512 // 3), 256)
    assert "Output one unified diff only" in captured[0]
    assert "Target file: aletheia/learning_adapter.py" in captured[0]
