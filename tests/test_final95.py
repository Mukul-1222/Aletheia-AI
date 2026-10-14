from pathlib import Path

from aletheia.agent import AletheiaAgent
from aletheia.config import load_config
from aletheia.curriculum import Lesson, prepare_language_text


def tiny_cfg(tmp_path: Path):
    cfg = load_config(Path("config.json"))
    cfg.runtime.device = "cpu"
    cfg.runtime.teacher_epochs = 1
    cfg.runtime.teacher_steps_per_sequence = 1
    cfg.runtime.replay_mix_fraction = 0.0
    cfg.runtime.replay_min_sequences = 0
    cfg.runtime.pretrain_learning_rate = 5e-4
    cfg.runtime.pretrain_gradient_accumulation_steps = 2
    cfg.runtime.pretrain_warmup_steps = 1
    cfg.runtime.pretrain_decay_steps = 4
    cfg.runtime.pretrain_min_lr_ratio = 0.1
    cfg.runtime.pretrain_ema_decay = 0.9
    cfg.runtime.pretrain_eval_every_shards = 999
    cfg.runtime.pretrain_finalize_ema = True
    cfg.runtime.teacher_shard_max_chars = 2000
    cfg.model.dim = 32
    cfg.model.layers = 1
    cfg.model.heads = 4
    cfg.model.ffn = 64
    cfg.model.recurrent_dim = 48
    cfg.model.context = 64
    cfg.model.dropout = 0.0
    cfg.model.gradient_checkpointing = False
    evaluation = tmp_path / "evaluation"
    evaluation.mkdir(parents=True, exist_ok=True)
    (evaluation / "heldout.txt").write_text(
        "A river moved through the valley while the traveler recorded changes in weather and light.\n" * 200,
        encoding="utf-8",
    )
    return cfg


def test_prepare_language_text_strips_chat_annotations_but_keeps_utterances():
    raw = (
        "@Begin\n"
        "@Participants: *CHI Child, *MOT Mother\n"
        "*CHI: more tapioca.\n"
        "%mor: adv|more n|tapioca .\n"
        "*MOT: more what?\n"
        "%int: question .\n"
        "*CHI: more juice.\n"
        "@End\n"
    )
    cleaned, meta = prepare_language_text(raw)
    assert meta["transcript_detected"] is True
    assert meta["annotation_lines_removed"] >= 4
    assert "more tapioca." in cleaned
    assert "more what?" in cleaned
    assert "%mor" not in cleaned and "*CHI:" not in cleaned and "@Begin" not in cleaned


def test_same_speaker_turns_are_merged_before_dialogue_pairing():
    from aletheia.dialogue import dialogue_response_pairs, extract_dialogue_turns

    text = (
        "*MOT: what are you doing?\n"
        "*MOT: come here.\n"
        "%int: directive .\n"
        "*CHI: no.\n"
        "*CHI: I am busy.\n"
        "*MOT: come here please.\n"
    )
    runs = extract_dialogue_turns(text, min_turn_chars=1)
    assert len(runs) == 1
    assert [turn.speaker for turn in runs[0]] == ["*MOT", "*CHI", "*MOT"]
    assert runs[0][0].text == "what are you doing? come here."
    assert runs[0][1].text == "no. I am busy."
    pairs = dialogue_response_pairs(text, max_examples=20, seed=7)
    assert pairs
    assert any("what are you doing? come here." in prompt for prompt, _ in pairs)


def test_pretrain_interleaves_sources_deterministically():
    shards = [
        Lesson("A1", "/tmp/a.txt#shard=1/2", "a"),
        Lesson("A2", "/tmp/a.txt#shard=2/2", "a"),
        Lesson("B1", "/tmp/b.txt#shard=1/2", "b"),
        Lesson("B2", "/tmp/b.txt#shard=2/2", "b"),
    ]
    ordered1 = AletheiaAgent._interleave_pretrain_shards(shards, 42)
    ordered2 = AletheiaAgent._interleave_pretrain_shards(shards, 42)
    assert [x.source for x in ordered1] == [x.source for x in ordered2]
    roots = [x.source.split("#shard=", 1)[0] for x in ordered1]
    assert roots[0] != roots[1]
    assert roots[1] != roots[2]


def test_pretrain_uses_accumulation_and_ema_without_continual_veto(tmp_path):
    cfg = tiny_cfg(tmp_path)
    source_dir = tmp_path / "corpus"
    source_dir.mkdir()
    (source_dir / "one.txt").write_text(
        "Language learners observe patterns in ordinary speech and text. " * 24,
        encoding="utf-8",
    )
    (source_dir / "two.txt").write_text(
        "Travelers record weather, roads, measurements, and conversations while exploring new places. " * 24,
        encoding="utf-8",
    )
    agent = AletheiaAgent(tmp_path, cfg)
    try:
        results = agent.pretrain(source_dir, epochs=1, seed=7)
        assert results
        assert agent.training_steps > 0
        assert agent.processed_tokens > 0
        assert agent.total_tokens > 0
        assert agent.memory.lesson_count() >= 2
        assert agent._training_mode == "continual"
        assert agent._ema_shadow is None  # EMA is finalized into live weights and then released.
        assert agent.last_eval_bpb is not None
        assert all(row["title"].startswith("[pretrain epoch 1]") for row in agent.lessons(10))
    finally:
        agent.close()


def test_pretrain_scheduler_warms_then_decays(tmp_path):
    cfg = tiny_cfg(tmp_path)
    agent = AletheiaAgent(tmp_path, cfg)
    try:
        agent._configure_pretraining(10)
        warm = agent._schedule_multiplier_for_next_step()
        agent.training_steps = 1
        post_warm = agent._schedule_multiplier_for_next_step()
        agent.training_steps = 10
        final = agent._schedule_multiplier_for_next_step()
        assert 0.0 < warm <= 1.0
        assert post_warm >= final
        assert final == cfg.runtime.pretrain_min_lr_ratio
    finally:
        agent.close()
