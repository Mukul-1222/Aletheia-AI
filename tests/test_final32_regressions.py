from __future__ import annotations

import json
from pathlib import Path

from aletheia import __version__
from aletheia.agent import AletheiaAgent
from aletheia.config import load_config


def test_release_metadata_is_consistent():
    root = Path(__file__).resolve().parents[1]
    assert __version__ == "0.34.0"
    project = (root / "pyproject.toml").read_text(encoding="utf-8")
    assert 'version = "0.34.0"' in project
    config = json.loads((root / "config.json").read_text(encoding="utf-8"))
    assert config["pyproject_version"] == "0.34.0"


def test_default_configuration_allows_labeled_early_chat():
    config = load_config(Path("config.json"))
    assert config.runtime.chat_allow_unready_generation is True
    assert 1 <= config.runtime.chat_unready_max_response_tokens <= config.model.context


def test_experimental_text_integrity_rejects_corrupted_characters():
    assert not AletheiaAgent._experimental_text_is_usable("` ѯ\x00k�K�k�\x06|n�t �\\�")
    assert not AletheiaAgent._experimental_text_is_usable("\ufffd\ufffd garbage")
    assert AletheiaAgent._experimental_text_is_usable("A tentative response about the current lesson.")


def test_formal_readiness_remains_separate_from_early_chat_option(tmp_path):
    config = load_config(Path("config.json"))
    config.model.dim = 32
    config.model.layers = 1
    config.model.heads = 4
    config.model.ffn = 64
    config.model.recurrent_dim = 48
    config.model.context = 96
    config.runtime.device = "cpu"
    config.runtime.language_ready_min_tokens = 10**9
    agent = AletheiaAgent(tmp_path, config)
    try:
        assert agent._chat_generation_ready() is False
        assert config.runtime.chat_allow_unready_generation is True
        assert agent._language_ready() is False
    finally:
        agent.close()


def test_identical_chat_turns_are_preserved_as_distinct_history_rows(tmp_path, monkeypatch):
    config = load_config(Path("config.json"))
    config.model.dim = 32
    config.model.layers = 1
    config.model.heads = 4
    config.model.ffn = 64
    config.model.recurrent_dim = 48
    config.model.context = 96
    config.runtime.device = "cpu"
    config.runtime.language_ready_min_tokens = 10**9
    agent = AletheiaAgent(tmp_path, config)
    try:
        monkeypatch.setattr(
            agent.model,
            "generate_candidates",
            lambda *args, **kwargs: (
                "Hello there, I am trying to respond.",
                agent.state.detach().clone(),
                {"candidate_count": 1, "generated_tokens": 8},
            ),
        )
        for _ in range(2):
            assert agent.communicate("hello").startswith("[experimental] ")
        user_count = agent.memory.db.execute(
            "SELECT COUNT(*) FROM memories WHERE kind='conversation_user' AND text='hello'"
        ).fetchone()[0]
        reply_count = agent.memory.db.execute(
            "SELECT COUNT(*) FROM memories WHERE kind='conversation_response' AND text='[experimental] Hello there, I am trying to respond.'"
        ).fetchone()[0]
        assert user_count == 2
        assert reply_count == 2
    finally:
        agent.close()
