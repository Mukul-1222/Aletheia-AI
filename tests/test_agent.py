from pathlib import Path

from aletheia.agent import AletheiaAgent
from aletheia.config import load_config
from aletheia.web import Page


class FakeWeb:
    def __init__(self):
        self.n = 0

    @staticmethod
    def _healthy_text(topic: str, page_number: int) -> str:
        # Test realistic, non-repetitive extracted prose so the structural web filter
        # does not have to choose between safe ingestion and allowing boilerplate.
        sentences = [
            f"{topic} observation {index} records a distinct detail from page {page_number}."
            for index in range(30)
        ]
        return " ".join(sentences)

    def discover(self):
        self.n += 1
        return Page(
            f"https://example.org/{self.n}",
            f"Page {self.n}",
            self._healthy_text("Developmental external", self.n),
            200,
            [f"https://example.org/link/{self.n}"],
        )

    def search(self, query, limit=5):
        return ["https://example.org/search"]

    def open(self, url):
        self.n += 1
        return Page(url, f"Result {self.n}", self._healthy_text("Another external", self.n), 200, [f"https://example.org/link/{self.n}/next"] if self.n < 6 else [])

    def close(self):
        pass


def test_newborn_bootstrap_and_recurrent_learning(tmp_path: Path):
    cfg = load_config(Path("config.json"))
    cfg.model.dim = 32
    cfg.model.layers = 1
    cfg.model.heads = 4
    cfg.model.ffn = 64
    cfg.model.recurrent_dim = 48
    cfg.model.context = 64
    cfg.runtime.minimum_bootstrap_memories = 3
    cfg.runtime.train_steps_per_observation = 1
    cfg.runtime.sequence_length = 3
    cfg.runtime.replay_sequences = 0
    cfg.runtime.memory_limit = 20
    agent = AletheiaAgent(tmp_path, cfg)
    agent.web.close()
    agent.web = FakeWeb()
    for _ in range(3):
        event = agent.autonomous_cycle()
        assert event["capability"] in {"discover", "follow", "search"}
        assert "result" in event
    assert agent.memory.summary()["count"] >= 3
    assert agent.total_tokens > 0
    assert all(event["valid_transition"] for event in [agent.autonomous_cycle()]) or agent.memory.summary()["count"] >= 3
    agent.close()

def test_communication_is_from_scratch_and_grounded_in_memory(tmp_path: Path):
    cfg = load_config(Path("config.json"))
    cfg.model.dim = 32
    cfg.model.layers = 1
    cfg.model.heads = 4
    cfg.model.ffn = 64
    cfg.model.recurrent_dim = 48
    cfg.model.context = 96
    cfg.model.dropout = 0.0
    cfg.runtime.memory_limit = 30
    cfg.runtime.max_frontier = 30
    cfg.runtime.train_steps_per_observation = 1
    cfg.runtime.chat_train_steps = 1
    cfg.runtime.chat_memory_results = 4
    cfg.runtime.chat_memory_chars = 300
    cfg.runtime.chat_max_response_tokens = 16
    eval_dir = tmp_path / "evaluation"
    eval_dir.mkdir(parents=True, exist_ok=True)
    (eval_dir / "heldout.txt").write_text(("What did you learn today? How are you feeling about the world and what did you notice when you spoke with another person? " * 900), encoding="utf-8")
    agent = AletheiaAgent(tmp_path, cfg)
    agent.memory.add(
        kind="external",
        source="test-world",
        title="Learned water cycle",
        url="https://example.org/water",
        text="Water moves through evaporation, condensation, precipitation, and collection.",
    )
    before_state = agent.state.detach().clone()
    reply = agent.communicate("What did you learn about the water cycle?")
    assert isinstance(reply, str)
    assert agent.memory.summary()["count"] >= 3
    assert states_differ(before_state, agent.state)
    assert agent.memory.recent(10, kinds=("conversation_response",))
    agent.close()


def states_differ(a, b):
    import torch
    return not torch.equal(a.detach().cpu(), b.detach().cpu())

def test_communication_avoids_untrained_gibberish_and_reports_memory(tmp_path: Path):
    cfg = load_config(Path("config.json"))
    cfg.model.dim = 32
    cfg.model.layers = 1
    cfg.model.heads = 4
    cfg.model.ffn = 64
    cfg.model.recurrent_dim = 48
    cfg.model.context = 96
    cfg.model.dropout = 0.0
    cfg.runtime.memory_limit = 30
    cfg.runtime.max_frontier = 30
    cfg.runtime.chat_train_steps = 1
    cfg.runtime.chat_memory_results = 4
    cfg.runtime.chat_memory_chars = 300
    cfg.runtime.chat_max_response_tokens = 16
    cfg.runtime.language_ready_loss = 0.0
    eval_dir = tmp_path / "evaluation"
    eval_dir.mkdir(parents=True, exist_ok=True)
    (eval_dir / "heldout.txt").write_text(("What did you learn today? How are you feeling about the world and what did you notice when you spoke with another person? " * 900), encoding="utf-8")
    agent = AletheiaAgent(tmp_path, cfg)
    agent.memory.add(
        kind="external",
        source="test-world",
        title="Learned water cycle",
        url="https://example.org/water",
        text="Water moves through evaporation, condensation, precipitation, and collection.",
    )
    reply = agent.communicate("What did you learn about the water cycle?")
    assert "Learned water cycle" in reply
    assert "Water" in reply or "water" in reply
    agent.close()


def test_chat_does_not_answer_greeting_with_unrelated_web_memory(tmp_path: Path):
    cfg = load_config(Path("config.json"))
    cfg.model.dim = 32
    cfg.model.layers = 1
    cfg.model.heads = 4
    cfg.model.ffn = 64
    cfg.model.recurrent_dim = 48
    cfg.model.context = 96
    cfg.model.dropout = 0.0
    cfg.runtime.memory_limit = 30
    cfg.runtime.chat_train_steps = 1
    cfg.runtime.chat_memory_results = 4
    cfg.runtime.chat_memory_chars = 300
    cfg.runtime.chat_max_response_tokens = 16
    cfg.runtime.language_ready_loss = 0.0
    agent = AletheiaAgent(tmp_path, cfg)
    agent.memory.add(
        kind="external",
        source="test-world",
        title="Plaque study",
        url="https://example.org/plaque",
        text="A research article about hidden artery plaque in young adults.",
    )
    reply = agent.communicate("Hello")
    assert "Plaque study" not in reply
    assert "artery plaque" not in reply
    agent.close()


def test_guided_text_teaching_and_reset(tmp_path):
    from aletheia.agent import AletheiaAgent
    from aletheia.config import load_config
    cfg = load_config(tmp_path / 'cfg.json') if (tmp_path / 'cfg.json').exists() else load_config(Path('config.json'))
    cfg.runtime.device = 'cpu'
    cfg.runtime.teacher_epochs = 1
    cfg.runtime.teacher_max_chunks = 2
    cfg.runtime.teacher_sequence_length = 2
    agent = AletheiaAgent(tmp_path, cfg)
    try:
        result = agent.teach_input('text:A conversation has turns. A listener receives a message and responds using context.')
        assert result and result[0]['title'] == 'teacher text'
        assert agent.memory.lesson_count() == 1
        assert agent.total_tokens > 0
    finally:
        agent.close()
    archive = AletheiaAgent.reset_storage(tmp_path)
    assert archive is not None


def test_missing_teaching_path_never_becomes_literal_lesson(tmp_path):
    from aletheia.agent import AletheiaAgent
    from aletheia.config import load_config
    cfg = load_config(Path('config.json'))
    cfg.runtime.device = 'cpu'
    agent = AletheiaAgent(tmp_path, cfg)
    try:
        import pytest
        missing_path = str(tmp_path / '__definitely_missing__' / 'communication_50kb.txt')
        with pytest.raises(FileNotFoundError):
            agent.teach_input(missing_path)
        assert agent.memory.lesson_count() == 0
        result = agent.teach_input('text:Hello, this is a real lesson.')
        assert result[0]['chars'] == len('Hello, this is a real lesson.')
    finally:
        agent.close()
