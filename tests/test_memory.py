from pathlib import Path

from aletheia.memory import ExperienceMemory


def test_persistence_frontier_and_fts(tmp_path: Path):
    path = tmp_path / "memory.sqlite3"
    m = ExperienceMemory(path, 10, 10)
    m.add(kind="external", source="test", title="Neural world model", url="https://example.org/a", text="world model predictive learning")
    m.add_links(["https://example.org/b", "https://example.org/c"], "https://example.org/a", 1, 0.5)
    assert m.summary()["count"] == 1
    assert m.next_link()["url"] in {"https://example.org/b", "https://example.org/c"}
    assert m.search("predictive")[0]["title"] == "Neural world model"
    m.close()
    m2 = ExperienceMemory(path, 10, 10)
    assert m2.summary()["count"] == 1
    m2.close()


def test_conversation_history_is_not_evicted_by_general_memory_cap(tmp_path: Path):
    memory = ExperienceMemory(
        tmp_path / "history.sqlite3", limit=2, frontier_limit=10,
        eviction_order=["conversation_user", "conversation_response"],
        kind_limits={"conversation_user": 1, "conversation_response": 1},
    )
    try:
        for index in range(5):
            memory.add(kind="conversation_user", source="human_chat", title=f"user {index}", url="", text=f"message {index}")
            memory.add(kind="conversation_response", source="aletheia", title=f"reply {index}", url="", text=f"reply {index}")
            memory.add(kind="external", source="web", title=f"page {index}", url=f"https://example.org/{index}", text=f"external facts {index}")
        rows = memory.db.execute(
            "SELECT kind, COUNT(*) AS n FROM memories WHERE kind IN ('conversation_user','conversation_response') GROUP BY kind"
        ).fetchall()
        counts = {row["kind"]: int(row["n"]) for row in rows}
        assert counts == {"conversation_response": 5, "conversation_user": 5}
        non_history = int(memory.db.execute(
            "SELECT COUNT(*) FROM memories WHERE kind NOT IN ('conversation_user','conversation_response')"
        ).fetchone()[0])
        assert non_history <= 2
        assert [row["text"] for row in memory.recent(10, kinds=("conversation_user",))] == [f"message {index}" for index in range(4, -1, -1)]
    finally:
        memory.close()
