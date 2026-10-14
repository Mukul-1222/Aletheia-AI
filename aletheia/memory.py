from __future__ import annotations

import hashlib
import json
import random
import re
import sqlite3
import time
import unicodedata
from pathlib import Path



def lexical_terms(query: str) -> list[str]:
    """Extract Unicode-aware searchable terms, including letters with combining marks."""
    seen = set()
    result = []
    current: list[str] = []

    def flush() -> None:
        token = "".join(current).casefold()
        current.clear()
        if not token:
            return
        # Keep short tokens for scripts where whitespace is not a word boundary.
        # Latin-like tokens retain the previous minimum length to avoid noisy matches.
        has_cjk = any("CJK" in unicodedata.name(ch, "") or "HIRAGANA" in unicodedata.name(ch, "") or "KATAKANA" in unicodedata.name(ch, "") or "THAI" in unicodedata.name(ch, "") for ch in token)
        minimum = 1 if has_cjk else 3
        if len(token) >= minimum and token not in seen:
            seen.add(token)
            result.append(token)

    for char in str(query):
        category = unicodedata.category(char)
        if category[0] in {"L", "N"} or category.startswith("M") or char == "_":
            current.append(char)
        else:
            flush()
    flush()
    return result


def meaningful_terms(query: str, *, memory: "ExperienceMemory | None" = None, kinds: tuple[str, ...] = ("external", "lesson"), max_common_fraction: float = 0.60) -> list[str]:
    """Filter generic terms using corpus rarity, not a hand-written stopword list."""
    terms = lexical_terms(query)
    if not memory or not terms:
        return terms
    total = memory.document_count(kinds=kinds)
    if total <= 1:
        return terms
    informative = []
    for term in terms:
        df = memory.term_document_frequency(term, kinds=kinds)
        if df == 0 or (df / total) <= float(max_common_fraction):
            informative.append(term)
    return informative


class ExperienceMemory:
    """Persistent memory, frontier, and environmental resource status.

    User/assistant turns are an append-only conversation history and are not subject to the
    general experience-retention cap. The cap applies to searchable world/learning memories;
    conversation history grows with disk usage and should be exported/deleted by user request.
    """

    _unbounded_history_kinds = frozenset({"conversation_user", "conversation_response"})

    def __init__(self, path: Path, limit: int, frontier_limit: int, *, kind_limits: dict[str, int] | None = None, eviction_order: list[str] | None = None):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.limit = int(limit)
        self.frontier_limit = int(frontier_limit)
        self.kind_limits = {str(k): max(0, int(v)) for k, v in (kind_limits or {}).items()}
        self.eviction_order = [str(k) for k in (eviction_order or [])]
        self._trigram_enabled = False
        self.db = sqlite3.connect(path)
        self.db.row_factory = sqlite3.Row
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute(
            """CREATE TABLE IF NOT EXISTS memories(
                id INTEGER PRIMARY KEY,
                created REAL NOT NULL,
                kind TEXT NOT NULL,
                source TEXT NOT NULL,
                title TEXT NOT NULL,
                url TEXT NOT NULL,
                text TEXT NOT NULL,
                loss_before REAL NOT NULL DEFAULT 0,
                loss_after REAL NOT NULL DEFAULT 0,
                learning_progress REAL NOT NULL DEFAULT 0,
                action_name TEXT,
                prediction_error REAL NOT NULL DEFAULT 0,
                visits INTEGER NOT NULL DEFAULT 0,
                train_context INTEGER NOT NULL DEFAULT 512,
                holdout_indices TEXT NOT NULL DEFAULT '[]',
                accepted INTEGER NOT NULL DEFAULT 1,
                replay_eligible INTEGER NOT NULL DEFAULT 1,
                content_fingerprint TEXT NOT NULL DEFAULT ''
            )"""
        )
        self._ensure_column("memories", "loss_before", "REAL NOT NULL DEFAULT 0")
        self._ensure_column("memories", "loss_after", "REAL NOT NULL DEFAULT 0")
        self._ensure_column("memories", "action_name", "TEXT")
        self._ensure_column("memories", "prediction_error", "REAL NOT NULL DEFAULT 0")
        self._ensure_column("memories", "train_context", "INTEGER NOT NULL DEFAULT 512")
        self._ensure_column("memories", "holdout_indices", "TEXT NOT NULL DEFAULT '[]'")
        self._ensure_column("memories", "accepted", "INTEGER NOT NULL DEFAULT 1")
        self._ensure_column("memories", "replay_eligible", "INTEGER NOT NULL DEFAULT 1")
        self._ensure_column("memories", "content_fingerprint", "TEXT NOT NULL DEFAULT ''")
        rows = self.db.execute("SELECT id,text,content_fingerprint FROM memories WHERE content_fingerprint=''").fetchall()
        for row in rows:
            digest = hashlib.sha256(re.sub(r"\s+", " ", str(row[1])).strip().encode("utf-8", errors="replace")).hexdigest()
            self.db.execute("UPDATE memories SET content_fingerprint=? WHERE id=?", (digest, int(row[0])))
        # Legacy rollback rows are known rejected experiences; make them ineligible for replay.
        self.db.execute("UPDATE memories SET accepted=0, replay_eligible=0 WHERE action_name LIKE '%_rollback'")
        self.db.execute(
            """CREATE VIRTUAL TABLE IF NOT EXISTS memories_fts USING fts5(
                title, text, content='memories', content_rowid='id'
            )"""
        )
        self.db.execute(
            """CREATE TABLE IF NOT EXISTS memory_chunks(
                id INTEGER PRIMARY KEY,
                memory_id INTEGER NOT NULL,
                chunk_index INTEGER NOT NULL,
                text TEXT NOT NULL,
                created REAL NOT NULL,
                UNIQUE(memory_id, chunk_index)
            )"""
        )
        self.db.execute(
            """CREATE VIRTUAL TABLE IF NOT EXISTS memory_chunks_fts USING fts5(
                memory_id UNINDEXED, chunk_index UNINDEXED, text
            )"""
        )
        try:
            self.db.execute(
                """CREATE VIRTUAL TABLE IF NOT EXISTS memories_trigram_fts USING fts5(
                    memory_id UNINDEXED, title, text, tokenize='trigram'
                )"""
            )
            self.db.execute(
                """CREATE VIRTUAL TABLE IF NOT EXISTS memory_chunks_trigram_fts USING fts5(
                    chunk_id UNINDEXED, memory_id UNINDEXED, text, tokenize='trigram'
                )"""
            )
            self._trigram_enabled = True
        except sqlite3.DatabaseError:
            self._trigram_enabled = False
        self.db.execute("CREATE INDEX IF NOT EXISTS idx_memory_chunks_parent ON memory_chunks(memory_id, chunk_index)")
        self.db.execute(
            """CREATE TABLE IF NOT EXISTS cognitive_events(
                id INTEGER PRIMARY KEY,
                created REAL NOT NULL,
                event_kind TEXT NOT NULL,
                modality TEXT NOT NULL,
                source TEXT NOT NULL DEFAULT '',
                action_name TEXT,
                entity_id TEXT,
                salience REAL NOT NULL DEFAULT 0,
                novelty REAL NOT NULL DEFAULT 0,
                prediction_error REAL NOT NULL DEFAULT 0,
                detail_json TEXT NOT NULL DEFAULT '{}'
            )"""
        )
        self.db.execute("CREATE INDEX IF NOT EXISTS idx_cognitive_events_created ON cognitive_events(created)")
        self.db.execute("CREATE INDEX IF NOT EXISTS idx_cognitive_events_kind ON cognitive_events(event_kind, created)")
        self.db.execute(
            """CREATE TABLE IF NOT EXISTS frontier(
                url TEXT PRIMARY KEY,
                discovered REAL NOT NULL,
                parent_url TEXT NOT NULL,
                depth INTEGER NOT NULL DEFAULT 0,
                priority REAL NOT NULL DEFAULT 0,
                visits INTEGER NOT NULL DEFAULT 0
            )"""
        )
        self.db.execute(
            """CREATE TABLE IF NOT EXISTS blocked_resources(
                url TEXT PRIMARY KEY,
                blocked_at REAL NOT NULL,
                reason TEXT NOT NULL,
                source TEXT NOT NULL DEFAULT ''
            )"""
        )
        self._ensure_column("frontier", "priority", "REAL NOT NULL DEFAULT 0")
        self.db.execute("CREATE INDEX IF NOT EXISTS idx_memories_created ON memories(created)")
        self.db.execute("CREATE INDEX IF NOT EXISTS idx_memories_url ON memories(url)")
        self.db.execute("CREATE INDEX IF NOT EXISTS idx_frontier_rank ON frontier(visits, priority DESC, discovered DESC)")
        self.db.execute("CREATE TABLE IF NOT EXISTS memory_metadata(key TEXT PRIMARY KEY, value TEXT NOT NULL)")
        self.db.commit()
        self._rebuild_fts_if_empty()

    def _ensure_column(self, table: str, name: str, ddl: str) -> None:
        columns = {row[1] for row in self.db.execute(f"PRAGMA table_info({table})")}
        if name not in columns:
            self.db.execute(f"ALTER TABLE {table} ADD COLUMN {name} {ddl}")
            self.db.commit()

    def _rebuild_fts_if_empty(self) -> None:
        """Rebuild full-text indexes only for a new/unknown index schema version."""
        try:
            meta = self.db.execute(
                "SELECT value FROM memory_metadata WHERE key='fts_schema_version'"
            ).fetchone()
            needs_rebuild = meta is None or str(meta[0]) != "1"
            if needs_rebuild:
                self.db.execute("INSERT INTO memories_fts(memories_fts) VALUES('rebuild')")
                if self._trigram_enabled:
                    self.db.execute("DELETE FROM memories_trigram_fts")
                    self.db.execute("DELETE FROM memory_chunks_trigram_fts")
                    rows = self.db.execute("SELECT id,title,text FROM memories").fetchall()
                    for row in rows:
                        self.db.execute(
                            "INSERT INTO memories_trigram_fts(memory_id,title,text) VALUES(?,?,?)",
                            (int(row[0]), str(row[1]), str(row[2])),
                        )
                    chunk_rows = self.db.execute("SELECT id,memory_id,text FROM memory_chunks").fetchall()
                    for row in chunk_rows:
                        self.db.execute(
                            "INSERT INTO memory_chunks_trigram_fts(chunk_id,memory_id,text) VALUES(?,?,?)",
                            (int(row[0]), int(row[1]), str(row[2])),
                        )
                self.db.execute(
                    "INSERT OR REPLACE INTO memory_metadata(key,value) VALUES('fts_schema_version','1')"
                )
            self.db.commit()
        except sqlite3.DatabaseError:
            self.db.rollback()

    def add(
        self,
        *,
        kind: str,
        source: str,
        title: str,
        url: str,
        text: str,
        loss_before: float = 0.0,
        loss_after: float = 0.0,
        learning_progress: float = 0.0,
        action_name: str | None = None,
        prediction_error: float = 0.0,
        train_context: int = 512,
        holdout_indices: str = "[]",
        accepted: bool = True,
        replay_eligible: bool = True,
        merge_existing: bool = True,
    ) -> int:
        fingerprint = hashlib.sha256(re.sub(r"\s+", " ", str(text)).strip().encode("utf-8", errors="replace")).hexdigest()
        row = None
        if merge_existing:
            row = self.db.execute("SELECT id FROM memories WHERE url=? AND text=? LIMIT 1", (url[:4000], text)).fetchone()
        if row:
            self.db.execute(
                "UPDATE memories SET visits=visits+1, loss_before=?, loss_after=?, learning_progress=?, action_name=?, prediction_error=?, train_context=?, holdout_indices=?, accepted=?, replay_eligible=?, content_fingerprint=? WHERE id=?",
                (float(loss_before), float(loss_after), float(learning_progress), action_name, float(prediction_error), int(train_context), holdout_indices, int(bool(accepted)), int(bool(replay_eligible)), fingerprint, int(row[0])),
            )
            self.db.commit()
            return int(row[0])
        cur = self.db.execute(
            "INSERT INTO memories(created,kind,source,title,url,text,loss_before,loss_after,learning_progress,action_name,prediction_error,train_context,holdout_indices,accepted,replay_eligible,content_fingerprint) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (
                time.time(), kind, source, title[:500], url[:4000], text,
                float(loss_before), float(loss_after), float(learning_progress), action_name, float(prediction_error), int(train_context), holdout_indices, int(bool(accepted)), int(bool(replay_eligible)), fingerprint,
            ),
        )
        rowid = int(cur.lastrowid)
        try:
            self.db.execute("INSERT INTO memories_fts(rowid,title,text) VALUES(?,?,?)", (rowid, title[:500], text))
        except sqlite3.DatabaseError:
            pass
        if self._trigram_enabled:
            try:
                self.db.execute(
                    "INSERT INTO memories_trigram_fts(memory_id,title,text) VALUES(?,?,?)",
                    (rowid, title[:500], text),
                )
            except sqlite3.DatabaseError:
                pass
        self.db.commit()
        self._trim_memories()
        return rowid

    def update_learning_metrics(
        self,
        memory_id: int,
        *,
        loss_after: float,
        learning_progress: float,
        prediction_error: float,
        action_name: str | None = None,
        holdout_indices: str | None = None,
        train_context: int | None = None,
    ) -> None:
        fields = ["loss_after=?", "learning_progress=?", "prediction_error=?"]
        params: list[object] = [float(loss_after), float(learning_progress), float(prediction_error)]
        if action_name is not None:
            fields.append("action_name=?")
            params.append(action_name)
        if holdout_indices is not None:
            fields.append("holdout_indices=?")
            params.append(holdout_indices)
        if train_context is not None:
            fields.append("train_context=?")
            params.append(int(train_context))
        params.append(int(memory_id))
        self.db.execute(f"UPDATE memories SET {', '.join(fields)} WHERE id=?", params)
        self.db.commit()

    def update_learning_status(self, memory_id: int, *, accepted: bool, replay_eligible: bool) -> None:
        self.db.execute(
            "UPDATE memories SET accepted=?, replay_eligible=? WHERE id=?",
            (int(bool(accepted)), int(bool(replay_eligible)), int(memory_id)),
        )
        self.db.commit()

    def replace_chunks(self, memory_id: int, chunks: list[str]) -> None:
        """Persist chunk-level text so learned retrieval can address parts of a memory.

        Representations are intentionally not persisted here: they become stale whenever
        Aletheia learns. Retrieval recomputes representations with the current model.
        """
        memory_id = int(memory_id)
        old = self.db.execute("SELECT id FROM memory_chunks WHERE memory_id=?", (memory_id,)).fetchall()
        for row in old:
            try:
                self.db.execute("DELETE FROM memory_chunks_fts WHERE rowid=?", (int(row[0]),))
            except sqlite3.DatabaseError:
                pass
        self.db.execute("DELETE FROM memory_chunks WHERE memory_id=?", (memory_id,))
        for index, chunk in enumerate(chunks):
            text = str(chunk).strip()
            if not text:
                continue
            cur = self.db.execute(
                "INSERT INTO memory_chunks(memory_id,chunk_index,text,created) VALUES(?,?,?,?)",
                (memory_id, index, text, time.time()),
            )
            chunk_id = int(cur.lastrowid)
            try:
                self.db.execute(
                    "INSERT INTO memory_chunks_fts(rowid,memory_id,chunk_index,text) VALUES(?,?,?,?)",
                    (chunk_id, memory_id, index, text),
                )
            except sqlite3.DatabaseError:
                pass
        self.db.commit()

    def chunk_count(self, memory_id: int | None = None) -> int:
        if memory_id is None:
            row = self.db.execute("SELECT COUNT(*) FROM memory_chunks").fetchone()
        else:
            row = self.db.execute("SELECT COUNT(*) FROM memory_chunks WHERE memory_id=?", (int(memory_id),)).fetchone()
        return int(row[0]) if row else 0

    def search_chunks(self, query: str, limit: int = 24) -> list[dict]:
        terms = lexical_terms(query)
        if not terms:
            return []
        fts_query = " OR ".join(f'"{term}"' for term in terms[:24])
        try:
            rows = self.db.execute(
                """SELECT c.id AS chunk_id, c.memory_id, c.chunk_index, c.text AS chunk_text,
                          m.kind, m.source, m.title, m.url, m.loss_before, m.loss_after,
                          m.learning_progress, m.action_name, m.prediction_error
                   FROM memory_chunks_fts f
                   JOIN memory_chunks c ON c.id=f.rowid
                   JOIN memories m ON m.id=c.memory_id
                   WHERE memory_chunks_fts MATCH ?
                   ORDER BY bm25(memory_chunks_fts)
                   LIMIT ?""",
                (fts_query, int(limit)),
            ).fetchall()
            result=[]
            for row in rows:
                item=dict(row)
                item["id"] = item["chunk_id"]
                item["text"] = item["chunk_text"]
                result.append(item)
            if result:
                return result
        except sqlite3.DatabaseError:
            pass
        if self._trigram_enabled and any(len(term) >= 3 for term in terms):
            try:
                tri_query = " OR ".join(f'"{term}"' for term in terms[:24] if len(term) >= 3)
                rows = self.db.execute(
                    """SELECT f.chunk_id, f.memory_id, c.chunk_index, f.text AS chunk_text,
                              m.kind, m.source, m.title, m.url, m.loss_before, m.loss_after,
                              m.learning_progress, m.action_name, m.prediction_error
                       FROM memory_chunks_trigram_fts f
                       JOIN memory_chunks c ON c.id=f.chunk_id
                       JOIN memories m ON m.id=f.memory_id
                       WHERE memory_chunks_trigram_fts MATCH ?
                       ORDER BY f.rowid DESC LIMIT ?""",
                    (tri_query, int(limit)),
                ).fetchall()
                if rows:
                    return [dict(row) | {"id": int(row[0]), "text": row[3]} for row in rows]
            except sqlite3.DatabaseError:
                pass
        clauses = ["instr(lower(c.text), lower(?)) > 0" for _ in terms[:24]]
        if not clauses:
            return []
        params: list[object] = terms[:24] + [int(limit)]
        try:
            rows = self.db.execute(
                """SELECT c.id AS chunk_id, c.memory_id, c.chunk_index, c.text AS chunk_text,
                          m.kind, m.source, m.title, m.url, m.loss_before, m.loss_after,
                          m.learning_progress, m.action_name, m.prediction_error
                   FROM memory_chunks c JOIN memories m ON m.id=c.memory_id
                   WHERE """ + " OR ".join(clauses) + " ORDER BY c.id DESC LIMIT ?",
                params,
            ).fetchall()
        except sqlite3.DatabaseError:
            return []
        result = []
        for row in rows:
            item = dict(row)
            item["id"] = item["chunk_id"]
            item["text"] = item["chunk_text"]
            result.append(item)
        return result

    def chunks_for_memory(self, memory_id: int) -> list[dict]:
        rows=self.db.execute(
            "SELECT * FROM memory_chunks WHERE memory_id=? ORDER BY chunk_index", (int(memory_id),)
        ).fetchall()
        return [dict(row) for row in rows]

    def sample_chunk_rows(self, limit: int = 16, *, kinds: tuple[str, ...] = ("lesson", "external")) -> list[dict]:
        placeholders=",".join("?" for _ in kinds)
        rows=self.db.execute(
            f"""SELECT c.*, m.kind, m.source, m.title, m.url
                FROM memory_chunks c JOIN memories m ON m.id=c.memory_id
                WHERE m.kind IN ({placeholders}) AND m.replay_eligible=1
                ORDER BY RANDOM() LIMIT ?""",
            (*kinds, int(limit)),
        ).fetchall()
        return [dict(row) for row in rows]

    def _delete_memory_rows(self, ids: list[int]) -> None:
        if not ids:
            return
        for rowid in ids:
            try:
                self.db.execute("DELETE FROM memories_fts WHERE rowid=?", (rowid,))
            except sqlite3.DatabaseError:
                pass
            if self._trigram_enabled:
                try:
                    self.db.execute("DELETE FROM memories_trigram_fts WHERE memory_id=?", (rowid,))
                except sqlite3.DatabaseError:
                    pass
            chunk_ids = [int(x[0]) for x in self.db.execute(
                "SELECT id FROM memory_chunks WHERE memory_id=?", (rowid,)
            ).fetchall()]
            for chunk_id in chunk_ids:
                try:
                    self.db.execute("DELETE FROM memory_chunks_fts WHERE rowid=?", (chunk_id,))
                except sqlite3.DatabaseError:
                    pass
                if self._trigram_enabled:
                    try:
                        self.db.execute("DELETE FROM memory_chunks_trigram_fts WHERE chunk_id=?", (chunk_id,))
                    except sqlite3.DatabaseError:
                        pass
            self.db.execute("DELETE FROM memory_chunks WHERE memory_id=?", (rowid,))
        self.db.execute("DELETE FROM memories WHERE id IN (%s)" % ",".join("?" for _ in ids), ids)

    def _trim_memories(self) -> None:
        # First enforce configured per-kind caps. These are retention controls, not semantic
        # rules: the kinds themselves are supplied by configuration.
        for kind, cap in self.kind_limits.items():
            if cap <= 0 or kind in self._unbounded_history_kinds:
                continue
            # Retain the newest `cap` rows. OFFSET must be applied to a newest-first
            # ordering; the previous oldest-first query selected the rows to delete.
            rows = self.db.execute(
                "SELECT id FROM memories WHERE kind=? ORDER BY created DESC, id DESC LIMIT -1 OFFSET ?",
                (kind, int(cap)),
            ).fetchall()
            if rows:
                self._delete_memory_rows([int(row[0]) for row in rows])

        # Conversation history is retained independently of the general memory cap.
        # Only non-transcript experience rows count against `limit`.
        protected = tuple(sorted(self._unbounded_history_kinds))
        placeholders = ",".join("?" for _ in protected)
        count = int(self.db.execute(
            f"SELECT COUNT(*) FROM memories WHERE kind NOT IN ({placeholders})", protected
        ).fetchone()[0])
        if count <= self.limit:
            self.db.commit()
            return
        remove = count - self.limit
        ids: list[int] = []
        seen: set[int] = set()
        # Configured transient kinds are evicted first. This prevents high-frequency sensory
        # streams from consuming the same retention pool as durable lessons and web knowledge.
        for kind in self.eviction_order:
            if kind in self._unbounded_history_kinds:
                continue
            if len(ids) >= remove:
                break
            rows = self.db.execute(
                "SELECT id FROM memories WHERE kind=? ORDER BY created ASC, id ASC LIMIT ?",
                (kind, remove - len(ids)),
            ).fetchall()
            for row in rows:
                rowid = int(row[0])
                if rowid not in seen:
                    ids.append(rowid)
                    seen.add(rowid)
        if len(ids) < remove:
            rows = self.db.execute(
                f"SELECT id FROM memories WHERE kind != 'lesson' AND kind NOT IN ({placeholders}) ORDER BY created ASC, id ASC LIMIT ?",
                (*protected, remove - len(ids)),
            ).fetchall()
            for row in rows:
                rowid = int(row[0])
                if rowid not in seen:
                    ids.append(rowid)
                    seen.add(rowid)
                    if len(ids) >= remove:
                        break
        if ids:
            self._delete_memory_rows(ids)
            self.db.commit()

    def add_cognitive_event(
        self,
        *,
        event_kind: str,
        modality: str = "internal",
        source: str = "",
        action_name: str | None = None,
        entity_id: str | None = None,
        salience: float = 0.0,
        novelty: float = 0.0,
        prediction_error: float = 0.0,
        detail: dict | None = None,
        limit: int = 50000,
    ) -> int:
        """Persist a compact developmental event without copying raw sensor payloads."""
        cur = self.db.execute(
            """INSERT INTO cognitive_events(
                created,event_kind,modality,source,action_name,entity_id,salience,novelty,prediction_error,detail_json
            ) VALUES(?,?,?,?,?,?,?,?,?,?)""",
            (
                time.time(), str(event_kind), str(modality), str(source), action_name, entity_id,
                float(max(0.0, salience)), float(max(0.0, novelty)), float(max(0.0, prediction_error)),
                json.dumps(detail or {}, ensure_ascii=False, separators=(",", ":")),
            ),
        )
        event_id = int(cur.lastrowid)
        cap = max(0, int(limit))
        if cap:
            excess = self.db.execute("SELECT COUNT(*) FROM cognitive_events").fetchone()[0] - cap
            if excess > 0:
                self.db.execute(
                    "DELETE FROM cognitive_events WHERE id IN (SELECT id FROM cognitive_events ORDER BY created ASC LIMIT ?)",
                    (int(excess),),
                )
        self.db.commit()
        return event_id

    def recent_cognitive_events(self, limit: int = 32, *, event_kind: str | None = None) -> list[dict]:
        limit = max(1, int(limit))
        if event_kind:
            rows = self.db.execute(
                "SELECT * FROM cognitive_events WHERE event_kind=? ORDER BY created DESC LIMIT ?",
                (str(event_kind), limit),
            ).fetchall()
        else:
            rows = self.db.execute(
                "SELECT * FROM cognitive_events ORDER BY created DESC LIMIT ?",
                (limit,),
            ).fetchall()
        result = []
        for row in rows:
            item = dict(row)
            try:
                item["detail"] = json.loads(item.pop("detail_json") or "{}")
            except (TypeError, ValueError):
                item["detail"] = {}
            result.append(item)
        return result

    def cognitive_summary(self, limit: int = 256) -> dict:
        rows = self.db.execute(
            "SELECT event_kind, modality, AVG(salience) AS mean_salience, AVG(novelty) AS mean_novelty, COUNT(*) AS count "
            "FROM (SELECT * FROM cognitive_events ORDER BY created DESC LIMIT ?) GROUP BY event_kind, modality",
            (max(1, int(limit)),),
        ).fetchall()
        return {
            "events": sum(int(row["count"]) for row in rows),
            "kinds": [
                {
                    "event_kind": str(row["event_kind"]),
                    "modality": str(row["modality"]),
                    "count": int(row["count"]),
                    "mean_salience": float(row["mean_salience"] or 0.0),
                    "mean_novelty": float(row["mean_novelty"] or 0.0),
                }
                for row in rows
            ],
        }

    def add_links(self, urls: list[str], parent_url: str, depth: int, priority: float = 0.0) -> None:
        for url in urls[: self.frontier_limit]:
            if self.is_blocked(url):
                continue
            self.db.execute(
                "INSERT INTO frontier(url,discovered,parent_url,depth,priority) VALUES(?,?,?,?,?) ON CONFLICT(url) DO UPDATE SET priority=MAX(priority, excluded.priority)",
                (url[:4000], time.time(), parent_url[:4000], int(depth), float(priority)),
            )
        self._trim_frontier()
        self.db.commit()

    def block_url(self, url: str, reason: str, source: str = "") -> None:
        self.db.execute(
            "INSERT INTO blocked_resources(url,blocked_at,reason,source) VALUES(?,?,?,?) ON CONFLICT(url) DO UPDATE SET blocked_at=excluded.blocked_at, reason=excluded.reason, source=excluded.source",
            (url[:4000], time.time(), reason[:500], source[:200]),
        )
        self.db.execute("DELETE FROM frontier WHERE url=?", (url[:4000],))
        self.db.commit()

    def is_blocked(self, url: str) -> bool:
        row = self.db.execute("SELECT 1 FROM blocked_resources WHERE url=? LIMIT 1", (url[:4000],)).fetchone()
        return row is not None

    def _trim_frontier(self) -> None:
        count = int(self.db.execute("SELECT COUNT(*) FROM frontier").fetchone()[0])
        if count <= self.frontier_limit:
            return
        self.db.execute(
            "DELETE FROM frontier WHERE url IN (SELECT url FROM frontier ORDER BY visits ASC, priority ASC, discovered ASC LIMIT ?)",
            (count - self.frontier_limit,),
        )

    def next_link(self) -> dict | None:
        row = self.db.execute(
            """SELECT f.url,f.parent_url,f.depth,f.priority,f.visits
               FROM frontier f
               LEFT JOIN blocked_resources b ON b.url=f.url
               WHERE f.visits=0 AND b.url IS NULL
               ORDER BY f.priority DESC, f.discovered ASC LIMIT 1"""
        ).fetchone()
        return dict(row) if row else None

    def mark_visit(self, url: str) -> None:
        self.db.execute("UPDATE frontier SET visits=visits+1 WHERE url=?", (url,))
        self.db.commit()

    def recent_replayable(self, limit: int, kinds: tuple[str, ...] | None = None) -> list[dict]:
        if kinds:
            placeholders = ",".join("?" for _ in kinds)
            rows = self.db.execute(
                f"SELECT * FROM memories WHERE kind IN ({placeholders}) AND replay_eligible=1 ORDER BY created DESC LIMIT ?",
                (*kinds, int(limit)),
            ).fetchall()
        else:
            rows = self.db.execute(
                "SELECT * FROM memories WHERE replay_eligible=1 ORDER BY created DESC LIMIT ?",
                (int(limit),),
            ).fetchall()
        return [dict(x) for x in rows]

    def recent(self, limit: int, kinds: tuple[str, ...] | None = None) -> list[dict]:
        if kinds:
            placeholders = ",".join("?" for _ in kinds)
            rows = self.db.execute(
                f"SELECT * FROM memories WHERE kind IN ({placeholders}) ORDER BY created DESC LIMIT ?",
                (*kinds, int(limit)),
            ).fetchall()
        else:
            rows = self.db.execute("SELECT * FROM memories ORDER BY created DESC LIMIT ?", (int(limit),)).fetchall()
        return [dict(x) for x in rows]

    def sample(self, limit: int, *, exclude_internal: bool = False) -> list[dict]:
        if exclude_internal:
            rows = self.db.execute(
                "SELECT * FROM memories WHERE kind='external' AND replay_eligible=1 ORDER BY RANDOM() LIMIT ?", (int(limit),)
            ).fetchall()
        else:
            rows = self.db.execute("SELECT * FROM memories ORDER BY RANDOM() LIMIT ?", (int(limit),)).fetchall()
        return [dict(x) for x in rows]

    def sample_sequence(self, length: int, *, exclude_internal: bool = True) -> list[dict]:
        rows = self.db.execute(
            "SELECT * FROM memories WHERE kind='external' AND replay_eligible=1 ORDER BY id DESC LIMIT ?",
            (max(int(length) * 6, int(length)),),
        ).fetchall()
        rows = list(reversed(rows))
        if len(rows) <= length:
            return [dict(x) for x in rows]
        start = random.randint(0, len(rows) - length)
        return [dict(x) for x in rows[start:start + length]]

    def document_count(self, *, kinds: tuple[str, ...] | None = None) -> int:
        if kinds:
            placeholders = ",".join("?" for _ in kinds)
            row = self.db.execute(f"SELECT COUNT(*) FROM memories WHERE kind IN ({placeholders})", kinds).fetchone()
        else:
            row = self.db.execute("SELECT COUNT(*) FROM memories").fetchone()
        return int(row[0]) if row else 0

    def term_document_frequency(self, term: str, *, kinds: tuple[str, ...] | None = None) -> int:
        safe = str(term).strip().lower()
        if not safe or len(lexical_terms(safe)) != 1:
            return 0
        try:
            if kinds:
                placeholders = ",".join("?" for _ in kinds)
                row = self.db.execute(
                    "SELECT COUNT(*) FROM memories_fts f JOIN memories m ON m.id=f.rowid "
                    f"WHERE m.kind IN ({placeholders}) AND memories_fts MATCH ?",
                    (*kinds, f'"{safe}"'),
                ).fetchone()
            else:
                row = self.db.execute(
                    "SELECT COUNT(*) FROM memories_fts WHERE memories_fts MATCH ?",
                    (f'"{safe}"',),
                ).fetchone()
            return int(row[0]) if row else 0
        except sqlite3.DatabaseError:
            if kinds:
                placeholders = ",".join("?" for _ in kinds)
                row = self.db.execute(
                    "SELECT COUNT(*) FROM memories WHERE kind IN (" + placeholders + ") AND instr(lower(text), lower(?)) > 0",
                    (*kinds, safe),
                ).fetchone()
            else:
                row = self.db.execute(
                    "SELECT COUNT(*) FROM memories WHERE instr(lower(text), lower(?)) > 0",
                    (safe,),
                ).fetchone()
            return int(row[0]) if row else 0

    def search(self, query: str, limit: int = 8) -> list[dict]:
        terms = meaningful_terms(query)
        if not terms:
            return []
        fts_query = " OR ".join(f'"{term}"' for term in terms[:16])
        try:
            rows = self.db.execute(
                "SELECT m.* FROM memories_fts f JOIN memories m ON m.id=f.rowid WHERE memories_fts MATCH ? ORDER BY rank LIMIT ?",
                (fts_query, int(limit)),
            ).fetchall()
            result = [dict(x) for x in rows]
            if result:
                return result
        except sqlite3.DatabaseError:
            pass
        clauses = ["instr(lower(m.text), lower(?)) > 0" for _ in terms[:16]]
        if not clauses:
            return []
        kinds = ("external", "lesson", "conversation_user", "conversation_response", "sensory_vision", "sensory_audio")
        placeholders = ",".join("?" for _ in kinds)
        params: list[object] = terms[:16] + list(kinds) + [int(limit)]
        try:
            rows = self.db.execute(
                "SELECT m.* FROM memories m WHERE (" + " OR ".join(clauses) + ") "
                f"AND m.kind IN ({placeholders}) ORDER BY m.created DESC LIMIT ?",
                params,
            ).fetchall()
        except sqlite3.DatabaseError:
            return []
        return [dict(x) for x in rows]


    def external_candidates(self, query: str, limit: int = 16, recent_limit: int = 16) -> list[dict]:
        """Return a broad, non-authoritative candidate pool for learned retrieval.

        Retrieval uses lexical search only to narrow the pool; the neural model
        ranks the candidates later in the communication layer. This avoids a
        second pretrained embedding model and keeps the representation owned by
        Aletheia itself.
        """
        candidates: dict[int, dict] = {}
        for row in self.search(query, max(int(limit) * 2, 8)):
            if row.get("kind") == "external":
                candidates[int(row["id"])] = row
        for row in self.recent(max(int(recent_limit), int(limit)), kinds=("external",)):
            candidates[int(row["id"])] = row
        if not candidates:
            for row in self.sample(max(int(limit), 1), exclude_internal=True):
                candidates[int(row["id"])] = row
        return list(candidates.values())[: max(int(limit) * 3, 1)]

    def query_seed(self) -> str:
        recent = self.recent(3, kinds=("external", "conversation"))
        for item in recent:
            title = " ".join(item["title"].split()).strip()
            if 3 <= len(title) <= 180:
                return title
        frontier = self.next_link()
        if frontier:
            return frontier["url"]
        return ""

    def summary(self) -> dict:
        count = int(self.db.execute("SELECT COUNT(*) FROM memories").fetchone()[0])
        avg = float(self.db.execute("SELECT COALESCE(AVG(learning_progress),0) FROM memories").fetchone()[0])
        front = int(self.db.execute("SELECT COUNT(*) FROM frontier WHERE visits=0").fetchone()[0])
        blocked = int(self.db.execute("SELECT COUNT(*) FROM blocked_resources").fetchone()[0])
        last = self.db.execute("SELECT created FROM memories ORDER BY created DESC LIMIT 1").fetchone()
        return {
            "count": count,
            "mean_learning_progress": avg,
            "unvisited_frontier": front,
            "blocked_resources": blocked,
            "last_created": float(last[0]) if last else None,
        }

    def lesson_count(self, *, validated_only: bool = True) -> int:
        if validated_only:
            row = self.db.execute("SELECT COUNT(DISTINCT content_fingerprint) FROM memories WHERE kind='lesson' AND accepted=1").fetchone()
        else:
            row = self.db.execute("SELECT COUNT(*) FROM memories WHERE kind='lesson'").fetchone()
        return int(row[0]) if row else 0

    def sample_validated_lessons(self, limit: int = 16) -> list[dict]:
        """Sample replay sources across the full validated lesson history."""
        rows = self.db.execute(
            "SELECT * FROM memories WHERE kind='lesson' AND accepted=1 AND replay_eligible=1 ORDER BY RANDOM() LIMIT ?",
            (int(limit),),
        ).fetchall()
        return [dict(x) for x in rows]

    def recent_lessons(self, limit: int = 10, *, validated_only: bool = True) -> list[dict]:
        if validated_only:
            rows = self.db.execute(
                "SELECT * FROM memories WHERE kind='lesson' AND accepted=1 ORDER BY created DESC LIMIT ?",
                (int(limit),),
            ).fetchall()
        else:
            rows = self.db.execute(
                "SELECT * FROM memories WHERE kind='lesson' ORDER BY created DESC LIMIT ?",
                (int(limit),),
            ).fetchall()
        return [dict(row) for row in rows]

    def experience_lesson_count(self) -> int:
        row = self.db.execute("SELECT COUNT(*) FROM memories WHERE kind='lesson'").fetchone()
        return int(row[0]) if row else 0

    def lesson_candidates(self, query: str, limit: int = 24) -> list[dict]:
        terms = meaningful_terms(query)
        if not terms:
            return []
        fts_query = " OR ".join(f'"{term}"' for term in terms[:16])
        try:
            rows = self.db.execute(
                "SELECT m.* FROM memories_fts f JOIN memories m ON m.id=f.rowid "
                "WHERE m.kind='lesson' AND memories_fts MATCH ? ORDER BY rank LIMIT ?",
                (fts_query, int(limit)),
            ).fetchall()
            return [dict(row) for row in rows]
        except sqlite3.DatabaseError:
            return []

    def clear_learning(self) -> None:
        """Clear learned experiences and web frontier without deleting the database file."""
        for table in ("memory_chunks_fts", "memory_chunks", "memories_fts", "memories", "frontier", "blocked_resources"):
            try:
                self.db.execute(f"DELETE FROM {table}")
            except sqlite3.DatabaseError:
                pass
        self.db.commit()

    def close(self) -> None:
        self.db.commit()
        self.db.close()
