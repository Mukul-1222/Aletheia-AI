"""Local-first FastAPI web application for Aletheia FINAL34.

The server binds to loopback by default. Browser camera and microphone permissions
remain user-controlled; raw sensory samples are temporary and are not archived by
this layer. Aletheia's neural engine and learning jobs share one lock so model updates
cannot race user chat or sensory operations.
"""
from __future__ import annotations

import argparse
import json
import sqlite3
import threading
import time
import uuid
import wave
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from starlette.background import BackgroundTask
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .agent import AletheiaAgent
from .config import load_config
from .learning_pipeline import LearningPipeline, SUPPORTED_SUFFIXES, DEFAULT_MAX_UPLOAD_BYTES

APP_VERSION = "0.34.0"
MAX_CHAT_CHARS = 12_000
MAX_IMAGE_BYTES = 10 * 1024 * 1024
MAX_AUDIO_BYTES = 30 * 1024 * 1024
_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}
_AUDIO_SUFFIXES = {".wav"}


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=MAX_CHAT_CHARS)
    max_tokens: int = Field(default=128, ge=8, le=512)


class NewChatRequest(BaseModel):
    title: str = Field(default="New conversation", max_length=120)


class FeedbackRequest(BaseModel):
    positive: bool
    note: str = Field(default="", max_length=2000)


class TextLearningRequest(BaseModel):
    text: str = Field(min_length=40, max_length=2_000_000)
    title: str = Field(default="Pasted lesson", max_length=300)
    license: str = Field(default="user-provided", max_length=500)
    auto_train: bool = True


class WebApplicationService:
    def __init__(self, root: Path, config_path: Path, *, auto_train: bool = True):
        self.root = root.expanduser().resolve()
        self.config_path = config_path.expanduser().resolve()
        self.data_dir = self.root / "data"
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.db_path = self.data_dir / "webapp.sqlite3"
        self.staging_dir = self.data_dir / ".webapp_staging"
        self.staging_dir.mkdir(parents=True, exist_ok=True)
        self.agent_lock = threading.RLock()
        self.agent: AletheiaAgent | None = None
        self.agent_load_error: str | None = None
        self.auto_train = bool(auto_train)
        self.worker_stop = threading.Event()
        self.worker_thread: threading.Thread | None = None
        self.worker_status: dict[str, Any] = {
            "enabled": self.auto_train, "running": False, "current_job_id": None,
            "current_job_state": None, "last_error": None,
        }
        self.last_status: dict[str, Any] = {
            "engine_loaded": False, "engine_state": "not_loaded", "language_ready": False,
            "device": "not_initialized", "memory_count": 0, "lessons": 0,
            "speech_ready": False, "tts_ready": False, "parameter_count": None,
        }
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.db_path, timeout=15.0)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA busy_timeout=15000")
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("PRAGMA foreign_keys=ON")
        return connection

    def _init_db(self) -> None:
        with self._connect() as db:
            db.executescript("""
                CREATE TABLE IF NOT EXISTS chats (
                    id TEXT PRIMARY KEY,
                    title TEXT NOT NULL,
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL,
                    archived INTEGER NOT NULL DEFAULT 0
                );
                CREATE TABLE IF NOT EXISTS chat_messages (
                    id TEXT PRIMARY KEY,
                    chat_id TEXT NOT NULL REFERENCES chats(id) ON DELETE CASCADE,
                    role TEXT NOT NULL CHECK(role IN ('user','assistant','system')),
                    content TEXT NOT NULL,
                    created_at REAL NOT NULL,
                    response_mode TEXT NOT NULL DEFAULT '',
                    reply_to TEXT NOT NULL DEFAULT ''
                );
                CREATE INDEX IF NOT EXISTS idx_chat_messages_chat_time ON chat_messages(chat_id, created_at);
                CREATE INDEX IF NOT EXISTS idx_chats_updated ON chats(updated_at DESC);
                CREATE TABLE IF NOT EXISTS message_feedback (
                    message_id TEXT PRIMARY KEY REFERENCES chat_messages(id) ON DELETE CASCADE,
                    positive INTEGER NOT NULL CHECK(positive IN (0,1)),
                    note TEXT NOT NULL DEFAULT '',
                    created_at REAL NOT NULL,
                    online_learning_applied INTEGER NOT NULL DEFAULT 0,
                    online_learning_json TEXT NOT NULL DEFAULT '{}'
                );
            """)
            row = db.execute("SELECT COUNT(*) FROM chats").fetchone()
            if not row or int(row[0]) == 0:
                now = time.time()
                db.execute("INSERT INTO chats(id,title,created_at,updated_at) VALUES(?,?,?,?)", (uuid.uuid4().hex, "First conversation", now, now))

    def get_agent(self) -> AletheiaAgent:
        if self.agent is not None:
            return self.agent
        try:
            config = load_config(self.config_path)
            self.agent = AletheiaAgent(self.root, config)
            self.agent_load_error = None
            self._refresh_status_locked()
            return self.agent
        except Exception as exc:
            self.agent_load_error = f"{type(exc).__name__}: {exc}"
            raise

    def _refresh_status_locked(self) -> dict[str, Any]:
        if self.agent is None:
            return dict(self.last_status)
        agent = self.agent
        try:
            self.last_status = {
                "engine_loaded": True,
                "engine_state": "ready" if agent._language_ready() else "learning",
                "language_ready": bool(agent._language_ready()),
                "chat_generation_ready": bool(agent._chat_generation_ready()),
                "device": str(agent.device),
                "memory_count": int(agent.memory.summary().get("count", 0)),
                "lessons": int(agent.memory.lesson_count()),
                "experienced_lessons": int(agent.memory.experience_lesson_count()),
                "training_tokens": int(getattr(agent, "total_tokens", 0)),
                "training_steps": int(getattr(agent, "training_steps", 0)),
                "parameter_count": int(agent.parameter_count()),
                "context_tokens": int(agent.model.cfg.context),
                "speech_ready": bool(agent.speech.ready),
                "speech_steps": int(agent.speech.steps),
                "tts_ready": bool(agent.tts.ready),
                "tts_steps": int(agent.tts.steps),
                "camera_entities": int(len(agent.perception.visual_entities.entities)),
                "audio_steps": int(agent.perception.audio_steps),
                "language_bpb": agent.last_eval_bpb,
                "baseline_bpb": agent.eval_baseline_bpb,
            }
        except Exception as exc:
            self.last_status = {**self.last_status, "status_error": f"{type(exc).__name__}: {exc}"}
        return dict(self.last_status)

    def status(self) -> dict[str, Any]:
        if self.agent_lock.acquire(blocking=False):
            try:
                status = self._refresh_status_locked()
                return {
                    **status,
                    "version": APP_VERSION,
                    "busy": False,
                    "worker": dict(self.worker_status),
                    "agent_load_error": self.agent_load_error,
                    "conversation_count": self.conversation_count(),
                }
            finally:
                self.agent_lock.release()
        return {
            **dict(self.last_status), "version": APP_VERSION, "busy": True,
            "worker": dict(self.worker_status), "agent_load_error": self.agent_load_error,
            "conversation_count": self.conversation_count(),
        }

    def conversation_count(self) -> int:
        with self._connect() as db:
            return int(db.execute("SELECT COUNT(*) FROM chats WHERE archived=0").fetchone()[0])

    def list_chats(self) -> list[dict[str, Any]]:
        with self._connect() as db:
            rows = db.execute("""
                SELECT c.id,c.title,c.created_at,c.updated_at,
                    (SELECT COUNT(*) FROM chat_messages m WHERE m.chat_id=c.id) AS message_count,
                    (SELECT content FROM chat_messages m WHERE m.chat_id=c.id ORDER BY created_at DESC LIMIT 1) AS last_message
                FROM chats c WHERE c.archived=0 ORDER BY c.updated_at DESC
            """).fetchall()
        result = []
        for row in rows:
            item = dict(row)
            item["last_message"] = (item.get("last_message") or "")[:140]
            result.append(item)
        return result

    def create_chat(self, title: str = "New conversation") -> dict[str, Any]:
        chat_id = uuid.uuid4().hex
        name = " ".join(str(title).split()).strip() or "New conversation"
        now = time.time()
        with self._connect() as db:
            db.execute("INSERT INTO chats(id,title,created_at,updated_at) VALUES(?,?,?,?)", (chat_id, name[:120], now, now))
        return {"id": chat_id, "title": name[:120], "created_at": now, "updated_at": now, "message_count": 0, "last_message": ""}

    def get_chat(self, chat_id: str) -> dict[str, Any] | None:
        with self._connect() as db:
            row = db.execute("SELECT * FROM chats WHERE id=? AND archived=0", (chat_id,)).fetchone()
            if not row:
                return None
            messages = db.execute("SELECT id,role,content,created_at,response_mode,reply_to FROM chat_messages WHERE chat_id=? ORDER BY created_at,id", (chat_id,)).fetchall()
        return {"id": row["id"], "title": row["title"], "created_at": row["created_at"], "updated_at": row["updated_at"], "messages": [dict(x) for x in messages]}

    def _add_message(self, chat_id: str, role: str, content: str, *, response_mode: str = "", reply_to: str = "") -> dict[str, Any]:
        message_id = uuid.uuid4().hex
        now = time.time()
        with self._connect() as db:
            exists = db.execute("SELECT id FROM chats WHERE id=? AND archived=0", (chat_id,)).fetchone()
            if not exists:
                raise KeyError("chat not found")
            db.execute("INSERT INTO chat_messages(id,chat_id,role,content,created_at,response_mode,reply_to) VALUES(?,?,?,?,?,?,?)", (message_id, chat_id, role, content, now, response_mode, reply_to))
            if role == "user":
                count = int(db.execute("SELECT COUNT(*) FROM chat_messages WHERE chat_id=? AND role='user'", (chat_id,)).fetchone()[0])
                existing = db.execute("SELECT title FROM chats WHERE id=?", (chat_id,)).fetchone()
                if count == 1 and existing and str(existing[0]) in {"New conversation", "First conversation"}:
                    title = " ".join(str(content).split())[:54].strip() or "New conversation"
                    db.execute("UPDATE chats SET title=?,updated_at=? WHERE id=?", (title, now, chat_id))
                else:
                    db.execute("UPDATE chats SET updated_at=? WHERE id=?", (now, chat_id))
            else:
                db.execute("UPDATE chats SET updated_at=? WHERE id=?", (now, chat_id))
        return {"id": message_id, "chat_id": chat_id, "role": role, "content": content, "created_at": now, "response_mode": response_mode, "reply_to": reply_to}

    def worker_loop(self) -> None:
        self.worker_status["running"] = True
        while not self.worker_stop.is_set():
            if not self.auto_train:
                self.worker_stop.wait(1.0)
                continue
            try:
                with LearningPipeline(self.root) as queue:
                    queued = list(reversed(queue.list_jobs(limit=50, state="queued")))
                if not queued:
                    self.worker_stop.wait(1.0)
                    continue
                job_id = str(queued[0]["id"])
                self.worker_status.update({"current_job_id": job_id, "current_job_state": "starting", "last_error": None})
                try:
                    with self.agent_lock:
                        agent = self.get_agent()
                        self.worker_status["current_job_state"] = "training"
                        with LearningPipeline(self.root) as queue:
                            result = queue.run_job(job_id, agent)
                    self.worker_status["current_job_state"] = str(result.get("state", "finished"))
                    self._refresh_status_locked()
                except Exception as exc:
                    self.worker_status["last_error"] = f"{type(exc).__name__}: {exc}"
                    # run_job records job-specific errors; a setup failure must not stop future jobs.
                    try:
                        with LearningPipeline(self.root) as queue:
                            job = queue.get_job(job_id)
                            if job.get("state") == "queued":
                                queue._update(job_id, state="failed", progress=100, status_message="Worker could not initialize Aletheia.", error=self.worker_status["last_error"])
                    except Exception:
                        pass
                finally:
                    self.worker_status["current_job_id"] = None
                    self.worker_status["current_job_state"] = None
            except Exception as exc:
                self.worker_status["last_error"] = f"{type(exc).__name__}: {exc}"
                self.worker_stop.wait(2.0)
        self.worker_status["running"] = False

    def start_worker(self) -> None:
        if self.worker_thread and self.worker_thread.is_alive():
            return
        self.worker_stop.clear()
        self.worker_thread = threading.Thread(target=self.worker_loop, name="aletheia-learning-worker", daemon=True)
        self.worker_thread.start()

    def stop(self) -> None:
        self.worker_stop.set()
        if self.worker_thread and self.worker_thread.is_alive():
            self.worker_thread.join(timeout=3.0)
        with self.agent_lock:
            if self.agent is not None:
                try:
                    self.agent.save()
                    self.agent.close()
                finally:
                    self.agent = None


def _check_local_origin(request: Request) -> None:
    origin = request.headers.get("origin")
    if not origin:
        return
    parsed = urlparse(origin)
    if parsed.scheme not in {"http", "https"} or parsed.hostname not in {"127.0.0.1", "localhost", "::1"}:
        raise HTTPException(status_code=403, detail="This local-only Aletheia server accepts requests only from localhost origins.")


def create_app(root: Path | str | None = None, config_path: Path | str | None = None, *, auto_train: bool = True) -> FastAPI:
    project_root = Path(root or Path.cwd()).expanduser().resolve()
    config_file = Path(config_path or (project_root / "config.json")).expanduser().resolve()
    service = WebApplicationService(project_root, config_file, auto_train=auto_train)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.service = service
        service.start_worker()
        try:
            yield
        finally:
            service.stop()

    app = FastAPI(title="Aletheia", version=APP_VERSION, docs_url=None, redoc_url=None, lifespan=lifespan)
    app.state.service = service

    @app.middleware("http")
    async def local_host_and_security_headers(request: Request, call_next):
        host_header = request.headers.get("host", "").strip().lower()
        hostname = host_header.split("]", 1)[0][1:] if host_header.startswith("[") else host_header.split(":", 1)[0]
        if hostname and hostname not in {"127.0.0.1", "localhost", "::1", "testserver"}:
            return JSONResponse(status_code=403, content={"detail": "Aletheia web app is local-only; use localhost."})
        response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["Referrer-Policy"] = "same-origin"
        response.headers["Permissions-Policy"] = "camera=(self), microphone=(self), geolocation=(), payment=()"
        response.headers["Content-Security-Policy"] = "default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline'; img-src 'self' blob: data:; media-src 'self' blob:; connect-src 'self'; font-src 'self' data:; object-src 'none'; base-uri 'self'; frame-ancestors 'none'"
        return response

    app.mount("/static", StaticFiles(directory=Path(__file__).parent / "web" / "static"), name="static")

    @app.get("/", response_class=HTMLResponse)
    def index():
        index_path = Path(__file__).parent / "web" / "index.html"
        return HTMLResponse(index_path.read_text(encoding="utf-8"))

    @app.get("/api/health")
    def health():
        return {"ok": True, "app": "Aletheia", "version": APP_VERSION, "local_only": True}

    @app.get("/api/status")
    def status():
        return service.status()

    @app.get("/api/chats")
    def chats():
        return {"chats": service.list_chats()}

    @app.post("/api/chats")
    def create_chat(body: NewChatRequest, request: Request):
        _check_local_origin(request)
        return service.create_chat(body.title)

    @app.get("/api/chats/{chat_id}")
    def get_chat(chat_id: str):
        result = service.get_chat(chat_id)
        if result is None:
            raise HTTPException(status_code=404, detail="Conversation not found.")
        return result

    @app.post("/api/chats/{chat_id}/messages")
    def send_chat(chat_id: str, body: ChatRequest, request: Request):
        _check_local_origin(request)
        message = " ".join(body.message.strip().split())
        if not message:
            raise HTTPException(status_code=422, detail="Message cannot be empty.")
        if service.get_chat(chat_id) is None:
            raise HTTPException(status_code=404, detail="Conversation not found.")
        user_message = service._add_message(chat_id, "user", message)
        try:
            with service.agent_lock:
                agent = service.get_agent()
                reply = agent.communicate(message, max_tokens=body.max_tokens, source=f"web_chat:{chat_id}")
                target = dict(agent._last_feedback_target or {})
                response_mode = str(target.get("kind", "chat"))
                reply_mode = "experimental" if reply.startswith("[experimental]") else "generated"
                if reply.startswith("[quoted]"):
                    reply_mode = "memory"
                elif reply.startswith("[not_ready]"):
                    reply_mode = "not_ready"
                assistant_message = service._add_message(chat_id, "assistant", reply, response_mode=reply_mode, reply_to=user_message["id"])
                state = service._refresh_status_locked()
            return {"user_message": user_message, "assistant_message": assistant_message, "reply": reply, "response_mode": reply_mode, "language_ready": state.get("language_ready", False)}
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"Aletheia could not complete this message: {type(exc).__name__}: {exc}") from exc

    @app.post("/api/chats/{chat_id}/messages/{message_id}/feedback")
    def feedback(chat_id: str, message_id: str, body: FeedbackRequest, request: Request):
        _check_local_origin(request)
        with service._connect() as db:
            row = db.execute("SELECT * FROM chat_messages WHERE id=? AND chat_id=? AND role='assistant'", (message_id, chat_id)).fetchone()
            user_row = db.execute("SELECT * FROM chat_messages WHERE id=? AND chat_id=? AND role='user'", (row["reply_to"], chat_id)).fetchone() if row and row["reply_to"] else None
        if not row or not user_row:
            raise HTTPException(status_code=404, detail="Reply not found for feedback.")

        # Always persist feedback first. Only apply online learning when this response
        # is the exact most-recent engine target; old-chat ratings remain useful data
        # but must not accidentally train against another turn's target.
        now = time.time()
        with service._connect() as db:
            db.execute(
                """INSERT INTO message_feedback(message_id,positive,note,created_at,online_learning_applied,online_learning_json)
                   VALUES(?,?,?,?,0,'{}')
                   ON CONFLICT(message_id) DO UPDATE SET positive=excluded.positive,note=excluded.note,
                     created_at=excluded.created_at,online_learning_applied=0,online_learning_json='{}'""",
                (message_id, int(body.positive), body.note, now),
            )

        result: dict[str, Any] = {
            "updated": True,
            "stored": True,
            "online_learning": {"accepted": False, "reason": "feedback_saved_for_future_review; not applied to a different engine turn"},
        }
        try:
            with service.agent_lock:
                agent = service.agent  # Do not initialize a large model just to record a rating.
                if agent is not None:
                    target = dict(agent._last_feedback_target or {})
                    matches_target = (
                        str(target.get("message", "")) == str(user_row["content"])
                        and str(target.get("reply", "")) == str(row["content"])
                    )
                    if matches_target:
                        learned = agent.objective_feedback(body.positive, body.note)
                        result.update(learned)
                        result["stored"] = True
                        with service._connect() as db:
                            db.execute(
                                "UPDATE message_feedback SET online_learning_applied=1,online_learning_json=? WHERE message_id=?",
                                (json.dumps(learned, ensure_ascii=False), message_id),
                            )
                        service._refresh_status_locked()
            return result
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"Feedback was stored, but online learning failed: {type(exc).__name__}: {exc}") from exc

    @app.post("/api/learning/upload")
    async def upload_learning_file(
        request: Request,
        file: UploadFile = File(...),
        license_name: str = Form(default="unspecified; please confirm usage rights"),
        title: str = Form(default=""),
        auto_train: bool = Form(default=True),
    ):
        _check_local_origin(request)
        filename = Path(file.filename or "upload.txt").name
        suffix = Path(filename).suffix.lower()
        if suffix not in SUPPORTED_SUFFIXES:
            raise HTTPException(status_code=415, detail=f"Unsupported file type. Supported: {', '.join(sorted(SUPPORTED_SUFFIXES))}")
        upload_temp_dir = service.staging_dir / uuid.uuid4().hex
        upload_temp_dir.mkdir(parents=True, exist_ok=True)
        temp_path = upload_temp_dir / filename
        total = 0
        try:
            with temp_path.open("wb") as target:
                while True:
                    chunk = await file.read(1024 * 1024)
                    if not chunk:
                        break
                    total += len(chunk)
                    if total > DEFAULT_MAX_UPLOAD_BYTES:
                        raise HTTPException(status_code=413, detail=f"Upload exceeds the {DEFAULT_MAX_UPLOAD_BYTES // (1024 * 1024)} MiB limit.")
                    target.write(chunk)
            if total == 0:
                raise HTTPException(status_code=422, detail="Uploaded file is empty.")
            with LearningPipeline(service.root) as queue:
                result = queue.enqueue_file(temp_path, license_name=license_name, title=title or filename)
            service.auto_train = bool(auto_train)
            service.worker_status["enabled"] = service.auto_train
            if not service.worker_thread or not service.worker_thread.is_alive():
                service.start_worker()
            return {"job": result, "auto_train": service.auto_train, "message": "Queued for automatic guarded training." if service.auto_train and not result.get("duplicate") else ("This exact file already exists in the queue." if result.get("duplicate") else "Queued; start automatic training to process it.")}
        except HTTPException:
            raise
        except (FileNotFoundError, ValueError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"Could not stage learning material: {type(exc).__name__}: {exc}") from exc
        finally:
            temp_path.unlink(missing_ok=True)
            try:
                upload_temp_dir.rmdir()
            except OSError:
                pass
            await file.close()

    @app.post("/api/learning/text")
    def upload_learning_text(body: TextLearningRequest, request: Request):
        _check_local_origin(request)
        try:
            with LearningPipeline(service.root) as queue:
                result = queue.enqueue_text(body.text, title=body.title, license_name=body.license)
            service.auto_train = bool(body.auto_train)
            service.worker_status["enabled"] = service.auto_train
            if service.auto_train:
                if not service.worker_thread or not service.worker_thread.is_alive():
                    service.start_worker()
            return {"job": result, "auto_train": service.auto_train}
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.get("/api/learning/jobs")
    def learning_jobs(limit: int = 50, state: str | None = None):
        try:
            with LearningPipeline(service.root) as queue:
                return {"jobs": queue.list_jobs(limit=min(100, max(1, limit)), state=state)}
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.get("/api/learning/jobs/{job_id}")
    def learning_job(job_id: str):
        try:
            with LearningPipeline(service.root) as queue:
                return {"job": queue.get_job(job_id)}
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="Learning job not found.") from exc

    @app.get("/api/learning/jobs/{job_id}/report")
    def learning_job_report(job_id: str):
        try:
            with LearningPipeline(service.root) as queue:
                job = queue.get_job(job_id)
            report_path = Path(str(job.get("report_path", ""))).expanduser().resolve()
            report_root = (service.data_dir / ".learning_jobs" / "reports").resolve()
            if not report_path.is_relative_to(report_root) or not report_path.is_file():
                raise HTTPException(status_code=404, detail="No learning report is available for this job yet.")
            return FileResponse(report_path, media_type="application/json", filename=f"aletheia-learning-{job_id}.json")
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="Learning job not found.") from exc

    @app.post("/api/learning/jobs/{job_id}/run")
    def run_learning_job(job_id: str, request: Request):
        _check_local_origin(request)
        try:
            with LearningPipeline(service.root) as queue:
                job = queue.get_job(job_id)
            if job.get("state") != "queued":
                raise HTTPException(status_code=409, detail=f"Only queued jobs can be run; this job is {job.get('state')}.")
            service.auto_train = True
            service.worker_status["enabled"] = True
            if not service.worker_thread or not service.worker_thread.is_alive():
                service.start_worker()
            return {"job": job, "message": "Automatic worker enabled; this queued job will be processed in FIFO order."}
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="Learning job not found.") from exc

    @app.post("/api/learning/jobs/{job_id}/retry")
    def retry_learning_job(job_id: str, request: Request):
        _check_local_origin(request)
        try:
            with LearningPipeline(service.root) as queue:
                result = queue.retry(job_id)
            service.auto_train = True
            service.worker_status["enabled"] = True
            if not service.worker_thread or not service.worker_thread.is_alive():
                service.start_worker()
            return {"job": result}
        except (KeyError, ValueError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/api/perception/image")
    async def perceive_image(request: Request, file: UploadFile = File(...), alias: str = Form(default="")):
        _check_local_origin(request)
        suffix = Path(file.filename or "snapshot.jpg").suffix.lower()
        if suffix not in _IMAGE_SUFFIXES:
            raise HTTPException(status_code=415, detail="Use JPG, PNG or WebP images.")
        path = service.staging_dir / f"vision-{uuid.uuid4().hex}{suffix}"
        total = 0
        try:
            with path.open("wb") as output:
                while True:
                    chunk = await file.read(1024 * 512)
                    if not chunk:
                        break
                    total += len(chunk)
                    if total > MAX_IMAGE_BYTES:
                        raise HTTPException(status_code=413, detail="Image exceeds the 10 MiB limit.")
                    output.write(chunk)
            if not total:
                raise HTTPException(status_code=422, detail="Image is empty.")
            try:
                from PIL import Image
                with Image.open(path) as image:
                    image.verify()
            except Exception as exc:
                raise HTTPException(status_code=422, detail="The uploaded image could not be decoded.") from exc
            with service.agent_lock:
                agent = service.get_agent()
                result = agent.observe_image(path, alias=alias[:120], train=False)
                service._refresh_status_locked()
            # Do not retain the camera frame; only Aletheia's derived observation is persistent.
            result.pop("content_hash", None)
            return {"observation": result, "retained_image": False}
        except HTTPException:
            raise
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"Visual observation failed: {type(exc).__name__}: {exc}") from exc
        finally:
            path.unlink(missing_ok=True)
            await file.close()

    @app.post("/api/perception/audio")
    async def perceive_audio(request: Request, file: UploadFile = File(...)):
        _check_local_origin(request)
        suffix = Path(file.filename or "microphone.wav").suffix.lower()
        if suffix not in _AUDIO_SUFFIXES:
            raise HTTPException(status_code=415, detail="Aletheia's audio perception endpoint currently accepts WAV/PCM.")
        path = service.staging_dir / f"audio-{uuid.uuid4().hex}.wav"
        total = 0
        try:
            with path.open("wb") as output:
                while True:
                    chunk = await file.read(1024 * 512)
                    if not chunk:
                        break
                    total += len(chunk)
                    if total > MAX_AUDIO_BYTES:
                        raise HTTPException(status_code=413, detail="Audio exceeds the 30 MiB limit.")
                    output.write(chunk)
            if not total:
                raise HTTPException(status_code=422, detail="Audio is empty.")
            try:
                with wave.open(str(path), "rb") as wav:
                    if wav.getnchannels() not in {1, 2} or wav.getframerate() < 8000 or wav.getframerate() > 96000:
                        raise ValueError("Unsupported WAV channel count or sample rate.")
                    wav.getparams()
            except Exception as exc:
                raise HTTPException(status_code=422, detail=f"Invalid PCM WAV audio: {exc}") from exc
            with service.agent_lock:
                agent = service.get_agent()
                result = agent.hear_audio(path, train=False)
                service._refresh_status_locked()
            return {"observation": result, "transcribed": False, "retained_audio": False, "note": "Aletheia encoded an audio observation; this from-scratch encoder does not itself transcribe speech unless the speech model has been trained and is ready."}
        except HTTPException:
            raise
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"Audio observation failed: {type(exc).__name__}: {exc}") from exc
        finally:
            path.unlink(missing_ok=True)
            await file.close()

    @app.get("/api/voice/status")
    def voice_status():
        with service.agent_lock:
            try:
                agent = service.get_agent()
                return {"speech_ready": bool(agent.speech.ready), "speech_steps": int(agent.speech.steps), "learned_tts_ready": bool(agent.tts.ready), "tts_steps": int(agent.tts.steps), "speakers": agent.speakers.summary(), "speaker_profiles": [{"speaker_id": str(profile.get("speaker_id", "")), "aliases": [str(alias) for alias in profile.get("aliases", [])]} for profile in agent.speakers.profiles]}
            except Exception as exc:
                return JSONResponse(status_code=503, content={"error": f"{type(exc).__name__}: {exc}", "speech_ready": False, "learned_tts_ready": False})

    @app.post("/api/voice/enroll")
    async def enroll_voice(
        request: Request,
        file: UploadFile = File(...),
        alias: str = Form(...),
        consent_confirmed: bool = Form(default=False),
    ):
        _check_local_origin(request)
        if not consent_confirmed:
            raise HTTPException(status_code=400, detail="Confirm that you have permission to store and use this voice sample.")
        if Path(file.filename or "sample.wav").suffix.lower() != ".wav":
            raise HTTPException(status_code=415, detail="Use a WAV voice sample.")
        path = service.staging_dir / f"speaker-{uuid.uuid4().hex}.wav"
        total = 0
        try:
            with path.open("wb") as output:
                while True:
                    chunk = await file.read(1024 * 512)
                    if not chunk:
                        break
                    total += len(chunk)
                    if total > MAX_AUDIO_BYTES:
                        raise HTTPException(status_code=413, detail="Voice sample exceeds 30 MiB.")
                    output.write(chunk)
            with service.agent_lock:
                agent = service.get_agent()
                result = agent.enroll_speaker(path, alias=alias.strip()[:120])
                service._refresh_status_locked()
            return {"speaker": result, "learned_tts_ready": bool(service.agent.tts.ready), "note": "Speaker enrollment stores a learned audio identity. Voice synthesis requires a trained paired TTS model and does not clone a voice by itself."}
        except HTTPException:
            raise
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"Voice enrollment failed: {type(exc).__name__}: {exc}") from exc
        finally:
            path.unlink(missing_ok=True)
            await file.close()

    @app.post("/api/voice/synthesize")
    def synthesize_voice(body: dict[str, Any], request: Request):
        _check_local_origin(request)
        text = " ".join(str(body.get("text", "")).split()).strip()
        speaker = str(body.get("speaker", "")).strip()
        if not text or len(text) > 2500:
            raise HTTPException(status_code=422, detail="Provide text between 1 and 2500 characters.")
        if not speaker:
            raise HTTPException(status_code=422, detail="Choose an enrolled speaker alias or ID.")
        output = service.data_dir / ".webapp_audio" / f"{uuid.uuid4().hex}.wav"
        output.parent.mkdir(parents=True, exist_ok=True)
        try:
            with service.agent_lock:
                agent = service.get_agent()
                result = agent.synthesize_voice(text, speaker=speaker, output=output, play=False)
            if not output.is_file():
                raise RuntimeError("voice synthesizer did not write output audio")
            return FileResponse(output, media_type="audio/wav", filename="aletheia-voice.wav", background=BackgroundTask(output.unlink, missing_ok=True), headers={"X-Aletheia-Voice-Backend": str(result.get("backend", "learned_tts"))})
        except Exception as exc:
            output.unlink(missing_ok=True)
            raise HTTPException(status_code=409, detail=f"Learned voice synthesis is unavailable: {type(exc).__name__}: {exc}") from exc

    @app.get("/api/memory/search")
    def memory_search(q: str, limit: int = 8):
        query = " ".join(q.split()).strip()
        if not query or len(query) > 400:
            raise HTTPException(status_code=422, detail="Search text must be between 1 and 400 characters.")
        with service.agent_lock:
            agent = service.get_agent()
            results = agent.memory.search(query, max(1, min(20, limit)))
        safe = [{key: row.get(key) for key in ("id", "kind", "source", "title", "text", "created", "accepted", "replay_eligible")} for row in results]
        return {"query": query, "results": safe}

    @app.get("/favicon.ico")
    def favicon():
        path = Path(__file__).parent / "web" / "static" / "favicon.svg"
        return FileResponse(path, media_type="image/svg+xml")

    return app


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Aletheia's local-first web application.")
    parser.add_argument("--root", default=".", help="Aletheia project directory (default: current directory).")
    parser.add_argument("--config", default=None, help="Path to config.json (default: <root>/config.json).")
    parser.add_argument("--host", default="127.0.0.1", help="Loopback only for privacy; remote binding is intentionally disabled.")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--no-auto-train", action="store_true", help="Queue uploads but do not start learning jobs automatically.")
    args = parser.parse_args()
    if args.host not in {"127.0.0.1", "localhost", "::1"}:
        parser.error("FINAL34 is local-only and only binds to 127.0.0.1 / localhost / ::1.")
    if not (1 <= args.port <= 65535):
        parser.error("port must be between 1 and 65535")
    try:
        import uvicorn
    except ImportError as exc:
        parser.error("Install web dependencies first: pip install -e '.[webapp]'")
    root = Path(args.root).expanduser().resolve()
    config = Path(args.config).expanduser().resolve() if args.config else root / "config.json"
    app = create_app(root, config, auto_train=not args.no_auto_train)
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
