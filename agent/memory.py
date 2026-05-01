"""Long-term memory for the agent.

Two stores:
- SQLite: structured facts/preferences with exact-key lookup.
- Qdrant: embedded session summaries for semantic retrieval.

Retrieval is synchronous on purpose — the LangGraph `call_model` node is
sync, and retrieval happens at most once per user turn.  Embedding on CPU
for a short query is <100ms; Qdrant HNSW search is single-digit ms.

Everything is a no-op when settings.memory_enabled is false, so the rest
of the codebase can call into MemoryStore unconditionally.
"""

from __future__ import annotations

import sqlite3
import threading
import time
import uuid
from dataclasses import dataclass
from pathlib import Path

import structlog

from agent.config import settings

log = structlog.get_logger("agent.memory")

_SQLITE_SCHEMA = """
CREATE TABLE IF NOT EXISTS facts (
    key        TEXT PRIMARY KEY,
    value      TEXT NOT NULL,
    source     TEXT NOT NULL,
    created_at REAL NOT NULL,
    updated_at REAL NOT NULL
);
"""

_INIT_RETRY_INTERVAL_S = 60.0


@dataclass
class MemoryHit:
    text: str
    score: float
    session_id: str
    created_at: float


class MemoryStore:
    """Singleton wrapper around the SQLite + Qdrant + embedder trio.

    Thread-safe for the expected access pattern (one writer, a few readers).
    Heavy resources load lazily on first use so disabling memory costs nothing.
    """

    _instance: MemoryStore | None = None
    _lock = threading.Lock()

    def __init__(self) -> None:
        self._embedder = None
        self._reranker = None
        self._qdrant = None
        self._sqlite: sqlite3.Connection | None = None
        self._dim: int | None = None
        self._ready = False
        self._last_init_attempt: float = 0.0

    # -- construction ------------------------------------------------------
    @classmethod
    def instance(cls) -> MemoryStore:
        with cls._lock:
            if cls._instance is None:
                cls._instance = cls()
            return cls._instance

    def _ensure_ready(self) -> bool:
        if self._ready:
            return True
        if not settings.memory_enabled:
            return False
        now = time.time()
        if now - self._last_init_attempt < _INIT_RETRY_INTERVAL_S:
            return False
        self._last_init_attempt = now
        try:
            self._init_sqlite()
            self._init_embedder()
            self._init_qdrant()
            self._ready = True
            log.info(
                "memory_ready",
                embed_model=settings.memory_embed_model,
                embed_device=settings.memory_embed_device,
                rerank_model=settings.memory_rerank_model or None,
                collection=settings.memory_qdrant_collection,
                dim=self._dim,
            )
        except Exception as e:
            log.error("memory_init_failed", error=str(e))
            self._ready = False
        return self._ready

    def _init_sqlite(self) -> None:
        path = Path(settings.memory_sqlite_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(str(path), check_same_thread=False, isolation_level=None)
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.executescript(_SQLITE_SCHEMA)
        self._sqlite = conn

    def _init_embedder(self) -> None:
        from sentence_transformers import SentenceTransformer

        try:
            self._embedder = SentenceTransformer(
                settings.memory_embed_model,
                device=settings.memory_embed_device,
                trust_remote_code=True,  # required by nomic-embed
            )
        except Exception as e:
            # Network blip during HF metadata check shouldn't kill init when
            # the model is already cached locally.
            log.warning("embedder_online_failed_using_cache", error=str(e))
            self._embedder = SentenceTransformer(
                settings.memory_embed_model,
                device=settings.memory_embed_device,
                trust_remote_code=True,
                local_files_only=True,
            )
        self._dim = int(self._embedder.get_sentence_embedding_dimension())

        if settings.memory_rerank_model:
            try:
                from sentence_transformers import CrossEncoder

                self._reranker = CrossEncoder(
                    settings.memory_rerank_model,
                    device=settings.memory_rerank_device,
                    trust_remote_code=True,
                )
            except Exception as e:
                log.warning("reranker_load_failed", error=str(e))
                self._reranker = None

    def _init_qdrant(self) -> None:
        from qdrant_client import QdrantClient
        from qdrant_client.models import Distance, VectorParams

        self._qdrant = QdrantClient(url=settings.memory_qdrant_url)
        name = settings.memory_qdrant_collection
        existing = {c.name for c in self._qdrant.get_collections().collections}
        if name not in existing:
            self._qdrant.create_collection(
                collection_name=name,
                vectors_config=VectorParams(size=self._dim, distance=Distance.COSINE),
            )

    # -- vector memory (summaries) -----------------------------------------
    def _embed(self, text: str) -> list[float]:
        # nomic prefixes — "search_document:" for stored items, "search_query:" for queries
        vec = self._embedder.encode(text, normalize_embeddings=True)
        return vec.tolist()

    def write_summary(self, text: str, session_id: str | None = None) -> str | None:
        """Embed a session summary and store it. Returns the point id."""
        if not self._ensure_ready() or not text.strip():
            return None
        from qdrant_client.models import PointStruct

        prefixed = f"search_document: {text}"
        vec = self._embed(prefixed)
        point_id = str(uuid.uuid4())
        payload = {
            "text": text,
            "session_id": session_id or point_id,
            "created_at": time.time(),
        }
        self._qdrant.upsert(
            collection_name=settings.memory_qdrant_collection,
            points=[PointStruct(id=point_id, vector=vec, payload=payload)],
        )
        log.info("memory_summary_stored", session_id=payload["session_id"], chars=len(text))
        return point_id

    def retrieve(self, query: str, k: int | None = None) -> list[MemoryHit]:
        """Top-k summaries for a query, optionally reranked."""
        if not self._ensure_ready() or not query.strip():
            return []
        k = k or settings.memory_top_k
        candidates_n = max(k, settings.memory_rerank_candidates if self._reranker else k)

        vec = self._embed(f"search_query: {query}")
        res = self._qdrant.query_points(
            collection_name=settings.memory_qdrant_collection,
            query=vec,
            limit=candidates_n,
        )
        hits = res.points
        results = [
            MemoryHit(
                text=h.payload.get("text", ""),
                score=float(h.score),
                session_id=h.payload.get("session_id", ""),
                created_at=float(h.payload.get("created_at", 0.0)),
            )
            for h in hits
        ]

        if self._reranker and len(results) > 1:
            pairs = [(query, r.text) for r in results]
            scores = self._reranker.predict(pairs).tolist()
            for r, s in zip(results, scores):
                r.score = float(s)
            results.sort(key=lambda r: r.score, reverse=True)

        threshold = settings.memory_score_threshold
        filtered = [r for r in results if r.score >= threshold]
        return filtered[:k]

    # -- structured facts --------------------------------------------------
    def set_fact(self, key: str, value: str, source: str = "user") -> None:
        if not self._ensure_ready():
            return
        now = time.time()
        self._sqlite.execute(
            """
            INSERT INTO facts(key, value, source, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(key) DO UPDATE SET
                value=excluded.value,
                source=excluded.source,
                updated_at=excluded.updated_at
            """,
            (key, value, source, now, now),
        )

    def get_fact(self, key: str) -> str | None:
        if not self._ensure_ready():
            return None
        row = self._sqlite.execute(
            "SELECT value FROM facts WHERE key = ?", (key,)
        ).fetchone()
        return row[0] if row else None

    def list_facts(self) -> list[tuple[str, str, str]]:
        """Return [(key, value, source), ...]."""
        if not self._ensure_ready():
            return []
        return self._sqlite.execute(
            "SELECT key, value, source FROM facts ORDER BY updated_at DESC"
        ).fetchall()

    def forget_fact(self, key: str) -> bool:
        if not self._ensure_ready():
            return False
        cur = self._sqlite.execute("DELETE FROM facts WHERE key = ?", (key,))
        return cur.rowcount > 0


def get_store() -> MemoryStore:
    return MemoryStore.instance()
