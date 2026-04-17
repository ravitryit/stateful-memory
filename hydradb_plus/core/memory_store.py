from __future__ import annotations

import os
import warnings
import sys

# Suppress noisy library logging and warnings
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
warnings.filterwarnings("ignore")

import hashlib
import math
import re
import shutil
import uuid
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import chromadb
from rich.console import Console
from rich.traceback import install as rich_install

rich_install(show_locals=False)
console = Console()


def _iso_now() -> str:
    """Return the current time as an ISO 8601 datetime string."""

    return datetime.now().isoformat()


@dataclass(frozen=True)
class MemoryResult:
    """Result item from MemoryStore retrieval."""

    chunk_id: str
    score: float
    metadata: Dict[str, Any]
    raw_text: Optional[str] = None
    enriched_text: Optional[str] = None
    keywords: Optional[List[str]] = None


@dataclass(frozen=True)
class MemoryStats:
    """Memory store statistics across all collections."""

    total_memories: int
    vcontent_size: int
    vlatent_size: int
    vsparse_size: int


class _FallbackHashEmbedder:
    """Deterministic hashing-based embedding for environments without model downloads.

    This approximates semantic similarity via a bag-of-words hash projection.
    """

    def __init__(self, dim: int = 384) -> None:
        """Initialize fallback embedder."""

        self.dim = dim

    def encode(self, texts: Sequence[str]) -> List[List[float]]:
        """Encode a list of texts into normalized vectors."""

        return [self._encode_one(t) for t in texts]

    def _encode_one(self, text: str) -> List[float]:
        """Encode a single text string into a normalized sparse-ish vector."""
        tokens = re.findall(r"[a-zA-Z0-9]+", text.lower())
        vec = [0.0] * self.dim
        for tok in tokens:
            h = hashlib.md5(tok.encode("utf-8")).hexdigest()
            idx = int(h[:8], 16) % self.dim
            vec[idx] += 1.0

        norm = math.sqrt(sum(x * x for x in vec))
        if norm <= 0.0:
            return [0.0] * self.dim
        return [x / norm for x in vec]


class MemoryStore:
    """ChromaDB wrapper storing raw, enriched, and sparse keyword signals."""

    def __init__(
        self,
        persist_dir: Optional[str] = None,
        embedding_model_name: str = "all-MiniLM-L6-v2",
        collection_prefix: str = "hydradb_plus",
        sparse_max_keywords: int = 15,
    ) -> None:
        """Initialize the MemoryStore.

        Args:
            persist_dir: Optional directory for Chroma persistence. If unset, it
                defaults to `hydradb_plus/.data/chroma`.
            embedding_model_name: SentenceTransformer model for dense embedding.
            collection_prefix: Prefix for chroma collection names.
            sparse_max_keywords: Max number of extracted keywords for sparse signal.
        """

        self._data_dir = Path(persist_dir) if persist_dir is not None else Path(__file__).resolve().parent.parent / ".data" / "chroma"
        self._data_dir.mkdir(parents=True, exist_ok=True)

        self._collection_prefix = collection_prefix
        self._sparse_max_keywords = sparse_max_keywords

        self._client = chromadb.PersistentClient(path=str(self._data_dir))

        try:
            self._vcontent_collection = self._client.get_or_create_collection(
                name=f"{collection_prefix}_vcontent", metadata={"hnsw:space": "cosine"}
            )
            self._vlatent_collection = self._client.get_or_create_collection(
                name=f"{collection_prefix}_vlatent", metadata={"hnsw:space": "cosine"}
            )
            self._vsparse_collection = self._client.get_or_create_collection(
                name=f"{collection_prefix}_vsparse", metadata={"hnsw:space": "cosine"}
            )
        except Exception as e:
            if "compaction" in str(e).lower() or "log store" in str(e).lower():
                console.print("[yellow]ChromaDB corruption detected on startup. Auto-healing...[/yellow]")
                self._reinit_collections(self._collection_prefix)
            else:
                raise e

        self._embedder_dense: Optional[Any] = None
        self._embedder_fallback = _FallbackHashEmbedder(dim=384)
        self._dense_model_name = embedding_model_name
        self._use_dense_embedder = False
        self._init_dense_embedder()

    def _init_dense_embedder(self) -> None:
        """Attempt to initialize SentenceTransformer; fallback to hashing if it fails."""

        try:
            from sentence_transformers import SentenceTransformer  # Local import for optional heavy dependency

            self._embedder_dense = SentenceTransformer(self._dense_model_name)
            self._use_dense_embedder = True
        except Exception:
            self._embedder_dense = None
            self._use_dense_embedder = False

    def _extract_keywords(self, text: str) -> List[str]:
        """Extract sparse keywords for BM25-style signal."""

        # Minimal stop-word list to keep the signal useful.
        stop_words = {
            "the",
            "a",
            "an",
            "and",
            "or",
            "to",
            "of",
            "in",
            "is",
            "it",
            "for",
            "on",
            "with",
            "as",
            "at",
            "by",
            "be",
            "are",
            "was",
            "were",
            "i",
            "you",
            "we",
            "they",
            "my",
            "your",
            "our",
            "their",
            "this",
            "that",
            "from",
        }

        tokens = re.findall(r"[a-zA-Z0-9]+", text.lower())
        tokens = [t for t in tokens if len(t) > 2 and t not in stop_words]
        freq: Dict[str, int] = {}
        for t in tokens:
            freq[t] = freq.get(t, 0) + 1

        # Keep top-K by term frequency.
        ranked = sorted(freq.items(), key=lambda kv: (-kv[1], kv[0]))
        return [t for t, _ in ranked[: self._sparse_max_keywords]]

    def _embed_texts(self, texts: Sequence[str]) -> List[List[float]]:
        """Embed a batch of texts into dense vectors."""

        if not texts:
            return []

        if self._use_dense_embedder and self._embedder_dense is not None:
            try:
                # SentenceTransformer returns numpy arrays; normalize for cosine search.
                vecs = self._embedder_dense.encode(list(texts), normalize_embeddings=True)
                return [v.tolist() for v in vecs]
            except Exception:
                # If anything fails (download, device, etc), fallback.
                return self._embedder_fallback.encode(texts)

        return self._embedder_fallback.encode(texts)

    def _query_collection(
        self, collection: Any, query_embedding: List[float], top_k: int
    ) -> List[Tuple[str, float, Dict[str, Any], Optional[str]]]:
        """Query a single chroma collection and return parsed results.

        Returns:
            List of tuples: (chunk_id, similarity_score, metadata, document_text).
        """

        res = collection.query(query_embeddings=[query_embedding], n_results=top_k)
        ids = res.get("ids", [[]])[0]
        distances = res.get("distances", [[]])[0]
        metadatas = res.get("metadatas", [[]])[0] if "metadatas" in res else [{}] * len(ids)
        documents = res.get("documents", [[]])[0] if "documents" in res else [None] * len(ids)

        out: List[Tuple[str, float, Dict[str, Any], Optional[str]]] = []
        for i, cid in enumerate(ids):
            dist = float(distances[i]) if i < len(distances) else 1.0
            similarity = 1.0 - dist
            meta = metadatas[i] if i < len(metadatas) and isinstance(metadatas[i], dict) else {}
            doc_text = documents[i] if i < len(documents) else None
            out.append((str(cid), similarity, meta, doc_text))
        return out

    def _reinit_collections(self, prefix: str) -> None:
        """Reset the client and collections, wiping the directory if needed."""
        try:
            # Try to release locks
            if hasattr(self, "_vcontent_collection"): del self._vcontent_collection
            if hasattr(self, "_vlatent_collection"): del self._vlatent_collection
            if hasattr(self, "_vsparse_collection"): del self._vsparse_collection
            if hasattr(self, "_client"): del self._client
            import gc
            gc.collect()

            if self._data_dir.exists():
                try:
                    shutil.rmtree(self._data_dir)
                except PermissionError:
                    # Silver Bullet: If we can't delete it (Windows lock), just move to a new versioned folder.
                    console.print("[yellow]Original chroma folder locked. Creating a new versioned storage...[/yellow]")
                    import time
                    suffix = int(time.time())
                    self._data_dir = self._data_dir.parent / f"chroma_{suffix}"
            
            self._data_dir.mkdir(parents=True, exist_ok=True)
            self._client = chromadb.PersistentClient(path=str(self._data_dir))
            self._vcontent_collection = self._client.get_or_create_collection(
                name=f"{prefix}_vcontent", metadata={"hnsw:space": "cosine"}
            )
            self._vlatent_collection = self._client.get_or_create_collection(
                name=f"{prefix}_vlatent", metadata={"hnsw:space": "cosine"}
            )
            self._vsparse_collection = self._client.get_or_create_collection(
                name=f"{prefix}_vsparse", metadata={"hnsw:space": "cosine"}
            )
        except Exception as e:
            console.print(f"[red]MemoryStore critical re-init failure: {e}[/red]")
            raise e

    def store_memory(
        self,
        chunk_id: str,
        raw_text: str,
        enriched_text: str,
        metadata: Dict[str, Any],
    ) -> None:
        """Store a memory chunk in ChromaDB as 3 embedded representations.

        Stored signals:
        1) Raw content embedding (vcontent)
        2) Enriched context embedding (vlatent)
        3) Sparse keyword embedding (vsparse), derived from extracted keywords
        """

        try:
            if not chunk_id:
                chunk_id = uuid.uuid4().hex

            keywords = self._extract_keywords(raw_text + " " + enriched_text)
            sparse_text = " ".join(keywords)

            vcontent, vlatent, vsparse = self._embed_texts([raw_text, enriched_text, sparse_text])

            metadata_out = dict(metadata)
            metadata_out.setdefault("created_at", _iso_now())

            # Convert any list values in metadata to strings (Chroma requirement)
            for k, v in metadata_out.items():
                if isinstance(v, list):
                    metadata_out[k] = ",".join([str(x) for x in v])

            # vcontent
            self._vcontent_collection.add(
                ids=[chunk_id],
                embeddings=[vcontent],
                documents=[raw_text],
                metadatas=[metadata_out],
            )
            # vlatent
            self._vlatent_collection.add(
                ids=[chunk_id],
                embeddings=[vlatent],
                documents=[enriched_text],
                metadatas=[metadata_out],
            )
            # vsparse
            meta_sparse = dict(metadata_out)
            # Ensure keywords is a string
            meta_sparse["keywords"] = ",".join(keywords) if keywords else ""
            self._vsparse_collection.add(
                ids=[chunk_id],
                embeddings=[vsparse],
                documents=[sparse_text],
                metadatas=[meta_sparse],
            )
        except Exception as e:
            err_msg = str(e).lower()
            if "compaction" in err_msg or "log store" in err_msg:
                console.print("[yellow]ChromaDB corruption detected during write. Wiping and retrying...[/yellow]")
                self._reinit_collections(self._collection_prefix)
                # Retry once
                return self.store_memory(chunk_id, raw_text, enriched_text, metadata)
            
            console.print("[red]MemoryStore.store_memory failed[/red]")
            console.print_exception(show_locals=False)
            raise e

    def retrieve(self, query: str, top_k: int = 5) -> List[MemoryResult]:
        """Retrieve top memories using weighted fusion across 3 representations."""

        try:
            if top_k <= 0:
                return []

            keywords = self._extract_keywords(query)
            sparse_text = " ".join(keywords)
            vcontent_q, vlatent_q, vsparse_q = self._embed_texts([query, query, sparse_text])

            vcontent_hits = self._query_collection(self._vcontent_collection, vcontent_q, top_k)
            vlatent_hits = self._query_collection(self._vlatent_collection, vlatent_q, top_k)
            vsparse_hits = self._query_collection(self._vsparse_collection, vsparse_q, top_k)

            weights = {"vcontent": 0.4, "vlatent": 0.4, "vsparse": 0.2}

            fused: Dict[str, float] = {}
            data: Dict[str, Dict[str, Any]] = {}

            def _accumulate(hits: List[Tuple[str, float, Dict[str, Any], Optional[str]]], key: str) -> None:
                for cid, sim, meta, doc in hits:
                    fused[cid] = fused.get(cid, 0.0) + weights[key] * sim
                    if cid not in data:
                        data[cid] = {"metadata": dict(meta), "raw_text": None, "enriched_text": None, "keywords": None}
                    if key == "vcontent" and doc is not None:
                        data[cid]["raw_text"] = doc
                    if key == "vlatent" and doc is not None:
                        data[cid]["enriched_text"] = doc
                    if key == "vsparse":
                        # Extract keywords from metadata if present
                        if "keywords" in meta and isinstance(meta["keywords"], list):
                            data[cid]["keywords"] = [str(x) for x in meta["keywords"]]
                        if doc is not None and data[cid].get("keywords") is None:
                            data[cid]["keywords"] = keywords

            _accumulate(vcontent_hits, "vcontent")
            _accumulate(vlatent_hits, "vlatent")
            _accumulate(vsparse_hits, "vsparse")

            ranked = sorted(fused.items(), key=lambda kv: kv[1], reverse=True)[:top_k]
            results: List[MemoryResult] = []
            for cid, score in ranked:
                md = data.get(cid, {}).get("metadata", {}) if cid in data else {}
                results.append(
                    MemoryResult(
                        chunk_id=cid,
                        score=float(score),
                        metadata=md,
                        raw_text=data[cid].get("raw_text"),
                        enriched_text=data[cid].get("enriched_text"),
                        keywords=data[cid].get("keywords"),
                    )
                )
            return results
        except Exception as e:
            console.print("[red]MemoryStore.retrieve failed[/red]")
            console.print_exception(show_locals=False)
            raise e

    def get_stats(self) -> MemoryStats:
        """Return collection sizes and total stored memory chunks."""

        try:
            vcontent_size = int(self._vcontent_collection.count())
            vlatent_size = int(self._vlatent_collection.count())
            vsparse_size = int(self._vsparse_collection.count())
            total_memories = min(vcontent_size, vlatent_size, vsparse_size)
            return MemoryStats(
                total_memories=total_memories,
                vcontent_size=vcontent_size,
                vlatent_size=vlatent_size,
                vsparse_size=vsparse_size,
            )
        except Exception as e:
            console.print("[red]MemoryStore.get_stats failed[/red]")
            console.print_exception(show_locals=False)
            raise e

