"""
MASIS RAG Pipeline — Hybrid retrieval (semantic + keyword) with re-ranking.

Handles:
  • Document ingestion & chunking
  • ChromaDB vector store management
  • Semantic search via embeddings
  • Keyword/metadata filtering
  • Reciprocal Rank Fusion for hybrid results
  • Context window management (avoids "lost in the middle")
"""

from __future__ import annotations

import hashlib
import logging
import os
import time
from pathlib import Path
from typing import Any

from langchain_community.document_loaders import (
    DirectoryLoader,
    TextLoader,
    PyPDFLoader,
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document

from masis.config import get_config, DOCUMENT_DIR, CHROMA_PERSIST_DIR

logger = logging.getLogger("masis")


# ──────────────────────────────────────────────────────────────
# Retrieval Cache — avoids duplicate searches within a session
# ──────────────────────────────────────────────────────────────

class _RetrievalCache:
    """Simple in-memory TTL cache for search results."""

    def __init__(self, ttl_seconds: int = 300):
        self._cache: dict[str, tuple[float, Any]] = {}
        self._ttl = ttl_seconds

    def get(self, key: str) -> Any | None:
        entry = self._cache.get(key)
        if entry is None:
            return None
        ts, value = entry
        if time.time() - ts > self._ttl:
            del self._cache[key]
            return None
        logger.debug("Cache HIT for key=%s", key[:40])
        return value

    def set(self, key: str, value: Any) -> None:
        self._cache[key] = (time.time(), value)

    def clear(self) -> None:
        self._cache.clear()


_search_cache = _RetrievalCache(ttl_seconds=300)


# ──────────────────────────────────────────────────────────────
# Document Ingestion
# ──────────────────────────────────────────────────────────────

_LOADER_MAP: dict[str, type] = {
    ".txt": TextLoader,
    ".md": TextLoader,
    ".pdf": PyPDFLoader,
}


def _load_documents(doc_dir: Path | None = None) -> list[Document]:
    """Load all supported documents from the document directory."""
    doc_dir = doc_dir or DOCUMENT_DIR
    doc_dir.mkdir(parents=True, exist_ok=True)

    documents: list[Document] = []
    for ext, loader_cls in _LOADER_MAP.items():
        for fpath in doc_dir.rglob(f"*{ext}"):
            try:
                loader = loader_cls(str(fpath))
                docs = loader.load()
                # Attach source metadata
                for doc in docs:
                    doc.metadata["source"] = fpath.name
                    doc.metadata["full_path"] = str(fpath)
                documents.extend(docs)
            except Exception as exc:
                print(f"[RAG] Warning: failed to load {fpath}: {exc}")
    return documents


def _chunk_documents(documents: list[Document]) -> list[Document]:
    """Split documents into chunks with overlap for context continuity."""
    cfg = get_config().rag
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=cfg.chunk_size,
        chunk_overlap=cfg.chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(documents)
    # Add deterministic chunk IDs for citation tracing
    for i, chunk in enumerate(chunks):
        source = chunk.metadata.get("source", "unknown")
        content_hash = hashlib.sha256(chunk.page_content.encode()).hexdigest()[:10]
        chunk.metadata["chunk_id"] = f"{source}::{i}::{content_hash}"
    return chunks


# ──────────────────────────────────────────────────────────────
# Vector Store Management
# ──────────────────────────────────────────────────────────────

_embeddings = None
_vectorstore = None


def _get_embeddings() -> OpenAIEmbeddings:
    global _embeddings
    if _embeddings is None:
        _embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    return _embeddings


def get_vectorstore(force_rebuild: bool = False) -> Chroma:
    """Get or create the ChromaDB vector store."""
    global _vectorstore
    if _vectorstore is not None and not force_rebuild:
        return _vectorstore

    persist_dir = str(CHROMA_PERSIST_DIR)
    embeddings = _get_embeddings()

    # If store exists on disk, load it
    if CHROMA_PERSIST_DIR.exists() and not force_rebuild:
        _vectorstore = Chroma(
            persist_directory=persist_dir,
            embedding_function=embeddings,
            collection_name="masis_docs",
        )
        return _vectorstore

    # Otherwise, ingest documents
    documents = _load_documents()
    if not documents:
        # Create an empty store
        CHROMA_PERSIST_DIR.mkdir(parents=True, exist_ok=True)
        _vectorstore = Chroma(
            persist_directory=persist_dir,
            embedding_function=embeddings,
            collection_name="masis_docs",
        )
        return _vectorstore

    chunks = _chunk_documents(documents)
    _vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=persist_dir,
        collection_name="masis_docs",
    )
    return _vectorstore


def ingest_documents(doc_dir: Path | None = None) -> int:
    """Ingest/re-ingest documents into the vector store. Returns chunk count."""
    documents = _load_documents(doc_dir)
    if not documents:
        return 0
    chunks = _chunk_documents(documents)
    embeddings = _get_embeddings()
    persist_dir = str(CHROMA_PERSIST_DIR)
    CHROMA_PERSIST_DIR.mkdir(parents=True, exist_ok=True)

    global _vectorstore
    _vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=persist_dir,
        collection_name="masis_docs",
    )
    return len(chunks)


# ──────────────────────────────────────────────────────────────
# Retrieval Strategies
# ──────────────────────────────────────────────────────────────

def semantic_search(query: str, top_k: int | None = None) -> list[Document]:
    """Pure vector similarity search."""
    cfg = get_config().rag
    k = top_k or cfg.top_k_semantic
    store = get_vectorstore()
    return store.similarity_search_with_relevance_scores(query, k=k)


def keyword_search(query: str, top_k: int | None = None) -> list[Document]:
    """
    Keyword/metadata filtering.
    Uses the vector store's built-in where_document filter for keyword matching.
    Falls back to semantic if the store doesn't support full-text search natively.
    """
    cfg = get_config().rag
    k = top_k or cfg.top_k_keyword
    store = get_vectorstore()
    # ChromaDB supports $contains for keyword filtering
    try:
        results = store.similarity_search(
            query, k=k,
            where_document={"$contains": query.split()[0]} if query.split() else {}
        )
        return [(doc, 0.5) for doc in results]  # Assign neutral score
    except Exception:
        # Fallback to semantic
        return semantic_search(query, top_k=k)


def hybrid_search(query: str) -> list[dict[str, Any]]:
    """
    Reciprocal Rank Fusion of semantic + keyword results.

    Returns a list of dicts with keys:
      - content, source, chunk_id, score, metadata
    Sorted by fused score descending.
    Applies "lost in the middle" mitigation by interleaving high and low rank items.

    Semantic and keyword searches run in parallel via ThreadPoolExecutor.
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed

    cfg = get_config().rag

    # Check cache first
    cache_key = f"hybrid:{query}"
    cached = _search_cache.get(cache_key)
    if cached is not None:
        return cached

    # Fan-out: run semantic and keyword searches concurrently
    with ThreadPoolExecutor(max_workers=2) as pool:
        sem_future = pool.submit(semantic_search, query)
        kw_future = pool.submit(keyword_search, query)
        semantic_results = sem_future.result()
        keyword_results = kw_future.result()

    # Build a map: chunk_id -> {content, source, scores}
    fused: dict[str, dict[str, Any]] = {}
    RRF_K = 60  # Standard RRF constant

    for rank, (doc, score) in enumerate(semantic_results):
        cid = doc.metadata.get("chunk_id", f"sem-{rank}")
        if cid not in fused:
            fused[cid] = {
                "content": doc.page_content,
                "source": doc.metadata.get("source", "unknown"),
                "chunk_id": cid,
                "metadata": doc.metadata,
                "rrf_score": 0.0,
                "semantic_score": score,
            }
        fused[cid]["rrf_score"] += 1.0 / (RRF_K + rank + 1)

    for rank, (doc, score) in enumerate(keyword_results):
        cid = doc.metadata.get("chunk_id", f"kw-{rank}")
        if cid not in fused:
            fused[cid] = {
                "content": doc.page_content,
                "source": doc.metadata.get("source", "unknown"),
                "chunk_id": cid,
                "metadata": doc.metadata,
                "rrf_score": 0.0,
                "semantic_score": 0.0,
            }
        fused[cid]["rrf_score"] += 1.0 / (RRF_K + rank + 1)

    # Sort by fused score
    ranked = sorted(fused.values(), key=lambda x: x["rrf_score"], reverse=True)
    top = ranked[: cfg.top_k_final]

    # "Lost in the middle" mitigation: place highest-scored items at edges
    if len(top) > 2:
        reordered = []
        for i, item in enumerate(top):
            if i % 2 == 0:
                reordered.append(item)
            else:
                reordered.insert(0, item)
        _search_cache.set(cache_key, reordered)
        return reordered

    _search_cache.set(cache_key, top)
    return top


def format_context(chunks: list[dict[str, Any]]) -> str:
    """Format retrieved chunks into a numbered context string for LLM consumption."""
    parts: list[str] = []
    for i, chunk in enumerate(chunks, 1):
        source = chunk.get("source", "unknown")
        cid = chunk.get("chunk_id", "n/a")
        content = chunk.get("content", "")
        parts.append(
            f"[Source {i}: {source} | ID: {cid}]\n{content}"
        )
    return "\n\n---\n\n".join(parts)
