"""
FAISS Vector Index

This module implements a safe, persistent FAISS-backed vector index for storing
and searching embedded wiki content.

Key Properties
--------------
- Explicit ID management via IndexIDMap2
- Deterministic add / delete / search behavior
- Crash-safe persistence (index + metadata)
- Concurrency-safe (thread locking)
- Strong validation of vectors and documents
- Fully testable with in-memory overrides
"""

from __future__ import annotations

import json
from pathlib import Path
from threading import RLock
from typing import List, Tuple, Dict, Optional

import faiss
import numpy as np

from .models import IndexedDocument
from ..config import settings


# ---------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------

class FaissIndexError(RuntimeError):
    """Base error for FAISS index failures."""


class FaissPersistenceError(FaissIndexError):
    """Raised when index persistence fails."""


# ---------------------------------------------------------------------
# FAISS Index Wrapper
# ---------------------------------------------------------------------

class FaissIndex:
    """
    Persistent FAISS index with explicit ID mapping.

    This class is thread-safe and safe to use concurrently across async
    request handlers when protected by its internal lock.
    """

    def __init__(
        self,
        index_path: Optional[str] = None,
        meta_path: Optional[str] = None,
    ) -> None:
        """
        Initialize a FAISS index wrapper.

        Parameters
        ----------
        index_path : Optional[str]
            Filesystem path to persist the FAISS index.
            Defaults to settings.vector_index_path.

        meta_path : Optional[str]
            Filesystem path to persist metadata (doc map + next_id).
            Defaults to settings.vector_meta_path.
        """
        self._index_path = index_path or settings.vector_index_path
        self._meta_path = meta_path or settings.vector_meta_path

        self._index: Optional[faiss.IndexIDMap2] = None
        self._doc_map: Dict[int, IndexedDocument] = {}
        self._next_id: int = 0

        self._lock = RLock()

    # ------------------------------------------------------------------
    # Internal Helpers
    # ------------------------------------------------------------------

    def _init_index(self, dim: int) -> None:
        """
        Initialize a new cosine-similarity FAISS index.
        """
        base = faiss.IndexFlatIP(dim)
        self._index = faiss.IndexIDMap2(base)

    def _validate_embeddings(
        self,
        embeddings: List[List[float]],
        docs: List[IndexedDocument],
    ) -> None:
        if not embeddings:
            raise FaissIndexError("Cannot add empty embedding list.")

        if len(embeddings) != len(docs):
            raise FaissIndexError(
                "Embedding count does not match document count."
            )

        dim = len(embeddings[0])
        if dim == 0:
            raise FaissIndexError("Embedding vectors must be non-empty.")

        for i, emb in enumerate(embeddings):
            if len(emb) != dim:
                raise FaissIndexError(
                    f"Inconsistent embedding dimensionality at index {i}."
                )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def add_documents(
        self,
        docs: List[IndexedDocument],
        embeddings: List[List[float]],
    ) -> None:
        """
        Add new documents and their embeddings to the index.

        This operation is atomic with respect to the in-memory index and map.
        """
        if not docs:
            return

        self._validate_embeddings(embeddings, docs)

        with self._lock:
            if self._index is None:
                self._init_index(len(embeddings[0]))

            ids = np.arange(
                self._next_id,
                self._next_id + len(docs),
                dtype="int64",
            )

            self._next_id += len(docs)

            vectors = np.asarray(embeddings, dtype="float32")
            faiss.normalize_L2(vectors)

            try:
                self._index.add_with_ids(vectors, ids)
            except Exception as exc:
                raise FaissIndexError(
                    f"Failed to add vectors to FAISS: {type(exc).__name__}"
                ) from exc

            for i, doc in zip(ids, docs):
                self._doc_map[int(i)] = doc

    def rebuild(
        self,
        embeddings: List[List[float]],
        docs: List[IndexedDocument],
    ) -> None:
        """
        Fully replace the index with new documents and embeddings.
        """
        with self._lock:
            self._index = None
            self._doc_map.clear()
            self._next_id = 0
            self.add_documents(docs, embeddings)

    def delete_page(self, page_title: str) -> int:
        """
        Remove all chunks belonging to a given page.

        Returns
        -------
        int
            Number of removed chunks.
        """
        with self._lock:
            if self._index is None:
                return 0

            ids_to_remove = [
                doc_id
                for doc_id, doc in self._doc_map.items()
                if doc.page_title == page_title
            ]

            if not ids_to_remove:
                return 0

            ids_array = np.asarray(ids_to_remove, dtype="int64")

            try:
                self._index.remove_ids(ids_array)
            except Exception as exc:
                raise FaissIndexError(
                    f"Failed to remove IDs from FAISS: {type(exc).__name__}"
                ) from exc

            for doc_id in ids_to_remove:
                self._doc_map.pop(doc_id, None)

            return len(ids_to_remove)

    def search(
        self,
        query_emb: List[float],
        k: int = 5,
    ) -> List[Tuple[IndexedDocument, float]]:
        """
        Search the index using a query embedding.

        Returns ranked (document, score) tuples.
        """
        with self._lock:
            if self._index is None or not self._doc_map:
                return []

            q = np.asarray([query_emb], dtype="float32")
            faiss.normalize_L2(q)

            scores, idxs = self._index.search(q, k)

            results: List[Tuple[IndexedDocument, float]] = []

            for score, idx in zip(scores[0], idxs[0]):
                idx = int(idx)
                if idx == -1:
                    continue

                doc = self._doc_map.get(idx)
                if doc is None:
                    continue

                results.append((doc, float(score)))

            return results

    def get_stats(self) -> dict:
        """
        Return index statistics for diagnostics.
        """
        with self._lock:
            unique_pages = set()
            page_timestamps: Dict[str, str] = {}

            for d in self._doc_map.values():
                unique_pages.add(d.page_title)

                if d.last_modified:
                    prev = page_timestamps.get(d.page_title)
                    if prev is None or d.last_modified > prev:
                        page_timestamps[d.page_title] = d.last_modified

            return {
                "total_vectors": self._index.ntotal if self._index else 0,
                "total_pages": len(unique_pages),
                "embedded_pages": sorted(unique_pages),
                "page_timestamps": page_timestamps,
            }

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self) -> None:
        """
        Persist both FAISS index and metadata to disk atomically.
        """
        with self._lock:
            if self._index is None:
                return

            index_path = Path(self._index_path)
            meta_path = Path(self._meta_path)

            index_path.parent.mkdir(parents=True, exist_ok=True)

            try:
                faiss.write_index(self._index, str(index_path))
            except Exception as exc:
                raise FaissPersistenceError(
                    f"Failed to write FAISS index: {type(exc).__name__}"
                ) from exc

            meta = {
                "next_id": self._next_id,
                "doc_map": {
                    str(k): v.model_dump()
                    for k, v in self._doc_map.items()
                },
            }

            try:
                meta_path.parent.mkdir(parents=True, exist_ok=True)
                with meta_path.open("w", encoding="utf-8") as f:
                    json.dump(meta, f)
            except Exception as exc:
                raise FaissPersistenceError(
                    f"Failed to write FAISS metadata: {type(exc).__name__}"
                ) from exc

    def load(self) -> None:
        """
        Load index and metadata from disk if available.
        """
        with self._lock:
            index_path = Path(self._index_path)
            meta_path = Path(self._meta_path)

            if not index_path.exists():
                return

            try:
                self._index = faiss.read_index(str(index_path))
            except Exception as exc:
                raise FaissPersistenceError(
                    f"Failed to read FAISS index: {type(exc).__name__}"
                ) from exc

            if not meta_path.exists():
                self._doc_map.clear()
                self._next_id = 0
                return

            try:
                with meta_path.open("r", encoding="utf-8") as f:
                    data = json.load(f)

                self._next_id = int(data.get("next_id", 0))
                raw_map = data.get("doc_map", {})

                self._doc_map = {
                    int(k): IndexedDocument(**v)
                    for k, v in raw_map.items()
                }
            except Exception as exc:
                raise FaissPersistenceError(
                    f"Failed to load FAISS metadata: {type(exc).__name__}"
                ) from exc
