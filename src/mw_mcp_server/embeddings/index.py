import faiss
import numpy as np
import json
from pathlib import Path
from typing import List, Tuple
from .models import IndexedDocument
from ..config import settings

class FaissIndex:
    def __init__(self):
        self.index: faiss.Index | None = None
        self.docs: List[IndexedDocument] = []

    def build(self, embeddings: List[list[float]], docs: List[IndexedDocument]):
        if not embeddings:
            return
        dim = len(embeddings[0])
        self.index = faiss.IndexFlatIP(dim)
        vecs = np.array(embeddings, dtype="float32")
        faiss.normalize_L2(vecs)
        self.index.add(vecs)
        self.docs = docs

    def search(self, query_emb: list[float], k: int = 5) -> List[Tuple[IndexedDocument, float]]:
        if self.index is None:
            return []
        q = np.array([query_emb], dtype="float32")
        faiss.normalize_L2(q)
        scores, idxs = self.index.search(q, k)
        results = []
        for score, idx in zip(scores[0], idxs[0]):
            if idx == -1:
                continue
            if idx < len(self.docs):
                results.append((self.docs[idx], float(score)))
        return results

    def save(self):
        if self.index is None:
            return
        Path(settings.vector_index_path).parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, settings.vector_index_path)
        with open(settings.vector_meta_path, "w") as f:
            json.dump([d.model_dump() for d in self.docs], f)

    def load(self):
        idx_path = Path(settings.vector_index_path)
        if not idx_path.exists():
            return
        self.index = faiss.read_index(str(idx_path))
        if Path(settings.vector_meta_path).exists():
            with open(settings.vector_meta_path) as f:
                data = json.load(f)
            self.docs = [IndexedDocument(**d) for d in data]
        else:
             self.docs = []
