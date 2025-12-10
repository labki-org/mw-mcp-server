import faiss
import numpy as np
import json
from pathlib import Path
from typing import List, Tuple, Dict
from .models import IndexedDocument
from ..config import settings

class FaissIndex:
    def __init__(self):
        self.index: faiss.IndexIDMap | None = None
        self.doc_map: Dict[int, IndexedDocument] = {}
        self.next_id = 0

    def _init_index(self, dim: int):
        # Use IndexIDMap2 or IndexIDMap for explicit ID handling
        # IndexFlatIP is inner index
        self.index = faiss.IndexIDMap2(faiss.IndexFlatIP(dim))

    def add_documents(self, docs: List[IndexedDocument], embeddings: List[list[float]]):
        if not docs:
            return
            
        dim = len(embeddings[0])
        if self.index is None:
            self._init_index(dim)
            
        ids = np.arange(self.next_id, self.next_id + len(docs))
        self.next_id += len(docs)
        
        vecs = np.array(embeddings, dtype="float32")
        faiss.normalize_L2(vecs)
        self.index.add_with_ids(vecs, ids)
        
        for i, doc in zip(ids, docs):
            self.doc_map[int(i)] = doc

    # Backwards compatibility / full build
    def build(self, embeddings: List[list[float]], docs: List[IndexedDocument]):
        self.doc_map = {}
        self.next_id = 0
        self.index = None
        self.add_documents(docs, embeddings)

    def delete_page(self, page_title: str) -> int:
        """Removes all chunks associated with a page title. Returns count removed."""
        if self.index is None:
            return 0
            
        ids_to_remove = []
        for doc_id, doc in self.doc_map.items():
            if doc.page_title == page_title:
                ids_to_remove.append(doc_id)
                
        if not ids_to_remove:
            return 0
            
        # Remove from index
        ids_array = np.array(ids_to_remove, dtype="int64")
        self.index.remove_ids(ids_array)
        
        # Remove from map
        for i in ids_to_remove:
            del self.doc_map[i]
            
        return len(ids_to_remove)

    def search(self, query_emb: list[float], k: int = 5) -> List[Tuple[IndexedDocument, float]]:
        if self.index is None or not self.doc_map:
            return []
            
        q = np.array([query_emb], dtype="float32")
        faiss.normalize_L2(q)
        scores, idxs = self.index.search(q, k)
        results = []
        for score, idx in zip(scores[0], idxs[0]):
            idx = int(idx)
            if idx == -1 or idx not in self.doc_map:
                continue
            results.append((self.doc_map[idx], float(score)))
        return results

    def get_stats(self) -> dict:
        unique_pages = set()
        page_timestamps = {}
        for d in self.doc_map.values():
            unique_pages.add(d.page_title)
            # Keep the latest timestamp seen for the page
            if d.last_modified:
                if d.page_title not in page_timestamps:
                    page_timestamps[d.page_title] = d.last_modified
                else:
                    # Simple string compare for ISO timestamps works
                    if d.last_modified > page_timestamps[d.page_title]:
                        page_timestamps[d.page_title] = d.last_modified

        return {
            "total_vectors": self.index.ntotal if self.index else 0,
            "total_pages": len(unique_pages),
            "embedded_pages": sorted(list(unique_pages)),
            "page_timestamps": page_timestamps
        }

    def save(self):
        if self.index is None:
            return
        Path(settings.vector_index_path).parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, settings.vector_index_path)
        
        meta = {
            "next_id": self.next_id,
            "doc_map": {str(k): v.model_dump() for k, v in self.doc_map.items()}
        }
        with open(settings.vector_meta_path, "w") as f:
            json.dump(meta, f)

    def load(self):
        idx_path = Path(settings.vector_index_path)
        if not idx_path.exists():
            return
        
        self.index = faiss.read_index(str(idx_path))
        
        if Path(settings.vector_meta_path).exists():
            with open(settings.vector_meta_path) as f:
                data = json.load(f)
                self.next_id = data.get("next_id", 0)
                # Convert keys back to int
                self.doc_map = {int(k): IndexedDocument(**v) for k, v in data.get("doc_map", {}).items()}
        else:
            self.doc_map = {}
            self.next_id = 0
