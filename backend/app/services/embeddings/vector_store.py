import os
import joblib
import numpy as np
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Tuple
from app.core.config import settings
from app.core.logging import logger
from app.schemas.embedding_schemas import ChunkMetadata, VectorSearchResult

try:
    import faiss
except ImportError:
    faiss = None


class BaseVectorStore(ABC):
    """
    Abstract Vector Database Interface for modular vector store substitution.
    """

    @abstractmethod
    def add_vectors(self, vectors: np.ndarray, metadata_list: List[ChunkMetadata]) -> None:
        pass

    @abstractmethod
    def search(
        self, query_vector: np.ndarray, top_k: int = 5, document_ids: Optional[List[int]] = None
    ) -> List[VectorSearchResult]:
        pass

    @abstractmethod
    def get_stats(self) -> Dict[str, Any]:
        pass

    @abstractmethod
    def save(self) -> None:
        pass

    @abstractmethod
    def load(self) -> None:
        pass


class FAISSVectorStore(BaseVectorStore):
    """
    FAISS Vector Database Implementation using `IndexFlatIP` (Inner Product on L2-normalized vectors = Cosine Similarity).
    """

    def __init__(self, index_path: Optional[str] = None, dimension: int = 384):
        self.index_path = index_path or settings.FAISS_INDEX_PATH
        self.meta_path = self.index_path + ".meta.pkl"
        self.dimension = dimension

        self.index = None
        self.metadata_store: List[ChunkMetadata] = []

        os.makedirs(os.path.dirname(self.index_path), exist_ok=True)
        self.load()

    def _init_index(self):
        if self.index is None:
            if faiss is not None:
                self.index = faiss.IndexFlatIP(self.dimension)
            else:
                logger.warning("FAISS module not available. Using in-memory numpy vector store fallback.")
                self.index = "numpy_fallback"

    def add_vectors(self, vectors: np.ndarray, metadata_list: List[ChunkMetadata]) -> None:
        self._init_index()

        if isinstance(self.index, str) and self.index == "numpy_fallback":
            self.metadata_store.extend(metadata_list)
            # Store vectors directly inside metadata if using fallback
            for meta, vec in zip(metadata_list, vectors):
                setattr(meta, "_fallback_vector", vec)
            self.save()
            return

        if faiss is not None and hasattr(self.index, "add"):
            self.index.add(vectors.astype(np.float32))
            self.metadata_store.extend(metadata_list)
            self.save()

    def search(
        self, query_vector: np.ndarray, top_k: int = 5, document_ids: Optional[List[int]] = None
    ) -> List[VectorSearchResult]:
        self._init_index()
        results: List[VectorSearchResult] = []

        if len(self.metadata_store) == 0:
            return results

        if query_vector.ndim == 1:
            query_vector = np.expand_dims(query_vector, axis=0)

        if isinstance(self.index, str) and self.index == "numpy_fallback":
            # Numpy cosine similarity search
            query_dim = query_vector.shape[1] if query_vector.ndim == 2 else query_vector.shape[0]
            sims = []
            for idx, meta in enumerate(self.metadata_store):
                vec = getattr(meta, "_fallback_vector", None)
                if vec is not None and vec.shape[0] == query_dim:
                    score = float(np.dot(query_vector[0], vec))
                else:
                    score = 0.0
                sims.append((score, meta))

            sims.sort(key=lambda x: x[0], reverse=True)
            for score, meta in sims:
                if document_ids and meta.document_id not in document_ids:
                    continue
                results.append(
                    VectorSearchResult(
                        score=round(max(0.0, min(1.0, float(score))), 4),
                        chunk_id=meta.chunk_id,
                        document_id=meta.document_id,
                        filename=meta.filename,
                        page_number=meta.page_number,
                        text=meta.text,
                    )
                )
                if len(results) >= top_k:
                    break
            return results

        # FAISS search
        search_k = min(len(self.metadata_store), top_k * 5)
        scores, indices = self.index.search(query_vector.astype(np.float32), search_k)

        for score, idx in zip(scores[0], indices[0]):
            if idx < 0 or idx >= len(self.metadata_store):
                continue
            meta = self.metadata_store[idx]
            if document_ids and meta.document_id not in document_ids:
                continue

            results.append(
                VectorSearchResult(
                    score=round(max(0.0, min(1.0, float(score))), 4),
                    chunk_id=meta.chunk_id,
                    document_id=meta.document_id,
                    filename=meta.filename,
                    page_number=meta.page_number,
                    text=meta.text,
                )
            )
            if len(results) >= top_k:
                break

        return results

    def get_stats(self) -> Dict[str, Any]:
        count = len(self.metadata_store)
        return {
            "vector_store_type": "FAISS IndexFlatIP",
            "total_vectors": count,
            "dimension": self.dimension,
            "index_path": self.index_path,
        }

    def save(self) -> None:
        try:
            if faiss is not None and self.index and not isinstance(self.index, str):
                faiss.write_index(self.index, self.index_path)
            joblib.dump(self.metadata_store, self.meta_path)
        except Exception as e:
            logger.error(f"Error saving FAISS vector store index: {e}")

    def load(self) -> None:
        try:
            if os.path.exists(self.meta_path):
                self.metadata_store = joblib.load(self.meta_path)
            if faiss is not None and os.path.exists(self.index_path):
                self.index = faiss.read_index(self.index_path)
        except Exception as e:
            logger.warning(f"Failed to load existing FAISS index: {e}")


class VectorStoreFactory:
    """
    Factory creating configured VectorStore instances (FAISS vs ChromaDB adapter).
    """

    @staticmethod
    def get_vector_store() -> BaseVectorStore:
        return FAISSVectorStore()
