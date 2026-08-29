from typing import List, Optional, Dict
import numpy as np
from app.schemas.embedding_schemas import ChunkMetadata, VectorSearchResult
from app.services.embeddings.embedding_service import EmbeddingService
from app.services.embeddings.vector_store import VectorStoreFactory
from app.services.embeddings.bm25_retriever import BM25Retriever
from app.core.logging import logger


class HybridRetriever:
    """
    Hybrid Search Engine combining:
    1. Dense Semantic Search (FAISS IndexFlatIP via sentence-transformers)
    2. Sparse Lexical Search (Okapi BM25)
    3. Reciprocal Rank Fusion (RRF) for robust multi-modal ranking.
    """

    def __init__(self, rrf_k: int = 60):
        self.embedding_service = EmbeddingService()
        self.vector_store = VectorStoreFactory.get_vector_store()
        self.bm25 = BM25Retriever()
        self.rrf_k = rrf_k
        self._sync_bm25_from_vector_store()

    def _sync_bm25_from_vector_store(self):
        """Sync BM25 index with current metadata in FAISS vector store."""
        try:
            if hasattr(self.vector_store, "metadata") and self.vector_store.metadata:
                self.bm25.index_chunks(self.vector_store.metadata)
        except Exception as e:
            logger.warning(f"Failed to sync BM25 index: {e}")

    def index_chunks(self, chunks: List[ChunkMetadata]):
        """Update both dense vector store and BM25 index."""
        self.bm25.index_chunks(chunks)

    def search(
        self,
        query: str,
        top_k: int = 4,
        dense_weight: float = 0.5,
        sparse_weight: float = 0.5,
        document_ids: Optional[List[int]] = None,
    ) -> List[VectorSearchResult]:
        """
        Execute Hybrid Search using Reciprocal Rank Fusion:
        RRF_score(d) = w_dense / (k + rank_dense(d)) + w_sparse / (k + rank_sparse(d))
        """
        # Ensure BM25 index is up to date
        if self.bm25.total_docs == 0:
            self._sync_bm25_from_vector_store()

        fetch_k = top_k * 3

        # 1. Dense Semantic Search
        query_vector = self.embedding_service.encode([query])
        dense_results = self.vector_store.search(
            query_vector=query_vector,
            top_k=fetch_k,
            document_ids=document_ids,
        )

        # 2. Sparse BM25 Search
        sparse_results = self.bm25.search(
            query=query,
            top_k=fetch_k,
            document_ids=document_ids,
        )

        # 3. Reciprocal Rank Fusion (RRF)
        rrf_scores: Dict[str, float] = {}
        chunk_map: Dict[str, VectorSearchResult] = {}

        # Process dense ranks
        for rank, res in enumerate(dense_results, start=1):
            chunk_map[res.chunk_id] = res
            score = dense_weight / (self.rrf_k + rank)
            rrf_scores[res.chunk_id] = rrf_scores.get(res.chunk_id, 0.0) + score

        # Process sparse ranks
        for rank, res in enumerate(sparse_results, start=1):
            if res.chunk_id not in chunk_map:
                chunk_map[res.chunk_id] = res
            score = sparse_weight / (self.rrf_k + rank)
            rrf_scores[res.chunk_id] = rrf_scores.get(res.chunk_id, 0.0) + score

        # Sort by final fused RRF score
        sorted_chunk_ids = sorted(
            rrf_scores.keys(), key=lambda cid: rrf_scores[cid], reverse=True
        )

        final_results: List[VectorSearchResult] = []
        max_rrf = rrf_scores[sorted_chunk_ids[0]] if sorted_chunk_ids else 1.0

        for cid in sorted_chunk_ids[:top_k]:
            orig = chunk_map[cid]
            norm_score = round(rrf_scores[cid] / (max_rrf + 1e-9), 4)
            final_results.append(
                VectorSearchResult(
                    chunk_id=orig.chunk_id,
                    document_id=orig.document_id,
                    filename=orig.filename,
                    page_number=orig.page_number,
                    text=orig.text,
                    score=norm_score,
                )
            )

        return final_results
