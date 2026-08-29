from typing import List, Optional, Dict
from app.schemas.genai_schemas import CitationEdge, CitationNetworkResponse
from app.services.embeddings.embedding_service import EmbeddingService
from app.services.embeddings.vector_store import VectorStoreFactory
from app.services.nlp.keyword_extractor import KeywordExtractor
import numpy as np


class CitationNetworkBuilder:
    """
    Builds a conceptual citation/similarity network between documents
    using their vector embeddings and shared keyword concepts.
    This is an additional feature that enables researchers to visualise
    how closely related their uploaded papers are to each other.
    """

    def __init__(self):
        self.embedding_service = EmbeddingService()
        self.vector_store = VectorStoreFactory.get_vector_store()
        self.keyword_extractor = KeywordExtractor()

    def build_network(
        self,
        documents: List[Dict],  # [{"id": int, "filename": str, "text": str}]
        similarity_threshold: float = 0.20,
    ) -> CitationNetworkResponse:

        if not documents:
            return CitationNetworkResponse(nodes=[], edges=[], central_document=None)

        # Build per-document embeddings (centroid of all their chunks from text)
        doc_vectors: Dict[int, np.ndarray] = {}
        nodes = []
        doc_keywords: Dict[int, List[str]] = {}

        for doc in documents:
            text = doc.get("text", "")
            if not text.strip():
                continue

            vec = self.embedding_service.encode([text[:2000]])[0]
            doc_vectors[doc["id"]] = vec

            # Extract top keywords for node metadata
            try:
                kw_result = self.keyword_extractor.extract_keywords(text[:1000], top_k=5)
                keywords = [kw.keyword for kw in kw_result.keywords]
            except Exception:
                keywords = []

            doc_keywords[doc["id"]] = keywords
            nodes.append({
                "id": doc["id"],
                "filename": doc["filename"],
                "word_count": len(text.split()),
                "top_keywords": keywords,
            })

        # Compute pairwise cosine similarity edges
        edges: List[CitationEdge] = []
        doc_ids = list(doc_vectors.keys())
        doc_id_to_filename = {doc["id"]: doc["filename"] for doc in documents}

        for i in range(len(doc_ids)):
            for j in range(i + 1, len(doc_ids)):
                id_a, id_b = doc_ids[i], doc_ids[j]
                vec_a = doc_vectors[id_a]
                vec_b = doc_vectors[id_b]
                score = float(np.dot(vec_a, vec_b))
                score = max(0.0, min(1.0, score))

                if score >= similarity_threshold:
                    shared = list(
                        set(doc_keywords.get(id_a, [])) & set(doc_keywords.get(id_b, []))
                    )
                    edges.append(CitationEdge(
                        source_doc=doc_id_to_filename.get(id_a, str(id_a)),
                        target_doc=doc_id_to_filename.get(id_b, str(id_b)),
                        shared_concepts=shared[:5],
                        similarity_score=round(score, 4),
                    ))

        # Find the most central document (highest average similarity)
        central = None
        if edges:
            degree: Dict[str, float] = {}
            for e in edges:
                degree[e.source_doc] = degree.get(e.source_doc, 0) + e.similarity_score
                degree[e.target_doc] = degree.get(e.target_doc, 0) + e.similarity_score
            central = max(degree, key=degree.get)

        return CitationNetworkResponse(nodes=nodes, edges=edges, central_document=central)
