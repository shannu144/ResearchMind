import re
from typing import List, Dict, Any
import numpy as np
from app.services.embeddings.embedding_service import EmbeddingService
from app.schemas.rag_schemas import RAGTriadScore


class RAGTriadEvaluator:
    """
    RAG Triad & LLMOps Evaluation Engine.
    Implements reference-free quantitative evaluation of RAG generations:
    1. Context Relevance: Cosine alignment between Query and Retrieved Context.
    2. Faithfulness / Groundedness: Sentence-level entailment against source chunks.
    3. Answer Relevance: Cosine similarity between Generated Answer and Query.
    """

    def __init__(self):
        self.embedding_service = EmbeddingService()

    def _split_into_claims(self, text: str) -> List[str]:
        """Split text into distinct claim sentences."""
        sentences = re.split(r"(?<=[.!?])\s+", text.strip())
        return [s.strip() for s in sentences if len(s.strip()) > 15]

    def evaluate(
        self,
        query: str,
        answer: str,
        context_chunks: List[str],
    ) -> RAGTriadScore:
        if not context_chunks or not answer.strip():
            return RAGTriadScore(
                context_relevance=0.0,
                faithfulness=0.0,
                answer_relevance=0.0,
                overall_triad_score=0.0,
                evaluation_breakdown={"verdict": "Insufficient context or empty answer"},
            )

        # 1. Context Relevance
        query_vec = self.embedding_service.encode([query])[0]
        context_vecs = self.embedding_service.encode(context_chunks)
        context_sims = [float(np.dot(query_vec, cv)) for cv in context_vecs]
        context_relevance = float(np.mean(context_sims)) if context_sims else 0.0
        context_relevance = max(0.0, min(1.0, context_relevance))

        # 2. Answer Relevance
        answer_vec = self.embedding_service.encode([answer[:1000]])[0]
        answer_relevance = float(np.dot(query_vec, answer_vec))
        answer_relevance = max(0.0, min(1.0, answer_relevance))

        # 3. Faithfulness / Groundedness (Claim sentence verification)
        claims = self._split_into_claims(answer)
        if not claims:
            claims = [answer]

        claim_vecs = self.embedding_service.encode(claims)
        supported_claims = 0

        for claim_vec in claim_vecs:
            max_sim_to_context = max(
                (float(np.dot(claim_vec, cv)) for cv in context_vecs), default=0.0
            )
            if max_sim_to_context >= 0.35:
                supported_claims += 1

        faithfulness = supported_claims / max(1, len(claims))
        faithfulness = round(max(0.0, min(1.0, faithfulness)), 4)

        # 4. Overall Composite Triad Score (weighted mean)
        overall = round(
            (0.35 * context_relevance) + (0.35 * faithfulness) + (0.30 * answer_relevance),
            4,
        )

        return RAGTriadScore(
            context_relevance=round(context_relevance, 4),
            faithfulness=round(faithfulness, 4),
            answer_relevance=round(answer_relevance, 4),
            overall_triad_score=overall,
            evaluation_breakdown={
                "total_claims_checked": len(claims),
                "supported_claims": supported_claims,
                "context_chunks_evaluated": len(context_chunks),
                "quality_grade": "High" if overall >= 0.70 else "Moderate" if overall >= 0.40 else "Needs Review",
            },
        )
