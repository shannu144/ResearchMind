from typing import List, Optional
from app.schemas.embedding_schemas import VectorSearchResult
from app.schemas.rag_schemas import SourceCitation, RAGQueryResponse, SummarizeRAGResponse, RAGTriadScore
from app.services.embeddings.embedding_service import EmbeddingService
from app.services.embeddings.vector_store import VectorStoreFactory
from app.services.embeddings.hybrid_retriever import HybridRetriever
from app.services.rag.evaluator import RAGTriadEvaluator
from app.services.llm.provider import LLMProviderFactory


SYSTEM_PROMPT = """You are ResearchMind, an expert AI research assistant. 
Your task is to answer questions grounded STRICTLY on the provided research document context.
Rules:
- ONLY use information present in the context passages below.
- Cite sources inline using the format: [Source: filename, Page N].
- If the context does not contain sufficient information to answer, state clearly: "The available documents do not contain sufficient information to answer this question."
- Do NOT hallucinate facts, models, or citations not present in the context.
- Be precise, academic, and thorough in your answer.
"""

SUMMARIZE_SYSTEM_PROMPT = """You are ResearchMind, an expert AI research summarizer.
Summarize the following research document content into a concise, structured paragraph.
Focus on: main contributions, methods, datasets used, results, and conclusions.
"""


class RAGEngine:
    """
    RAG (Retrieval-Augmented Generation) Pipeline Engine.

    Capabilities:
    1. Dense Semantic Search (FAISS IndexFlatIP via Sentence-Transformers).
    2. Hybrid Search (FAISS Dense + BM25 Sparse with Reciprocal Rank Fusion - RRF).
    3. Anti-hallucination cosine similarity cutoff safeguard.
    4. Grounded answer generation with page-level citations.
    5. Automated LLMOps RAG Triad Evaluation (Context Relevance, Faithfulness, Answer Relevance).
    """

    def __init__(self):
        self.embedding_service = EmbeddingService()
        self.vector_store = VectorStoreFactory.get_vector_store()
        self.hybrid_retriever = HybridRetriever()
        self.triad_evaluator = RAGTriadEvaluator()
        self.llm_provider = LLMProviderFactory.get_provider()

    def query(
        self,
        question: str,
        top_k: int = 4,
        document_ids: Optional[List[int]] = None,
        similarity_threshold: float = 0.20,
        use_hybrid_search: bool = False,
        dense_weight: float = 0.5,
        sparse_weight: float = 0.5,
        evaluate_triad: bool = False,
    ) -> RAGQueryResponse:
        """
        Execute RAG pipeline: retrieve relevant chunks, build grounded context, generate LLM answer.
        """
        # Step 1: Retrieval (Dense or Hybrid RRF)
        if use_hybrid_search:
            results: List[VectorSearchResult] = self.hybrid_retriever.search(
                query=question,
                top_k=top_k,
                dense_weight=dense_weight,
                sparse_weight=sparse_weight,
                document_ids=document_ids,
            )
            search_mode = "hybrid_rrf"
        else:
            query_vector = self.embedding_service.encode([question])
            results = self.vector_store.search(
                query_vector=query_vector,
                top_k=top_k,
                document_ids=document_ids,
            )
            search_mode = "dense_faiss"

        # Step 2: Anti-hallucination safeguard — filter by relevance threshold
        filtered = [r for r in results if r.score >= similarity_threshold]
        has_sufficient_context = len(filtered) > 0

        if not has_sufficient_context:
            return RAGQueryResponse(
                question=question,
                answer="The available documents do not contain sufficient information to answer this question. Please upload relevant research papers and generate their embeddings first.",
                sources=[],
                has_sufficient_context=False,
                llm_provider_used=self.llm_provider.provider_name,
                search_mode=search_mode,
                triad_score=None,
            )

        # Step 3: Build grounded context with page attribution
        context_blocks = []
        for i, chunk in enumerate(filtered):
            source_header = f"[Source: {chunk.filename}, Page {chunk.page_number}]"
            context_blocks.append(f"{source_header}\n{chunk.text}")

        context_text = "\n\n---\n\n".join(context_blocks)

        prompt = f"""RELEVANT DOCUMENT CONTEXT:
{context_text}

USER QUESTION:
{question}

Please provide a thorough, grounded answer citing specific sources using [Source: filename, Page N] format."""

        # Step 4: LLM Generation
        answer = self.llm_provider.generate(prompt=prompt, system_prompt=SYSTEM_PROMPT)

        # Step 5: Format citations
        citations = [
            SourceCitation(
                filename=chunk.filename,
                page_number=chunk.page_number,
                chunk_id=chunk.chunk_id,
                similarity_score=chunk.score,
                text_snippet=chunk.text[:200] + "..." if len(chunk.text) > 200 else chunk.text,
            )
            for chunk in filtered
        ]

        # Step 6: RAG Triad Evaluation (Optional or on-demand)
        triad_score: Optional[RAGTriadScore] = None
        if evaluate_triad:
            raw_context_texts = [c.text for c in filtered]
            triad_score = self.triad_evaluator.evaluate(
                query=question,
                answer=answer,
                context_chunks=raw_context_texts,
            )

        return RAGQueryResponse(
            question=question,
            answer=answer,
            sources=citations,
            has_sufficient_context=True,
            llm_provider_used=self.llm_provider.provider_name,
            search_mode=search_mode,
            triad_score=triad_score,
        )

    def summarize_document(
        self,
        document_id: int,
        filename: str,
        focus_topic: Optional[str] = None,
    ) -> SummarizeRAGResponse:
        """
        RAG-powered document summarization: retrieves broad representative chunks and summarizes.
        """
        question = focus_topic if focus_topic else "What are the main contributions, methods, datasets, results, and conclusions of this paper?"

        query_vector = self.embedding_service.encode([question])
        results = self.vector_store.search(
            query_vector=query_vector,
            top_k=6,
            document_ids=[document_id],
        )

        if not results:
            return SummarizeRAGResponse(
                document_id=document_id,
                filename=filename,
                summary="Insufficient indexed content found for this document. Please run /embeddings/create first.",
                sources=[],
            )

        context_blocks = []
        for chunk in results:
            source_header = f"[Source: {chunk.filename}, Page {chunk.page_number}]"
            context_blocks.append(f"{source_header}\n{chunk.text}")

        context_text = "\n\n---\n\n".join(context_blocks)
        prompt = f"""TEXT TO SUMMARIZE:
{context_text}

Focus: {question}
"""
        summary = self.llm_provider.generate(prompt=prompt, system_prompt=SUMMARIZE_SYSTEM_PROMPT)

        citations = [
            SourceCitation(
                filename=chunk.filename,
                page_number=chunk.page_number,
                chunk_id=chunk.chunk_id,
                similarity_score=chunk.score,
                text_snippet=chunk.text[:200] + "..." if len(chunk.text) > 200 else chunk.text,
            )
            for chunk in results
        ]

        return SummarizeRAGResponse(
            document_id=document_id,
            filename=filename,
            summary=summary,
            sources=citations,
        )
