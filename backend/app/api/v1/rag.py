from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from app.database.session import get_db
from app.models.document import Document
from app.schemas.rag_schemas import (
    RAGQueryRequest,
    RAGQueryResponse,
    SummarizeRAGRequest,
    SummarizeRAGResponse,
)
from app.services.rag.rag_engine import RAGEngine

router = APIRouter(prefix="/rag", tags=["RAG — Retrieval-Augmented Generation"])
rag_engine = RAGEngine()


@router.post("/query", response_model=RAGQueryResponse)
async def rag_query(request: RAGQueryRequest):
    """
    RAG Question Answering:
    - Embeds user question via Sentence-Transformers.
    - Retrieves top-K grounded chunks via FAISS dense search or Hybrid BM25+Dense RRF search.
    - Filters by relevance threshold (anti-hallucination safeguard).
    - Constructs grounded prompt with page citations.
    - Generates answer using configured LLM provider (OpenAI / Gemini / Ollama / Mock).
    - Optionally computes reference-free RAG Triad evaluation metrics.
    """
    if not request.question or not request.question.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Question must not be empty.",
        )

    return rag_engine.query(
        question=request.question,
        top_k=request.top_k,
        document_ids=request.document_ids,
        similarity_threshold=request.similarity_threshold,
        use_hybrid_search=request.use_hybrid_search,
        dense_weight=request.dense_weight,
        sparse_weight=request.sparse_weight,
        evaluate_triad=request.evaluate_triad,
    )


@router.post("/summarize", response_model=SummarizeRAGResponse)
async def rag_summarize_document(
    request: SummarizeRAGRequest,
    db: AsyncSession = Depends(get_db),
):
    """
    RAG-Powered Document Summarization:
    - Retrieves representative chunks from the FAISS index for a specific document.
    - Generates a structured summary including: contributions, methods, datasets, results, and conclusions.
    - Supports optional focus topic to narrow the summarization scope.
    """
    doc_res = await db.execute(select(Document).where(Document.id == request.document_id))
    doc = doc_res.scalar_one_or_none()
    if not doc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Document ID {request.document_id} not found.",
        )

    return rag_engine.summarize_document(
        document_id=doc.id,
        filename=doc.filename,
        focus_topic=request.focus_topic,
    )
