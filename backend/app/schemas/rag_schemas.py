from typing import List, Dict, Any, Optional
from pydantic import BaseModel


# ─── LLMOps RAG Triad Evaluation (defined here to avoid circular imports) ─────

class RAGTriadScore(BaseModel):
    context_relevance: float      # [0.0, 1.0] How relevant retrieved context is to query
    faithfulness: float           # [0.0, 1.0] Groundedness of answer against source chunks
    answer_relevance: float       # [0.0, 1.0] How directly the answer addresses the query
    overall_triad_score: float    # [0.0, 1.0] Weighted composite triad score
    evaluation_breakdown: Dict[str, Any]


# ─── Source Citations ─────────────────────────────────────────────────────────

class SourceCitation(BaseModel):
    filename: str
    page_number: int
    chunk_id: str
    similarity_score: float
    text_snippet: str


# ─── RAG Query Request / Response ─────────────────────────────────────────────

class RAGQueryRequest(BaseModel):
    question: str
    top_k: int = 4
    document_ids: Optional[List[int]] = None
    similarity_threshold: float = 0.20
    use_hybrid_search: bool = False
    dense_weight: float = 0.5
    sparse_weight: float = 0.5
    evaluate_triad: bool = False


class RAGQueryResponse(BaseModel):
    question: str
    answer: str
    sources: List[SourceCitation]
    has_sufficient_context: bool
    llm_provider_used: str
    search_mode: str = "dense_faiss"   # "dense_faiss" or "hybrid_rrf"
    triad_score: Optional[RAGTriadScore] = None


# ─── Document Summarization ───────────────────────────────────────────────────

class SummarizeRAGRequest(BaseModel):
    document_id: int
    focus_topic: Optional[str] = None


class SummarizeRAGResponse(BaseModel):
    document_id: int
    filename: str
    summary: str
    sources: List[SourceCitation]
