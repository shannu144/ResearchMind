import pytest
from httpx import AsyncClient, ASGITransport
from app.main import app
from app.services.llm.provider import (
    MockLLMProvider,
    OpenAILLMProvider,
    GeminiLLMProvider,
    OllamaLLMProvider,
    LLMProviderFactory,
)
from app.services.rag.rag_engine import RAGEngine
from app.schemas.rag_schemas import RAGQueryRequest


def test_mock_llm_provider():
    """Mock LLM provider must generate grounded answers from prompt context."""
    provider = MockLLMProvider()
    assert provider.provider_name == "Local Grounded Engine (Mock LLM)"

    prompt = "RELEVANT DOCUMENT CONTEXT:\nTransformers use multi-head self-attention.\n\nUSER QUESTION:\nWhat is self-attention?"
    answer = provider.generate(prompt)
    assert len(answer) > 10
    assert "Transformer" in answer or "attention" in answer or "synthesized" in answer


def test_mock_llm_summarizer_prompt():
    """Mock LLM summarizer prompt path must return summary-like text."""
    provider = MockLLMProvider()
    prompt = "TEXT TO SUMMARIZE:\nThis paper introduces the Transformer architecture for NLP tasks."
    answer = provider.generate(prompt)
    assert len(answer) > 10
    assert "Summary" in answer or "paper" in answer or "Transformer" in answer


def test_llm_provider_factory_returns_mock():
    """LLMProviderFactory must return MockLLMProvider when LLM_PROVIDER=mock."""
    from app.core.config import settings
    original = settings.LLM_PROVIDER
    settings.LLM_PROVIDER = "mock"
    provider = LLMProviderFactory.get_provider()
    settings.LLM_PROVIDER = original
    assert isinstance(provider, MockLLMProvider)


def test_rag_engine_insufficient_context():
    """RAGEngine must return insufficient context response when vector store is empty."""
    engine = RAGEngine()
    # Query with a very high threshold to force insufficient context
    result = engine.query(
        question="What is the unified field theory?",
        top_k=3,
        similarity_threshold=0.999,  # impossible threshold
    )
    assert result.has_sufficient_context is False
    assert len(result.sources) == 0
    assert "not contain sufficient" in result.answer or "insufficient" in result.answer.lower() or "Please upload" in result.answer


@pytest.mark.asyncio
async def test_rag_api_endpoints():
    """RAG API endpoints must integrate document ingestion → embedding → RAG query pipeline."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        # 1. Upload document
        content = b"Page 1: Transformers were introduced by Vaswani et al. in 2017. The architecture uses multi-head self-attention mechanisms to model global dependencies. Page 2: BERT is a bidirectional transformer model pretrained on large corpora. It achieves state-of-the-art results on multiple NLP benchmarks."
        files = {"file": ("transformers_paper.txt", content, "text/plain")}
        upload_resp = await ac.post("/api/v1/documents/upload", files=files)
        assert upload_resp.status_code == 201
        doc_id = upload_resp.json()["id"]

        # 2. Create embeddings
        embed_resp = await ac.post(
            "/api/v1/embeddings/create",
            json={"document_id": doc_id, "chunk_size": 300, "chunk_overlap": 50},
        )
        assert embed_resp.status_code == 200
        assert embed_resp.json()["total_chunks_created"] >= 1

        # 3. RAG query
        query_resp = await ac.post(
            "/api/v1/rag/query",
            json={
                "question": "What is a Transformer architecture?",
                "top_k": 3,
                "similarity_threshold": 0.01,
            },
        )
        assert query_resp.status_code == 200
        query_data = query_resp.json()
        assert "answer" in query_data
        assert len(query_data["answer"]) > 10
        assert "llm_provider_used" in query_data

        # 4. RAG document summarize
        sum_resp = await ac.post(
            "/api/v1/rag/summarize",
            json={"document_id": doc_id, "focus_topic": "Transformer architecture contributions"},
        )
        assert sum_resp.status_code == 200
        sum_data = sum_resp.json()
        assert "summary" in sum_data
        assert len(sum_data["summary"]) > 10
        assert sum_data["document_id"] == doc_id

        # 5. Empty question must return 400
        bad_resp = await ac.post("/api/v1/rag/query", json={"question": "   ", "top_k": 3})
        assert bad_resp.status_code == 400
