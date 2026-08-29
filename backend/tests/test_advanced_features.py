import pytest
from httpx import AsyncClient, ASGITransport
from app.main import app
from app.schemas.embedding_schemas import ChunkMetadata
from app.services.embeddings.bm25_retriever import BM25Retriever
from app.services.embeddings.hybrid_retriever import HybridRetriever
from app.services.rag.evaluator import RAGTriadEvaluator
from app.services.genai.hypothesis_generator import ScientificHypothesisGenerator
from app.services.genai.bibtex_exporter import BibTeXExporter


def test_bm25_retriever():
    """Test Okapi BM25 indexer & sparse keyword ranking on exact scientific terms."""
    chunks = [
        ChunkMetadata(chunk_id="c1", document_id=1, filename="bert.pdf", page_number=1, text="BERT uses bidirectional transformer encoders with AdamW optimizer.", word_count=9),
        ChunkMetadata(chunk_id="c2", document_id=2, filename="resnet.pdf", page_number=1, text="ResNet uses residual skip connections for deep convolutional networks.", word_count=9),
        ChunkMetadata(chunk_id="c3", document_id=3, filename="quantum.pdf", page_number=1, text="Quantum qubits exploit superposition and entanglement for computation.", word_count=8),
    ]

    retriever = BM25Retriever()
    retriever.index_chunks(chunks)

    # Query for exact acronym "AdamW"
    results = retriever.search("AdamW optimizer", top_k=2)
    assert len(results) >= 1
    assert results[0].chunk_id == "c1"
    assert results[0].score > 0.0


def test_hybrid_retriever_rrf():
    """Test Reciprocal Rank Fusion combining dense and sparse rankings."""
    chunks = [
        ChunkMetadata(chunk_id="c1", document_id=1, filename="paper1.pdf", page_number=1, text="Sparse attention mechanisms reduce quadratic compute in transformers.", word_count=9),
        ChunkMetadata(chunk_id="c2", document_id=2, filename="paper2.pdf", page_number=1, text="Convolutional feature pyramids enhance object detection in vision models.", word_count=9),
    ]

    retriever = HybridRetriever()
    retriever.index_chunks(chunks)

    results = retriever.search(
        query="sparse attention complexity",
        top_k=2,
        dense_weight=0.5,
        sparse_weight=0.5,
    )
    assert len(results) >= 1
    assert results[0].score > 0.0


def test_rag_triad_evaluator():
    """Test RAG Triad Reference-Free Quality Metrics."""
    evaluator = RAGTriadEvaluator()
    query = "What is the computational complexity of standard self-attention?"
    context = [
        "Standard multi-head self-attention exhibits O(N^2) quadratic computational complexity with respect to input sequence length.",
        "Attention mechanisms calculate softmax over dot product queries and keys.",
    ]
    answer = "Standard self-attention has a quadratic computational complexity of O(N^2) relative to sequence length."

    score = evaluator.evaluate(query=query, answer=answer, context_chunks=context)
    assert score.context_relevance > 0.1
    assert score.faithfulness > 0.5
    assert score.answer_relevance > 0.3
    assert score.overall_triad_score > 0.3
    assert "quality_grade" in score.evaluation_breakdown


def test_hypothesis_generator():
    """Test Automated Scientific Hypothesis formulation."""
    generator = ScientificHypothesisGenerator()
    res = generator.generate_hypotheses(research_domain="Efficient Vision Transformers", top_k=2)

    assert res.research_domain == "Efficient Vision Transformers"
    assert len(res.hypotheses) >= 1
    h = res.hypotheses[0]
    assert len(h.formal_hypothesis) > 10
    assert len(h.experiment_plan.independent_variables) > 0
    assert len(h.experiment_plan.evaluation_metrics) > 0


def test_bibtex_exporter():
    """Test standard BibTeX bibliography generation."""
    exporter = BibTeXExporter()
    docs = [
        {"id": 1, "filename": "attention.pdf", "title": "Attention Is All You Need", "author": "Vaswani et al.", "file_type": "pdf"},
        {"id": 2, "filename": "bert.txt", "title": "BERT Language Model", "author": "Devlin et al.", "file_type": "txt"},
    ]
    res = exporter.export_bibtex(docs)

    assert res.total_entries == 2
    assert "@article{" in res.bibtex_string or "@misc{" in res.bibtex_string
    assert "Attention Is All You Need" in res.bibtex_string
    assert "Vaswani et al." in res.bibtex_string


@pytest.mark.asyncio
async def test_advanced_api_endpoints():
    """Test all new API endpoints end-to-end."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        # 1. Test BibTeX Export
        bib_resp = await ac.post("/api/v1/genai/export-bibtex", json={})
        assert bib_resp.status_code == 200
        bib_data = bib_resp.json()
        assert "bibtex_string" in bib_data

        # 2. Test Hypothesis Generation
        hypo_resp = await ac.post("/api/v1/genai/hypotheses", json={"research_domain": "Diffusion Models for Audio", "top_k": 3})
        assert hypo_resp.status_code == 200
        hypo_data = hypo_resp.json()
        assert len(hypo_data["hypotheses"]) >= 1

        # 3. Test Hybrid RAG Query with Triad evaluation
        rag_resp = await ac.post(
            "/api/v1/rag/query",
            json={
                "question": "What are the core components of neural network architectures?",
                "top_k": 3,
                "similarity_threshold": 0.01,
                "use_hybrid_search": True,
                "dense_weight": 0.6,
                "sparse_weight": 0.4,
                "evaluate_triad": True,
            },
        )
        assert rag_resp.status_code == 200
        rag_data = rag_resp.json()
        assert rag_data["search_mode"] == "hybrid_rrf"
        assert rag_data["triad_score"] is not None
