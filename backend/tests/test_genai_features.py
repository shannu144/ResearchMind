import pytest
from httpx import AsyncClient, ASGITransport
from app.main import app
from app.services.genai.research_gap_finder import ResearchGapFinder
from app.services.genai.document_comparator import DocumentComparator
from app.services.genai.literature_review import LiteratureReviewGenerator
from app.services.genai.citation_network import CitationNetworkBuilder


def test_research_gap_finder_no_content():
    """Gap finder must handle empty vector store gracefully."""
    finder = ResearchGapFinder()
    result = finder.find_gaps(focus_area="quantum computing", top_k_chunks=2)
    assert result.focus_area is not None
    assert len(result.identified_gaps) >= 1
    assert result.llm_provider_used is not None


def test_literature_review_generator_no_content():
    """Literature review generator must handle empty vector store gracefully."""
    generator = LiteratureReviewGenerator()
    result = generator.generate(research_topic="Deep Reinforcement Learning", max_sections=3)
    assert result.research_topic == "Deep Reinforcement Learning"
    assert result.llm_provider_used is not None


def test_citation_network_empty():
    """Citation network must return empty graph when no documents given."""
    builder = CitationNetworkBuilder()
    result = builder.build_network(documents=[], similarity_threshold=0.1)
    assert result.nodes == []
    assert result.edges == []
    assert result.central_document is None


def test_citation_network_with_docs():
    """Citation network must build nodes and compute edges for similar documents."""
    builder = CitationNetworkBuilder()
    docs = [
        {"id": 1, "filename": "paper_a.txt", "text": "Deep learning neural networks use gradient descent and backpropagation for training."},
        {"id": 2, "filename": "paper_b.txt", "text": "Neural networks trained with gradient descent achieve high accuracy on image classification."},
        {"id": 3, "filename": "paper_c.txt", "text": "Quantum computing leverages superposition and entanglement for computation."},
    ]
    result = builder.build_network(documents=docs, similarity_threshold=0.0)
    assert len(result.nodes) == 3
    assert len(result.edges) >= 1
    # Papers A and B should be more similar than A and C
    ab_edge = next((e for e in result.edges if "paper_a" in e.source_doc and "paper_b" in e.target_doc), None)
    ac_edge = next((e for e in result.edges if "paper_a" in e.source_doc and "paper_c" in e.target_doc), None)
    if ab_edge and ac_edge:
        assert ab_edge.similarity_score > ac_edge.similarity_score


@pytest.mark.asyncio
async def test_genai_api_endpoints():
    """GenAI API endpoints must handle requests correctly end-to-end."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        # 1. Upload and index two documents
        docs_to_upload = [
            (b"Attention mechanisms were introduced by Vaswani et al. Transformers replaced RNNs for NLP tasks. Future work includes efficient attention for long sequences.", "transformer.txt"),
            (b"BERT uses bidirectional transformers pretrained on masked language modeling. Limitations include high compute cost. Future work includes smaller distilled models.", "bert.txt"),
        ]
        doc_ids = []
        for content, name in docs_to_upload:
            resp = await ac.post("/api/v1/documents/upload", files={"file": (name, content, "text/plain")})
            assert resp.status_code == 201
            doc_id = resp.json()["id"]
            doc_ids.append(doc_id)
            embed_resp = await ac.post("/api/v1/embeddings/create", json={"document_id": doc_id, "chunk_size": 200, "chunk_overlap": 30})
            assert embed_resp.status_code == 200

        # 2. Research Gap Finder
        gap_resp = await ac.post("/api/v1/genai/research-gaps", json={
            "document_ids": doc_ids,
            "focus_area": "transformer architectures and NLP",
            "top_k_chunks": 4,
        })
        assert gap_resp.status_code == 200
        gap_data = gap_resp.json()
        assert "identified_gaps" in gap_data
        assert len(gap_data["identified_gaps"]) >= 1

        # 3. Document Comparator
        compare_resp = await ac.post("/api/v1/genai/compare-documents", json={
            "document_ids": doc_ids,
            "comparison_aspects": ["Methodology", "Contributions"],
        })
        assert compare_resp.status_code == 200
        compare_data = compare_resp.json()
        assert len(compare_data["document_summaries"]) == 2
        assert "overall_synthesis" in compare_data

        # 4. Literature Review Generator
        lit_resp = await ac.post("/api/v1/genai/literature-review", json={
            "research_topic": "Transformer models for NLP",
            "document_ids": doc_ids,
            "max_sections": 3,
        })
        assert lit_resp.status_code == 200
        lit_data = lit_resp.json()
        assert "abstract" in lit_data
        assert "sections" in lit_data

        # 5. Citation Network
        net_resp = await ac.get("/api/v1/genai/citation-network?similarity_threshold=0.0")
        assert net_resp.status_code == 200
        net_data = net_resp.json()
        assert "nodes" in net_data
        assert len(net_data["nodes"]) >= 2
