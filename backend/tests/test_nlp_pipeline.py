import pytest
from httpx import AsyncClient, ASGITransport
from app.main import app
from app.services.nlp.entity_extractor import NamedEntityExtractor
from app.services.nlp.keyword_extractor import KeywordExtractor
from app.services.nlp.similarity_service import TextSimilarityService
from app.services.nlp.transformer_service import TransformerPipelineService


def test_named_entity_extractor():
    text = "Dr. Alice Smith at Stanford University proposed a new Gradient Descent algorithm for PyTorch using the ImageNet dataset."
    extractor = NamedEntityExtractor()
    res = extractor.extract_entities(text)

    assert res.total_entities >= 3
    labels = [e.label for e in res.entities]
    assert "ALGORITHM" in labels or "TECH" in labels or "DATASET" in labels or "PERSON" in labels or "ORG" in labels


def test_keyword_extractor():
    text = "Retrieval Augmented Generation improves Large Language Models by fetching grounded vector context."
    extractor = KeywordExtractor()
    res = extractor.extract_keywords(text, top_k=5)
    assert len(res.keywords) > 0
    assert any("generation" in kw.keyword or "models" in kw.keyword for kw in res.keywords)


def test_similarity_service():
    service = TextSimilarityService()
    text_a = "Deep Learning neural networks process vector embeddings."
    text_b = "Neural network models evaluate vector embeddings with backpropagation."
    text_c = "Cooking recipes for baking chocolate cake."

    res_ab = service.compute_similarity(text_a, text_b)
    res_ac = service.compute_similarity(text_a, text_c)

    assert res_ab.similarity_score > res_ac.similarity_score
    assert res_ab.similarity_score > 0.2


def test_transformer_summarizer():
    text = "ResearchMind is a research intelligence platform designed to extract, analyze, and synthesize technical papers. It combines traditional Machine Learning, PyTorch Deep Learning, Natural Language Processing, Sentence Embeddings, FAISS Vector Search, and Retrieval Augmented Generation to deliver grounded citations and literature summaries for AI researchers."
    service = TransformerPipelineService()
    res = service.summarize_text(text, max_length=100, min_length=20)
    assert len(res.summary) > 0
    assert res.summary_word_count <= res.original_word_count


@pytest.mark.asyncio
async def test_nlp_api_endpoints():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        ner_resp = await ac.post("/api/v1/nlp/ner", json={"text": "PyTorch was used at Google to train Transformer models."})
        assert ner_resp.status_code == 200
        ner_data = ner_resp.json()
        assert ner_data["total_entities"] > 0

        kw_resp = await ac.post("/api/v1/nlp/keywords?top_k=5", json={"text": "Superconducting qubits and quantum error correction in physics."})
        assert kw_resp.status_code == 200
        kw_data = kw_resp.json()
        assert len(kw_data["keywords"]) > 0

        sim_resp = await ac.post("/api/v1/nlp/similarity", json={"text_a": "Quantum Mechanics", "text_b": "Quantum Physics"})
        assert sim_resp.status_code == 200
        sim_data = sim_resp.json()
        assert sim_data["similarity_score"] > 0.0

        sum_resp = await ac.post("/api/v1/nlp/summarize", json={"text": "ResearchMind uses FastAPI backends and React frontends to process documents, extract entities, compute vector embeddings, and generate grounded answers with source citations."})
        assert sum_resp.status_code == 200
        sum_data = sum_resp.json()
        assert "summary" in sum_data
