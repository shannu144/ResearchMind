import pytest
import numpy as np
from httpx import AsyncClient, ASGITransport
from app.main import app
from app.services.embeddings.chunker import TextChunker
from app.services.embeddings.embedding_service import EmbeddingService
from app.services.embeddings.vector_store import FAISSVectorStore, VectorStoreFactory
from app.schemas.embedding_schemas import ChunkMetadata


def test_text_chunker():
    chunker = TextChunker()
    pages_data = [
        {"page_number": 1, "raw_text": "Deep Learning uses neural networks. Transformers implement self-attention mechanisms. Vector search retrieves top-k chunks."},
        {"page_number": 2, "raw_text": "Retrieval Augmented Generation provides grounded citations for LLM prompts."},
    ]
    chunks = chunker.chunk_document(document_id=1, filename="paper.pdf", pages_data=pages_data, chunk_size=100, chunk_overlap=20)

    assert len(chunks) >= 2
    assert chunks[0].metadata.page_number == 1
    assert chunks[-1].metadata.page_number == 2
    assert "doc_1_p1" in chunks[0].chunk_id


def test_embedding_service():
    service = EmbeddingService()
    texts = ["Artificial intelligence and machine learning.", "Quantum mechanics and superconducting qubits."]
    vectors = service.encode(texts)

    assert vectors.shape == (2, service.dimension)
    # Check L2 normalization (length of vector approx 1.0)
    norm0 = float(np.linalg.norm(vectors[0]))
    assert abs(norm0 - 1.0) < 1e-3


def test_faiss_vector_store(tmp_path):
    index_path = str(tmp_path / "test_faiss.bin")
    store = FAISSVectorStore(index_path=index_path, dimension=4)
    meta = ChunkMetadata(chunk_id="c1", document_id=10, filename="test.txt", page_number=1, text="Sample chunk text", word_count=3)
    vec = np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32)

    store.add_vectors(vec, [meta])
    query = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    results = store.search(query, top_k=1)

    assert len(results) == 1
    assert results[0].chunk_id == "c1"
    assert results[0].score > 0.9


@pytest.mark.asyncio
async def test_embeddings_api_endpoints():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        # 1. Upload sample document
        content = b"Page 1: Attention Is All You Need paper. Transformers use self-attention mechanisms.\nPage 2: RAG pipelines retrieve top-k vector chunks."
        files = {"file": ("attention.txt", content, "text/plain")}
        upload_resp = await ac.post("/api/v1/documents/upload", files=files)
        assert upload_resp.status_code == 201
        doc_id = upload_resp.json()["id"]

        # 2. Create embeddings and index into FAISS
        create_resp = await ac.post("/api/v1/embeddings/create", json={"document_id": doc_id, "chunk_size": 200, "chunk_overlap": 30})
        assert create_resp.status_code == 200
        create_data = create_resp.json()
        assert create_data["total_chunks_created"] >= 1

        # 3. Vector search query
        search_resp = await ac.post("/api/v1/embeddings/search", json={"query": "What are self-attention mechanisms?", "top_k": 3})
        assert search_resp.status_code == 200
        search_data = search_resp.json()
        assert len(search_data["results"]) >= 1
        assert search_data["results"][0]["page_number"] >= 1

        # 4. Get vector store stats
        stats_resp = await ac.get("/api/v1/embeddings/stats")
        assert stats_resp.status_code == 200
        stats_data = stats_resp.json()
        assert stats_data["total_vectors"] >= 1
