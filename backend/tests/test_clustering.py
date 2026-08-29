import pytest
from httpx import AsyncClient, ASGITransport
from app.main import app
from app.services.data_science.clustering_service import DocumentClusteringService


SAMPLE_DOCS = [
    {"id": 1, "filename": "ml_paper.txt", "text": "Machine learning neural networks gradient descent backpropagation deep learning convolutional layers dropout regularization accuracy loss function."},
    {"id": 2, "filename": "nlp_paper.txt", "text": "Natural language processing transformers attention BERT GPT tokenization embeddings text classification named entity recognition sentiment analysis."},
    {"id": 3, "filename": "cv_paper.txt", "text": "Computer vision image recognition ResNet convolutional neural network object detection YOLO segmentation pixel classification feature maps."},
    {"id": 4, "filename": "rl_paper.txt", "text": "Reinforcement learning reward policy gradient Q-learning actor-critic environment agent exploration exploitation Markov decision process."},
    {"id": 5, "filename": "nlp2_paper.txt", "text": "Language model pretraining fine-tuning BERT GPT-3 transformer attention mechanism masked language modeling question answering summarization."},
]


def test_lda_topic_modeling():
    service = DocumentClusteringService()
    result = service.run_lda(documents=SAMPLE_DOCS, n_topics=3, max_features=200, n_top_words=5)

    assert result.n_topics == 3
    assert len(result.topics) == 3
    assert result.model_perplexity > 0
    for topic in result.topics:
        assert len(topic.top_words) == 5
        assert topic.label != ""
        for tw in topic.top_words:
            assert tw.weight >= 0


def test_lda_single_document():
    """LDA with fewer docs than topics must clamp n_topics gracefully."""
    service = DocumentClusteringService()
    result = service.run_lda(documents=SAMPLE_DOCS[:1], n_topics=5)
    # With 1 doc, LDA should return n_topics=1 (clamped) or empty topics
    assert result.n_topics <= 1 or len(result.topics) == 0


def test_kmeans_clustering():
    service = DocumentClusteringService()
    result = service.run_kmeans(documents=SAMPLE_DOCS, n_clusters=3)

    assert result.n_clusters == 3
    assert len(result.clusters) == 3
    assert len(result.cluster_assignments) == len(SAMPLE_DOCS)
    assert result.inertia >= 0
    # Every document must be assigned
    assigned_ids = {node.document_id for node in result.cluster_assignments}
    expected_ids = {d["id"] for d in SAMPLE_DOCS}
    assert assigned_ids == expected_ids


def test_kmeans_cluster_labels():
    service = DocumentClusteringService()
    result = service.run_kmeans(documents=SAMPLE_DOCS, n_clusters=2)
    for cluster in result.clusters:
        assert cluster.label != ""
        assert len(cluster.top_words) > 0


@pytest.mark.asyncio
async def test_clustering_api_endpoints():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        # Upload 3 documents with distinct themes
        texts = [
            (b"Deep learning uses neural networks convolutional layers for image recognition tasks.", "deep_learning.txt"),
            (b"Natural language processing transforms text into embeddings using BERT and GPT models.", "nlp.txt"),
            (b"Reinforcement learning trains agents with reward signals in Markov decision processes.", "rl.txt"),
        ]
        doc_ids = []
        for content, name in texts:
            resp = await ac.post("/api/v1/documents/upload", files={"file": (name, content, "text/plain")})
            assert resp.status_code == 201
            doc_ids.append(resp.json()["id"])

        # LDA endpoint
        lda_resp = await ac.post("/api/v1/clustering/lda", json={
            "n_topics": 2,
            "max_features": 100,
            "n_top_words": 5,
            "document_ids": doc_ids,
        })
        assert lda_resp.status_code == 200
        lda_data = lda_resp.json()
        assert lda_data["n_topics"] == 2
        assert len(lda_data["topics"]) == 2

        # KMeans endpoint
        km_resp = await ac.post("/api/v1/clustering/kmeans", json={
            "n_clusters": 2,
            "document_ids": doc_ids,
        })
        assert km_resp.status_code == 200
        km_data = km_resp.json()
        assert km_data["n_clusters"] == 2
        assert len(km_data["cluster_assignments"]) == 3
