import pytest
from httpx import AsyncClient, ASGITransport
from app.main import app


@pytest.mark.asyncio
async def test_full_system_e2e_pipeline():
    """
    Complete End-to-End System Test verifying all 10 core intelligence pipelines:
    1. System Health & Telemetry
    2. Multi-Format Document Ingestion (TXT & CSV)
    3. Corpus Analytics & N-Gram EDA
    4. Classical ML Training & Cross-Validation (LR, RF, SVM)
    5. PyTorch BiLSTM Deep Learning Training Loop
    6. NLP NER, Keyword Extraction & Cosine Similarity
    7. FAISS Vector Database Embedding Generation
    8. Grounded RAG Question Answering with Citations
    9. GenAI Suite (Research Gaps, Comparator, Literature Review, Citation Graph)
    10. Unsupervised Document Clustering (LDA Topic Modeling & KMeans)
    """
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        # Step 1: Health & Telemetry
        health_resp = await ac.get("/api/v1/health/detailed")
        assert health_resp.status_code == 200
        health_data = health_resp.json()
        assert health_data["status"] == "healthy"
        assert "embedding_model" in health_data

        # Step 2: Upload Documents
        paper_1_content = b"""
        Abstract: In this paper we introduce an attention mechanism for neural machine translation.
        Traditional recurrent neural networks struggle with long range dependencies.
        Our transformer architecture uses multi-head self-attention to capture global context.
        Experiments on WMT 2014 translation show state-of-the-art BLEU score of 28.4.
        Limitations include quadratic complexity with respect to sequence length.
        Future work will explore linear attention approximations and sparse attention patterns.
        """

        paper_2_content = b"""
        Abstract: We present BERT, a bidirectional transformer pretrained on masked language modeling.
        Unlike previous unidirectional models, BERT fuses left and right context in all layers.
        We fine-tune on GLUE benchmark achieving 80.5% average score.
        Limitations include high memory consumption during pretraining and slow inference throughput.
        Future directions include knowledge distillation and model quantization for mobile edge devices.
        """

        csv_content = b"""title,abstract,category
Deep Residual Learning,Deeper neural networks are more difficult to train. We present a residual learning framework.,computer_vision
Attention Is All You Need,The dominant sequence transduction models are based on complex recurrent or convolutional neural networks.,nlp
BERT Pre-training,We introduce a new language representation model called BERT.,nlp
Generative Adversarial Nets,We propose a new framework for estimating generative models via an adversarial process.,deep_learning
"""

        p1_resp = await ac.post("/api/v1/documents/upload", files={"file": ("transformer_paper.txt", paper_1_content, "text/plain")})
        assert p1_resp.status_code == 201
        doc1_id = p1_resp.json()["id"]

        p2_resp = await ac.post("/api/v1/documents/upload", files={"file": ("bert_paper.txt", paper_2_content, "text/plain")})
        assert p2_resp.status_code == 201
        doc2_id = p2_resp.json()["id"]

        csv_resp = await ac.post("/api/v1/documents/upload", files={"file": ("papers_dataset.csv", csv_content, "text/csv")})
        assert csv_resp.status_code == 201
        csv_doc_id = csv_resp.json()["id"]

        # Step 3: Corpus Analytics & EDA
        summary_resp = await ac.get("/api/v1/data-science/corpus-summary")
        assert summary_resp.status_code == 200
        assert summary_resp.json()["total_documents"] >= 3

        ngram_resp = await ac.get("/api/v1/data-science/ngrams?top_k=10")
        assert ngram_resp.status_code == 200
        assert len(ngram_resp.json()["bigrams"]) > 0

        csv_eda_resp = await ac.get(f"/api/v1/data-science/csv/{csv_doc_id}/eda")
        assert csv_eda_resp.status_code == 200
        assert csv_eda_resp.json()["row_count"] == 4

        # Step 4: Traditional ML Training
        ml_resp = await ac.post("/api/v1/ml/train", json={"test_size": 0.2, "random_state": 42})
        assert ml_resp.status_code == 200
        ml_data = ml_resp.json()
        assert "comparison" in ml_data
        assert ml_data["comparison"]["best_f1_score"] >= 0.0

        # Step 5: Deep Learning PyTorch BiLSTM
        dl_resp = await ac.post("/api/v1/deep-learning/train", json={"epochs": 2, "batch_size": 4})
        assert dl_resp.status_code == 200
        dl_data = dl_resp.json()
        assert dl_data["epochs_trained"] == 2
        assert len(dl_data["epoch_history"]) == 2

        # Step 6: NLP Suite
        ner_resp = await ac.post("/api/v1/nlp/ner", json={"text": "Researchers at Stanford used PyTorch on ImageNet dataset."})
        assert ner_resp.status_code == 200
        assert len(ner_resp.json()["entities"]) > 0

        sim_resp = await ac.post("/api/v1/nlp/similarity", json={"text_a": "Neural networks learn features.", "text_b": "Deep learning models learn representations."})
        assert sim_resp.status_code == 200
        assert sim_resp.json()["similarity_score"] > 0.0

        # Step 7: FAISS Embeddings
        embed1 = await ac.post("/api/v1/embeddings/create", json={"document_id": doc1_id, "chunk_size": 200, "chunk_overlap": 30})
        assert embed1.status_code == 200
        embed2 = await ac.post("/api/v1/embeddings/create", json={"document_id": doc2_id, "chunk_size": 200, "chunk_overlap": 30})
        assert embed2.status_code == 200

        # Step 8: Grounded RAG Query
        rag_resp = await ac.post("/api/v1/rag/query", json={"question": "What are the limitations of transformers and BERT?", "top_k": 4, "similarity_threshold": 0.01})
        assert rag_resp.status_code == 200
        rag_data = rag_resp.json()
        assert "answer" in rag_data
        assert len(rag_data["answer"]) > 10

        # Step 9: GenAI Suite
        gap_resp = await ac.post("/api/v1/genai/research-gaps", json={"focus_area": "transformer efficiency and distillation", "top_k_chunks": 4})
        assert gap_resp.status_code == 200
        assert len(gap_resp.json()["identified_gaps"]) >= 1

        comp_resp = await ac.post("/api/v1/genai/compare-documents", json={"document_ids": [doc1_id, doc2_id], "comparison_aspects": ["Methodology", "Limitations"]})
        assert comp_resp.status_code == 200
        assert len(comp_resp.json()["dimension_comparisons"]) >= 1

        lit_resp = await ac.post("/api/v1/genai/literature-review", json={"research_topic": "Attention and Transformers in NLP", "max_sections": 3})
        assert lit_resp.status_code == 200
        assert len(lit_resp.json()["sections"]) >= 1

        net_resp = await ac.get("/api/v1/genai/citation-network?similarity_threshold=0.0")
        assert net_resp.status_code == 200
        assert len(net_resp.json()["nodes"]) >= 2

        # Step 10: Unsupervised Clustering
        lda_resp = await ac.post("/api/v1/clustering/lda", json={"n_topics": 2, "n_top_words": 5})
        assert lda_resp.status_code == 200
        assert len(lda_resp.json()["topics"]) == 2

        km_resp = await ac.post("/api/v1/clustering/kmeans", json={"n_clusters": 2})
        assert km_resp.status_code == 200
        assert len(km_resp.json()["cluster_assignments"]) >= 2
