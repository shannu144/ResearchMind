import os
import pytest
from httpx import AsyncClient, ASGITransport
from app.main import app
from app.services.ml.vectorizer import TFIDFVectorizerService
from app.services.ml.model_trainer import DocumentClassifierPipeline


def test_tfidf_vectorizer():
    texts = [
        "Quantum computing uses qubits and entanglement.",
        "Deep learning models optimize loss functions with gradient descent.",
    ]
    service = TFIDFVectorizerService(max_features=100)
    matrix = service.fit_transform(texts)
    assert matrix.shape[0] == 2
    assert matrix.shape[1] > 0
    features = service.get_feature_names()
    assert "quantum" in features or "learning" in features


def test_document_classifier_pipeline():
    pipeline = DocumentClassifierPipeline()
    res = pipeline.train_and_evaluate(test_size=0.2, random_state=42)

    assert res.comparison.best_model_name in ["logistic_regression", "random_forest", "svm"]
    assert res.comparison.best_f1_score > 0.0

    for m_name in ["logistic_regression", "random_forest", "svm"]:
        metric = res.comparison.metrics[m_name]
        assert metric.accuracy >= 0.0
        assert metric.precision >= 0.0
        assert metric.recall >= 0.0
        assert metric.f1_score >= 0.0
        assert len(metric.confusion_matrix) > 0

    # Test Prediction
    pred = pipeline.predict("Deep neural networks and Transformer models for NLP tasks.", model_name="logistic_regression")
    assert pred.predicted_category == "Computer Science & AI"
    assert pred.confidence > 0.0
    assert len(pred.all_probabilities) > 0


@pytest.mark.asyncio
async def test_ml_api_endpoints():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        train_resp = await ac.post("/api/v1/ml/train")
        assert train_resp.status_code == 200
        train_data = train_resp.json()
        assert "comparison" in train_data
        assert train_data["comparison"]["best_model_name"] in ["logistic_regression", "random_forest", "svm"]

        eval_resp = await ac.get("/api/v1/ml/evaluation")
        assert eval_resp.status_code == 200
        eval_data = eval_resp.json()
        assert "metrics" in eval_data

        predict_payload = {
            "text": "Clinical trial results for novel cancer immunotherapy treatment.",
            "model_name": "logistic_regression",
        }
        pred_resp = await ac.post("/api/v1/ml/predict", json=predict_payload)
        assert pred_resp.status_code == 200
        pred_data = pred_resp.json()
        assert pred_data["predicted_category"] == "Medical & Life Sciences"
