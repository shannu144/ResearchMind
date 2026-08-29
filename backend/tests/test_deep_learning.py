import os
import torch
import pytest
from httpx import AsyncClient, ASGITransport
from app.main import app
from app.services.deep_learning.vocab import Vocabulary
from app.services.deep_learning.dataset import TextDataset
from app.services.deep_learning.models import BiLSTMClassifier
from app.services.deep_learning.trainer import PyTorchDeepLearningPipeline
from app.schemas.deep_learning_schemas import DLTrainRequest


def test_pytorch_vocabulary_and_dataset():
    texts = ["Deep learning in PyTorch uses tensors.", "Transformers improve NLP models."]
    vocab = Vocabulary(max_vocab_size=100)
    vocab.build_vocab(texts)

    assert len(vocab) > 2
    encoded = vocab.encode("Deep learning NLP", max_seq_len=10)
    assert len(encoded) == 10
    assert encoded[0] != vocab.unk_idx

    dataset = TextDataset(texts, [0, 1], vocab, max_seq_len=10)
    x_tensor, y_tensor = dataset[0]
    assert x_tensor.shape[0] == 10
    assert isinstance(x_tensor, torch.Tensor)


def test_bilstm_model_forward():
    model = BiLSTMClassifier(vocab_size=50, embed_dim=32, hidden_dim=32, num_classes=5)
    dummy_input = torch.randint(0, 50, (4, 16))  # batch_size=4, seq_len=16
    logits = model(dummy_input)
    assert logits.shape == (4, 5)


def test_pytorch_trainer():
    pipeline = PyTorchDeepLearningPipeline()
    config = DLTrainRequest(epochs=2, batch_size=4)
    res = pipeline.train_and_evaluate(config=config)

    assert res.epochs_trained == 2
    assert len(res.epoch_history) == 2
    assert os.path.exists(res.saved_model_path)

    pred = pipeline.predict("Neural network architectures for quantum mechanics.")
    assert pred.predicted_category in [
        "Computer Science & AI",
        "Medical & Life Sciences",
        "Financial Analytics",
        "Physics & Quantum",
        "Climate & Environment",
    ]
    assert pred.confidence > 0.0


@pytest.mark.asyncio
async def test_deep_learning_api_endpoints():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        compare_resp = await ac.get("/api/v1/deep-learning/compare")
        assert compare_resp.status_code == 200
        comp_data = compare_resp.json()
        assert len(comp_data["benchmark_summary"]) >= 2
        assert "why_deep_learning_helps" in comp_data["architectural_insights"]

        train_payload = {"epochs": 2, "batch_size": 4}
        train_resp = await ac.post("/api/v1/deep-learning/train", json=train_payload)
        assert train_resp.status_code == 200
        train_data = train_resp.json()
        assert train_data["epochs_trained"] == 2

        pred_payload = {"text": "CRISPR gene editing in biological cell structures."}
        pred_resp = await ac.post("/api/v1/deep-learning/predict", json=pred_payload)
        assert pred_resp.status_code == 200
        pred_data = pred_resp.json()
        assert "predicted_category" in pred_data
