from typing import List, Dict, Any, Optional
from pydantic import BaseModel, ConfigDict


class DLTrainRequest(BaseModel):
    epochs: int = 10
    batch_size: int = 8
    learning_rate: float = 0.001
    embedding_dim: int = 128
    hidden_dim: int = 128
    max_seq_len: int = 64


class DLEpochProgress(BaseModel):
    epoch: int
    train_loss: float
    train_accuracy: float
    val_loss: float
    val_accuracy: float


class DLTrainResponse(BaseModel):
    message: str
    epochs_trained: int
    final_val_accuracy: float
    final_val_f1: float
    epoch_history: List[DLEpochProgress]
    saved_model_path: str


class DLPredictRequest(BaseModel):
    text: str


class DLPredictResponse(BaseModel):
    predicted_category: str
    confidence: float
    probabilities: Dict[str, float]
    model_architecture: str = "PyTorch Bidirectional LSTM (BiLSTM)"


class ModelComparisonSummaryItem(BaseModel):
    model_type: str  # "Traditional ML" vs "Deep Learning"
    algorithm: str   # "Logistic Regression", "SVM", "PyTorch BiLSTM"
    accuracy: float
    f1_score: float
    strengths: str
    limitations: str


class MLVsDLComparisonResponse(BaseModel):
    benchmark_summary: List[ModelComparisonSummaryItem]
    architectural_insights: Dict[str, str]
