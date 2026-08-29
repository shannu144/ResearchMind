from typing import List, Dict, Any, Optional
from pydantic import BaseModel, ConfigDict


class ModelEvaluationMetric(BaseModel):
    model_name: str
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    confusion_matrix: List[List[int]]
    labels: List[str]


class ModelComparisonResult(BaseModel):
    best_model_name: str
    best_f1_score: float
    metrics: Dict[str, ModelEvaluationMetric]


class MLTrainRequest(BaseModel):
    test_size: float = 0.2
    random_state: int = 42


class MLTrainResponse(BaseModel):
    message: str
    comparison: ModelComparisonResult
    saved_model_paths: Dict[str, str]


class PredictRequest(BaseModel):
    text: Optional[str] = None
    document_id: Optional[int] = None
    model_name: str = "logistic_regression"  # "logistic_regression", "random_forest", "svm"


class PredictionProbability(BaseModel):
    category: str
    confidence: float


class PredictResponse(BaseModel):
    predicted_category: str
    confidence: float
    all_probabilities: List[PredictionProbability]
    model_used: str
