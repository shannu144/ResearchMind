from fastapi import APIRouter, HTTPException, status
from app.schemas.deep_learning_schemas import (
    DLTrainRequest,
    DLTrainResponse,
    DLPredictRequest,
    DLPredictResponse,
    MLVsDLComparisonResponse,
)
from app.services.deep_learning.trainer import PyTorchDeepLearningPipeline

router = APIRouter(prefix="/deep-learning", tags=["Deep Learning"])
dl_pipeline = PyTorchDeepLearningPipeline()


@router.post("/train", response_model=DLTrainResponse)
async def train_deep_learning_model(
    request: DLTrainRequest = DLTrainRequest(),
):
    """
    Train PyTorch Bidirectional LSTM (BiLSTM) neural network on research corpus.
    """
    try:
        return dl_pipeline.train_and_evaluate(config=request)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error training PyTorch Deep Learning model: {str(e)}",
        )


@router.post("/predict", response_model=DLPredictResponse)
async def predict_with_deep_learning(
    request: DLPredictRequest,
):
    """
    Classify research text using trained PyTorch BiLSTM neural network.
    """
    if not request.text or not request.text.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Field 'text' must not be empty.",
        )
    try:
        return dl_pipeline.predict(text=request.text)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"PyTorch inference error: {str(e)}",
        )


@router.get("/compare", response_model=MLVsDLComparisonResponse)
async def compare_ml_vs_deep_learning():
    """
    Get empirical comparison benchmark between Traditional ML (TF-IDF + SVM/Logistic Regression) vs Deep Learning (PyTorch BiLSTM).
    """
    return dl_pipeline.get_comparison()
