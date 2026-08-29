from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from app.database.session import get_db
from app.models.document import Document
from app.models.document_page import DocumentPage
from app.schemas.ml_schemas import (
    MLTrainRequest,
    MLTrainResponse,
    PredictRequest,
    PredictResponse,
    ModelComparisonResult,
)
from app.services.ml.model_trainer import DocumentClassifierPipeline

router = APIRouter(prefix="/ml", tags=["Machine Learning"])
pipeline = DocumentClassifierPipeline()


@router.post("/train", response_model=MLTrainResponse)
async def train_ml_models(
    request: MLTrainRequest = MLTrainRequest(),
):
    """
    Train and benchmark Traditional Machine Learning classifiers (Logistic Regression, Random Forest, SVM).
    """
    try:
        return pipeline.train_and_evaluate(
            test_size=request.test_size, random_state=request.random_state
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error training ML models: {str(e)}",
        )


@router.get("/evaluation", response_model=ModelComparisonResult)
async def get_model_evaluation():
    """
    Get latest evaluation benchmark metrics (Accuracy, Precision, Recall, F1-Score, Confusion Matrix).
    """
    try:
        res = pipeline.train_and_evaluate()
        return res.comparison
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error evaluating ML models: {str(e)}",
        )


@router.post("/predict", response_model=PredictResponse)
async def predict_document_topic(
    request: PredictRequest,
    db: AsyncSession = Depends(get_db),
):
    """
    Classify research topic/category for raw input text or an uploaded document ID.
    """
    text_to_classify = request.text

    if not text_to_classify and request.document_id:
        pages_result = await db.execute(
            select(DocumentPage)
            .where(DocumentPage.document_id == request.document_id)
            .order_by(DocumentPage.page_number.asc())
        )
        pages = pages_result.scalars().all()
        if not pages:
            raise HTTPException(
                status_code=404,
                detail=f"Document ID {request.document_id} has no extracted text pages.",
            )
        text_to_classify = "\n".join([p.raw_text for p in pages])

    if not text_to_classify or not text_to_classify.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Either 'text' string or valid 'document_id' must be provided.",
        )

    try:
        return pipeline.predict(text=text_to_classify, model_name=request.model_name)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Inference error: {str(e)}",
        )
