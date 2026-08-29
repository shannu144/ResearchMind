from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from app.database.session import get_db
from app.models.document import Document
from app.models.document_page import DocumentPage
from app.schemas.data_science_schemas import (
    CorpusSummaryResponse,
    NGramAnalysisResponse,
    DatasetEDADetailResponse,
)
from app.services.data_science.corpus_analyzer import CorpusAnalyzer
from app.services.data_science.eda_engine import EDAEngine

router = APIRouter(prefix="/data-science", tags=["Data Science"])
corpus_analyzer = CorpusAnalyzer()
eda_engine = EDAEngine()


@router.get("/corpus-summary", response_model=CorpusSummaryResponse)
async def get_corpus_summary(
    db: AsyncSession = Depends(get_db),
):
    """
    Get corpus-wide Data Science statistics (vocabulary richness, document length distribution, top terms, upload trends).
    """
    docs_result = await db.execute(select(Document))
    documents = docs_result.scalars().all()

    pages_result = await db.execute(select(DocumentPage))
    pages = pages_result.scalars().all()

    return corpus_analyzer.analyze_corpus(documents, pages)


@router.get("/ngrams", response_model=NGramAnalysisResponse)
async def get_ngram_analysis(
    top_k: int = Query(20, ge=5, le=100),
    db: AsyncSession = Depends(get_db),
):
    """
    Get N-gram frequency distributions (unigrams, bigrams, trigrams) across document corpus.
    """
    pages_result = await db.execute(select(DocumentPage))
    pages = pages_result.scalars().all()

    return corpus_analyzer.generate_ngram_analysis(pages, top_k=top_k)


@router.get("/csv/{document_id}/eda", response_model=DatasetEDADetailResponse)
async def get_csv_dataset_eda(
    document_id: int,
    db: AsyncSession = Depends(get_db),
):
    """
    Get automated Data Science EDA report for a CSV dataset (Pearson correlation matrix, skewness/kurtosis, categorical distributions).
    """
    doc_result = await db.execute(select(Document).where(Document.id == document_id))
    doc = doc_result.scalar_one_or_none()

    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")

    if doc.file_type != "csv":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Document ID {document_id} is of type '{doc.file_type}'. EDA is only supported for CSV datasets.",
        )

    try:
        return eda_engine.analyze_dataset(doc.file_path, doc.id, doc.filename)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error computing EDA report: {str(e)}",
        )
