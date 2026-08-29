from typing import Optional, List, Dict
from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from app.database.session import get_db
from app.models.document import Document
from app.models.document_page import DocumentPage
from app.schemas.nlp_schemas import (
    TextAnalysisRequest,
    NERAnalysisResponse,
    KeywordExtractionResponse,
    TextSimilarityRequest,
    TextSimilarityResponse,
    SummarizationRequest,
    SummarizationResponse,
)
from app.services.nlp.entity_extractor import NamedEntityExtractor
from app.services.nlp.keyword_extractor import KeywordExtractor
from app.services.nlp.similarity_service import TextSimilarityService
from app.services.nlp.transformer_service import TransformerPipelineService

router = APIRouter(prefix="/nlp", tags=["NLP Analytics & Transformers"])
ner_extractor = NamedEntityExtractor()
keyword_extractor = KeywordExtractor()
similarity_service = TextSimilarityService()
transformer_service = TransformerPipelineService()


@router.post("/ner", response_model=NERAnalysisResponse)
async def analyze_named_entities(
    request: TextAnalysisRequest,
    db: AsyncSession = Depends(get_db),
):
    """
    Extract Named Entities (Algorithms, Datasets, Technologies, People, Organizations, Locations, Concepts).
    """
    target_text = request.text

    if not target_text and request.document_id:
        pages_res = await db.execute(
            select(DocumentPage)
            .where(DocumentPage.document_id == request.document_id)
            .order_by(DocumentPage.page_number.asc())
        )
        pages = pages_res.scalars().all()
        if not pages:
            raise HTTPException(status_code=404, detail=f"Document ID {request.document_id} has no pages.")
        target_text = "\n".join([p.raw_text for p in pages])

    if not target_text or not target_text.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Either 'text' or valid 'document_id' must be provided.",
        )

    return ner_extractor.extract_entities(target_text)


@router.post("/keywords", response_model=KeywordExtractionResponse)
async def extract_keywords(
    request: TextAnalysisRequest,
    top_k: int = Query(10, ge=3, le=50),
    db: AsyncSession = Depends(get_db),
):
    """
    Extract top keyphrases and n-gram domain terms.
    """
    target_text = request.text

    if not target_text and request.document_id:
        pages_res = await db.execute(
            select(DocumentPage)
            .where(DocumentPage.document_id == request.document_id)
            .order_by(DocumentPage.page_number.asc())
        )
        pages = pages_res.scalars().all()
        if not pages:
            raise HTTPException(status_code=404, detail=f"Document ID {request.document_id} has no pages.")
        target_text = "\n".join([p.raw_text for p in pages])

    if not target_text or not target_text.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Either 'text' or valid 'document_id' must be provided.",
        )

    return keyword_extractor.extract_keywords(target_text, top_k=top_k)


@router.post("/similarity", response_model=TextSimilarityResponse)
async def compute_text_similarity(
    request: TextSimilarityRequest,
    db: AsyncSession = Depends(get_db),
):
    """
    Compute Cosine Similarity score between two document texts or text snippets.
    """
    text_a = request.text_a
    text_b = request.text_b

    if not text_a and request.document_id_a:
        pages_res = await db.execute(
            select(DocumentPage)
            .where(DocumentPage.document_id == request.document_id_a)
            .order_by(DocumentPage.page_number.asc())
        )
        pages = pages_res.scalars().all()
        text_a = "\n".join([p.raw_text for p in pages]) if pages else ""

    if not text_b and request.document_id_b:
        pages_res = await db.execute(
            select(DocumentPage)
            .where(DocumentPage.document_id == request.document_id_b)
            .order_by(DocumentPage.page_number.asc())
        )
        pages = pages_res.scalars().all()
        text_b = "\n".join([p.raw_text for p in pages]) if pages else ""

    if not text_a or not text_b:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Both text_a and text_b (or valid document IDs) must be provided.",
        )

    return similarity_service.compute_similarity(text_a, text_b)


@router.post("/summarize", response_model=SummarizationResponse)
async def summarize_document(
    request: SummarizationRequest,
    db: AsyncSession = Depends(get_db),
):
    """
    Generate abstractive summary using Hugging Face Transformer pipeline.
    """
    target_text = request.text

    if not target_text and request.document_id:
        pages_res = await db.execute(
            select(DocumentPage)
            .where(DocumentPage.document_id == request.document_id)
            .order_by(DocumentPage.page_number.asc())
        )
        pages = pages_res.scalars().all()
        if not pages:
            raise HTTPException(status_code=404, detail=f"Document ID {request.document_id} has no pages.")
        target_text = "\n".join([p.raw_text for p in pages])

    if not target_text or not target_text.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Either 'text' or valid 'document_id' must be provided.",
        )

    return transformer_service.summarize_text(
        target_text, max_length=request.max_length, min_length=request.min_length
    )
