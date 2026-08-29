from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from typing import List, Optional
from app.database.session import get_db
from app.models.document import Document
from app.models.document_page import DocumentPage
from app.schemas.clustering_schemas import (
    LDATopicModelingRequest,
    LDATopicModelingResponse,
    KMeansClusteringRequest,
    KMeansClusteringResponse,
)
from app.services.data_science.clustering_service import DocumentClusteringService

router = APIRouter(prefix="/clustering", tags=["Document Clustering & Topic Modeling"])
clustering_service = DocumentClusteringService()


async def _load_documents(
    db: AsyncSession,
    document_ids: Optional[List[int]] = None,
) -> List[dict]:
    """Helper: load document texts from DB, optionally filtered by IDs."""
    query = select(Document)
    if document_ids:
        query = query.where(Document.id.in_(document_ids))
    docs_res = await db.execute(query)
    docs = docs_res.scalars().all()

    documents_data = []
    for doc in docs:
        pages_res = await db.execute(
            select(DocumentPage)
            .where(DocumentPage.document_id == doc.id)
            .order_by(DocumentPage.page_number.asc())
        )
        pages = pages_res.scalars().all()
        text = " ".join(p.raw_text for p in pages)
        if text.strip():
            documents_data.append({"id": doc.id, "filename": doc.filename, "text": text})

    return documents_data


@router.post("/lda", response_model=LDATopicModelingResponse)
async def run_lda_topic_modeling(
    request: LDATopicModelingRequest,
    db: AsyncSession = Depends(get_db),
):
    """
    📊 LDA Topic Modeling.
    Applies Latent Dirichlet Allocation to discover latent themes across
    uploaded research documents. Returns topic-word distributions and
    per-document topic assignments.
    """
    documents = await _load_documents(db, request.document_ids)
    if not documents:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No documents with text content found. Upload documents first.",
        )

    return clustering_service.run_lda(
        documents=documents,
        n_topics=request.n_topics,
        max_features=request.max_features,
        n_top_words=request.n_top_words,
    )


@router.post("/kmeans", response_model=KMeansClusteringResponse)
async def run_kmeans_clustering(
    request: KMeansClusteringRequest,
    db: AsyncSession = Depends(get_db),
):
    """
    🔵 KMeans Document Clustering.
    Partitions documents into k clusters using KMeans on L2-normalized
    TF-IDF vectors. Returns cluster assignments with top terms per cluster
    and inertia score.
    """
    documents = await _load_documents(db, request.document_ids)
    if not documents:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No documents with text content found. Upload documents first.",
        )

    return clustering_service.run_kmeans(
        documents=documents,
        n_clusters=request.n_clusters,
    )
