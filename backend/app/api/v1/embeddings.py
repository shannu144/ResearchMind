from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from app.database.session import get_db
from app.models.document import Document
from app.models.document_page import DocumentPage
from app.schemas.embedding_schemas import (
    EmbeddingGenerateRequest,
    EmbeddingGenerateResponse,
    VectorSearchRequest,
    VectorSearchResponse,
    VectorStoreStatsResponse,
)
from app.services.embeddings.chunker import TextChunker
from app.services.embeddings.embedding_service import EmbeddingService
from app.services.embeddings.vector_store import VectorStoreFactory

router = APIRouter(prefix="/embeddings", tags=["Embeddings & Vector Database"])
chunker = TextChunker()
embedding_service = EmbeddingService()
vector_store = VectorStoreFactory.get_vector_store()


@router.post("/create", response_model=EmbeddingGenerateResponse)
async def generate_document_embeddings(
    request: EmbeddingGenerateRequest,
    db: AsyncSession = Depends(get_db),
):
    """
    Chunk document pages, generate Sentence-Transformers vector embeddings, and index into FAISS vector database.
    """
    doc_res = await db.execute(select(Document).where(Document.id == request.document_id))
    doc = doc_res.scalar_one_or_none()
    if not doc:
        raise HTTPException(status_code=404, detail=f"Document ID {request.document_id} not found.")

    pages_res = await db.execute(
        select(DocumentPage)
        .where(DocumentPage.document_id == request.document_id)
        .order_by(DocumentPage.page_number.asc())
    )
    pages = pages_res.scalars().all()
    if not pages:
        raise HTTPException(status_code=400, detail=f"Document ID {request.document_id} has no pages.")

    pages_data = [{"page_number": p.page_number, "raw_text": p.raw_text} for p in pages]

    # 1. Chunk document pages
    chunks = chunker.chunk_document(
        document_id=doc.id,
        filename=doc.filename,
        pages_data=pages_data,
        chunk_size=request.chunk_size,
        chunk_overlap=request.chunk_overlap,
    )

    if not chunks:
        raise HTTPException(status_code=400, detail="No chunks generated from document text.")

    # 2. Generate Sentence-Transformer embeddings
    texts = [c.metadata.text for c in chunks]
    vectors = embedding_service.encode(texts)

    # 3. Add to FAISS Vector Database
    metadata_list = [c.metadata for c in chunks]
    vector_store.add_vectors(vectors, metadata_list)

    doc.status = "indexed"
    await db.commit()

    return EmbeddingGenerateResponse(
        document_id=doc.id,
        filename=doc.filename,
        total_chunks_created=len(chunks),
        embedding_dimension=embedding_service.dimension,
        vector_store="FAISS IndexFlatIP",
    )


@router.post("/search", response_model=VectorSearchResponse)
async def search_vector_embeddings(
    request: VectorSearchRequest,
):
    """
    Run Semantic Similarity Search against FAISS vector database using query embedding.
    """
    if not request.query or not request.query.strip():
        raise HTTPException(status_code=400, detail="Query string must not be empty.")

    # 1. Encode query into vector embedding
    query_vector = embedding_service.encode([request.query])

    # 2. Vector search in FAISS index
    results = vector_store.search(
        query_vector=query_vector,
        top_k=request.top_k,
        document_ids=request.document_ids,
    )

    stats = vector_store.get_stats()
    return VectorSearchResponse(
        query=request.query,
        top_k=request.top_k,
        results=results,
        total_chunks_searched=stats["total_vectors"],
    )


@router.get("/stats", response_model=VectorStoreStatsResponse)
async def get_vector_store_stats():
    """
    Get FAISS vector database index statistics (total vectors indexed, dimension, index file path).
    """
    stats = vector_store.get_stats()
    return VectorStoreStatsResponse(
        vector_store_type=stats["vector_store_type"],
        total_vectors=stats["total_vectors"],
        dimension=stats["dimension"],
        index_path=stats["index_path"],
    )
