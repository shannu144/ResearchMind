from typing import List, Dict, Any, Optional
from pydantic import BaseModel, ConfigDict


class ChunkMetadata(BaseModel):
    chunk_id: str
    document_id: int
    filename: str
    page_number: int
    text: str
    word_count: int


class DocumentChunk(BaseModel):
    chunk_id: str
    metadata: ChunkMetadata
    embedding: Optional[List[float]] = None


class EmbeddingGenerateRequest(BaseModel):
    document_id: int
    chunk_size: int = 500
    chunk_overlap: int = 100


class EmbeddingGenerateResponse(BaseModel):
    document_id: int
    filename: str
    total_chunks_created: int
    embedding_dimension: int
    vector_store: str


class VectorSearchRequest(BaseModel):
    query: str
    top_k: int = 5
    document_ids: Optional[List[int]] = None


class VectorSearchResult(BaseModel):
    score: float  # Cosine similarity score [0.0, 1.0]
    chunk_id: str
    document_id: int
    filename: str
    page_number: int
    text: str


class VectorSearchResponse(BaseModel):
    query: str
    top_k: int
    results: List[VectorSearchResult]
    total_chunks_searched: int


class VectorStoreStatsResponse(BaseModel):
    vector_store_type: str
    total_vectors: int
    dimension: int
    index_path: str
