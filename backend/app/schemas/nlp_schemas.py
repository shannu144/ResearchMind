from typing import List, Dict, Any, Optional
from pydantic import BaseModel, ConfigDict


class TextAnalysisRequest(BaseModel):
    text: Optional[str] = None
    document_id: Optional[int] = None


class EntityItem(BaseModel):
    text: str
    label: str  # PERSON, ORG, TECH, ALGORITHM, DATASET, LOCATION, CONCEPT
    start_char: int
    end_char: int
    confidence: float = 1.0


class NERAnalysisResponse(BaseModel):
    entities: List[EntityItem]
    entity_counts_by_type: Dict[str, int]
    total_entities: int


class KeywordItem(BaseModel):
    keyword: str
    score: float


class KeywordExtractionResponse(BaseModel):
    keywords: List[KeywordItem]


class TextSimilarityRequest(BaseModel):
    text_a: Optional[str] = None
    text_b: Optional[str] = None
    document_id_a: Optional[int] = None
    document_id_b: Optional[int] = None


class TextSimilarityResponse(BaseModel):
    similarity_score: float  # [0.0, 1.0]
    metric: str = "Cosine Similarity (TF-IDF Vector Space)"
    interpretation: str


class SummarizationRequest(BaseModel):
    text: Optional[str] = None
    document_id: Optional[int] = None
    max_length: int = 150
    min_length: int = 40


class SummarizationResponse(BaseModel):
    summary: str
    original_word_count: int
    summary_word_count: int
    compression_ratio: float
    model_used: str = "HuggingFace Transformer (DistilBART / Abstractive Summarizer)"
