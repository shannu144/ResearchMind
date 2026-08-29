from typing import List, Dict, Any, Optional
from pydantic import BaseModel, ConfigDict


class TermFrequencyItem(BaseModel):
    term: str
    frequency: int


class NGramAnalysisResponse(BaseModel):
    unigrams: List[TermFrequencyItem]
    bigrams: List[TermFrequencyItem]
    trigrams: List[TermFrequencyItem]


class DocumentLengthDistribution(BaseModel):
    min_words: int
    quantile_25: float
    median_words: float
    quantile_75: float
    max_words: int
    mean_words: float
    std_words: float


class CorpusSummaryResponse(BaseModel):
    total_documents: int
    total_pages: int
    total_words: int
    vocabulary_size: int
    type_token_ratio: float  # TTR = unique_words / total_words
    doc_types_breakdown: Dict[str, int]
    length_distribution: DocumentLengthDistribution
    top_terms: List[TermFrequencyItem]
    upload_trends: Dict[str, int]  # YYYY-MM-DD -> count


class CorrelationMatrixResponse(BaseModel):
    columns: List[str]
    matrix: List[List[float]]  # 2D correlation matrix values [-1.0, 1.0]


class CategoricalDistributionItem(BaseModel):
    category: str
    count: int
    percentage: float


class NumericalDistributionStats(BaseModel):
    column: str
    mean: float
    std: float
    min: float
    quantile_25: float
    median: float
    quantile_75: float
    max: float
    skewness: float
    kurtosis: float


class DatasetEDADetailResponse(BaseModel):
    document_id: int
    filename: str
    row_count: int
    column_count: int
    numerical_columns: List[str]
    categorical_columns: List[str]
    missing_values: Dict[str, int]
    missing_percentage: Dict[str, float]
    duplicate_rows: int
    correlation_matrix: Optional[CorrelationMatrixResponse] = None
    numerical_distributions: List[NumericalDistributionStats] = []
    categorical_distributions: Dict[str, List[CategoricalDistributionItem]] = {}
