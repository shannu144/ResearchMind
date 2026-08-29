from datetime import datetime
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, ConfigDict


class DocumentPageRead(BaseModel):
    id: int
    document_id: int
    page_number: int
    raw_text: str
    cleaned_text: Optional[str] = None
    word_count: int
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)


class DocumentRead(BaseModel):
    id: int
    filename: str
    title: Optional[str] = None
    author: Optional[str] = None
    file_type: str
    file_path: str
    file_size: int
    page_count: int
    word_count: int
    status: str
    metadata_json: Optional[Dict[str, Any]] = None
    created_at: datetime
    updated_at: datetime

    model_config = ConfigDict(from_attributes=True)


class DocumentDetailRead(DocumentRead):
    pages: List[DocumentPageRead] = []


class TextPreprocessingConfig(BaseModel):
    lowercase: bool = True
    remove_punctuation: bool = True
    remove_stopwords: bool = True
    lemmatize: bool = True
    min_word_length: int = 2


class TextPreprocessingResult(BaseModel):
    document_id: int
    page_number: int
    raw_text: str
    cleaned_text: str
    tokens: List[str]
    sentences: List[str]
    word_count: int
    unique_word_count: int


class CSVPreprocessingResult(BaseModel):
    document_id: int
    row_count: int
    column_count: int
    columns: List[str]
    numerical_columns: List[str]
    categorical_columns: List[str]
    missing_values: Dict[str, int]
    missing_percentage: Dict[str, float]
    duplicate_rows: int
    summary_statistics: Dict[str, Dict[str, float]]
    outliers_iqr: Dict[str, int]
