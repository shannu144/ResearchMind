from typing import List, Optional, Dict, Any
from pydantic import BaseModel


# ─── Research Gap Finder ────────────────────────────────────────────────────

class ResearchGap(BaseModel):
    gap_title: str
    description: str
    evidence_sources: List[str]  # ["paper.pdf Page 3", ...]


class ResearchGapFinderRequest(BaseModel):
    document_ids: Optional[List[int]] = None
    focus_area: Optional[str] = None
    top_k_chunks: int = 8


class ResearchGapFinderResponse(BaseModel):
    focus_area: str
    identified_gaps: List[ResearchGap]
    llm_provider_used: str
    sources_consulted: int


# ─── Multi-Document Comparator ───────────────────────────────────────────────

class DocumentComparisonItem(BaseModel):
    document_id: int
    filename: str
    summary: str


class DimensionComparison(BaseModel):
    dimension: str          # e.g. "Methodology", "Results", "Datasets Used"
    comparison: str


class DocumentComparatorRequest(BaseModel):
    document_ids: List[int]
    comparison_aspects: Optional[List[str]] = None


class DocumentComparatorResponse(BaseModel):
    document_summaries: List[DocumentComparisonItem]
    dimension_comparisons: List[DimensionComparison]
    overall_synthesis: str
    llm_provider_used: str


# ─── Literature Review Generator ─────────────────────────────────────────────

class LiteratureReviewSection(BaseModel):
    section_title: str
    content: str
    citations: List[str]


class LiteratureReviewRequest(BaseModel):
    document_ids: Optional[List[int]] = None
    research_topic: str
    max_sections: int = 5


class LiteratureReviewResponse(BaseModel):
    research_topic: str
    abstract: str
    sections: List[LiteratureReviewSection]
    conclusion: str
    llm_provider_used: str
    total_sources_used: int


# ─── Citation Network ────────────────────────────────────────────────────────

class CitationEdge(BaseModel):
    source_doc: str
    target_doc: str
    shared_concepts: List[str]
    similarity_score: float


class CitationNetworkResponse(BaseModel):
    nodes: List[Dict]       # {id, filename, word_count, top_keywords}
    edges: List[CitationEdge]
    central_document: Optional[str]


# ─── Scientific Hypothesis Generator ────────────────────────────────────────

class ExperimentPlan(BaseModel):
    independent_variables: List[str]
    dependent_variables: List[str]
    baseline_models: List[str]
    suggested_datasets: List[str]
    evaluation_metrics: List[str]


class ScientificHypothesis(BaseModel):
    hypothesis_id: int
    title: str
    rationale: str
    formal_hypothesis: str
    expected_outcome: str
    experiment_plan: ExperimentPlan


class HypothesisGeneratorRequest(BaseModel):
    research_domain: str
    document_ids: Optional[List[int]] = None
    top_k: int = 6


class HypothesisGeneratorResponse(BaseModel):
    research_domain: str
    hypotheses: List[ScientificHypothesis]
    llm_provider_used: str
    sources_analyzed: int


# ─── BibTeX Bibliography Exporter ───────────────────────────────────────────

class BibTeXEntry(BaseModel):
    key: str
    entry_type: str
    title: str
    author: str
    year: int
    raw_bibtex: str


class ExportBibliographyRequest(BaseModel):
    document_ids: Optional[List[int]] = None


class ExportBibliographyResponse(BaseModel):
    total_entries: int
    bibtex_string: str
    entries: List[BibTeXEntry]
