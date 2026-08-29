from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from typing import List
from app.database.session import get_db
from app.models.document import Document
from app.models.document_page import DocumentPage
from app.schemas.genai_schemas import (
    ResearchGapFinderRequest,
    ResearchGapFinderResponse,
    DocumentComparatorRequest,
    DocumentComparatorResponse,
    LiteratureReviewRequest,
    LiteratureReviewResponse,
    CitationNetworkResponse,
    HypothesisGeneratorRequest,
    HypothesisGeneratorResponse,
    ExportBibliographyRequest,
    ExportBibliographyResponse,
)
from app.services.genai.research_gap_finder import ResearchGapFinder
from app.services.genai.document_comparator import DocumentComparator
from app.services.genai.literature_review import LiteratureReviewGenerator
from app.services.genai.citation_network import CitationNetworkBuilder
from app.services.genai.hypothesis_generator import ScientificHypothesisGenerator
from app.services.genai.bibtex_exporter import BibTeXExporter

router = APIRouter(prefix="/genai", tags=["GenAI — Research Intelligence Features"])

gap_finder = ResearchGapFinder()
comparator = DocumentComparator()
lit_reviewer = LiteratureReviewGenerator()
network_builder = CitationNetworkBuilder()
hypothesis_generator = ScientificHypothesisGenerator()
bibtex_exporter = BibTeXExporter()


@router.post("/research-gaps", response_model=ResearchGapFinderResponse)
async def find_research_gaps(request: ResearchGapFinderRequest):
    """
    🔍 Research Gap Finder — Flagship Feature.
    Uses dual-angle RAG retrieval (topic + future-work queries) and LLM analysis
    to identify open problems and future research directions in a topic area.
    """
    return gap_finder.find_gaps(
        document_ids=request.document_ids,
        focus_area=request.focus_area,
        top_k_chunks=request.top_k_chunks,
    )


@router.post("/compare-documents", response_model=DocumentComparatorResponse)
async def compare_documents(
    request: DocumentComparatorRequest,
    db: AsyncSession = Depends(get_db),
):
    """
    📄 Multi-Document Comparator.
    Retrieves representative chunks from each document and generates a structured
    side-by-side comparison across methodology, results, datasets, and contributions.
    """
    if len(request.document_ids) < 2:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="At least 2 document IDs are required for comparison.",
        )

    filenames = []
    for doc_id in request.document_ids:
        doc_res = await db.execute(select(Document).where(Document.id == doc_id))
        doc = doc_res.scalar_one_or_none()
        if not doc:
            raise HTTPException(status_code=404, detail=f"Document ID {doc_id} not found.")
        filenames.append(doc.filename)

    return comparator.compare(
        document_ids=request.document_ids,
        filenames=filenames,
        comparison_aspects=request.comparison_aspects,
    )


@router.post("/literature-review", response_model=LiteratureReviewResponse)
async def generate_literature_review(request: LiteratureReviewRequest):
    """
    📚 Literature Review Generator.
    Performs multi-angle RAG retrieval across uploaded documents and generates
    a structured academic literature review with sections, citations, and a conclusion.
    """
    if not request.research_topic.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="research_topic must not be empty.",
        )

    return lit_reviewer.generate(
        research_topic=request.research_topic,
        document_ids=request.document_ids,
        max_sections=request.max_sections,
    )


@router.get("/citation-network", response_model=CitationNetworkResponse)
async def get_citation_network(
    similarity_threshold: float = 0.15,
    db: AsyncSession = Depends(get_db),
):
    """
    🕸️ Citation Network Builder.
    Computes pairwise cosine similarity between all uploaded documents and identifies
    shared research concepts, returning a graph structure (nodes + edges) suitable for
    visualisation in the frontend.
    """
    docs_res = await db.execute(select(Document))
    docs = docs_res.scalars().all()

    if not docs:
        return CitationNetworkResponse(nodes=[], edges=[], central_document=None)

    documents_data = []
    for doc in docs:
        pages_res = await db.execute(
            select(DocumentPage)
            .where(DocumentPage.document_id == doc.id)
            .order_by(DocumentPage.page_number.asc())
        )
        pages = pages_res.scalars().all()
        combined_text = " ".join(p.raw_text for p in pages[:5]) if pages else ""
        documents_data.append({"id": doc.id, "filename": doc.filename, "text": combined_text})

    return network_builder.build_network(
        documents=documents_data,
        similarity_threshold=similarity_threshold,
    )


@router.post("/hypotheses", response_model=HypothesisGeneratorResponse)
async def generate_scientific_hypotheses(request: HypothesisGeneratorRequest):
    """
    🔬 Automated Scientific Hypothesis & Experiment Designer.
    Synthesizes research corpus gaps into testable scientific hypotheses,
    independent/dependent variables, baselines, and evaluation metrics.
    """
    if not request.research_domain.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="research_domain must not be empty.",
        )

    return hypothesis_generator.generate_hypotheses(
        research_domain=request.research_domain,
        document_ids=request.document_ids,
        top_k=request.top_k,
    )


@router.post("/export-bibtex", response_model=ExportBibliographyResponse)
async def export_corpus_bibtex(
    request: ExportBibliographyRequest = ExportBibliographyRequest(),
    db: AsyncSession = Depends(get_db),
):
    """
    📜 BibTeX Bibliography & LaTeX Citation Exporter.
    Formats document metadata into standard BibTeX entries for academic papers and Overleaf.
    """
    query = select(Document)
    if request.document_ids:
        query = query.where(Document.id.in_(request.document_ids))
    docs_res = await db.execute(query)
    docs = docs_res.scalars().all()

    docs_data = [
        {
            "id": d.id,
            "filename": d.filename,
            "title": d.title,
            "author": d.author,
            "file_type": d.file_type,
        }
        for d in docs
    ]

    return bibtex_exporter.export_bibtex(docs_data)
