from typing import List, Optional
from app.schemas.genai_schemas import (
    DocumentComparisonItem,
    DimensionComparison,
    DocumentComparatorResponse,
)
from app.services.embeddings.embedding_service import EmbeddingService
from app.services.embeddings.vector_store import VectorStoreFactory
from app.services.llm.provider import LLMProviderFactory


SYSTEM_PROMPT = """You are an expert research paper analyst. Compare the following research documents 
across the specified dimensions. For each dimension, provide a structured comparison.
Be specific, cite sources by filename and page, and highlight key similarities and differences."""

DEFAULT_ASPECTS = [
    "Core Methodology",
    "Datasets and Benchmarks Used",
    "Key Results and Performance Metrics",
    "Limitations and Weaknesses",
    "Novelty and Contributions",
]


class DocumentComparator:
    """
    Multi-Document RAG Comparator — retrieves representative chunks from
    each document in the FAISS index and uses an LLM to generate a structured
    side-by-side comparison across key research dimensions.
    """

    def __init__(self):
        self.embedding_service = EmbeddingService()
        self.vector_store = VectorStoreFactory.get_vector_store()
        self.llm_provider = LLMProviderFactory.get_provider()

    def compare(
        self,
        document_ids: List[int],
        filenames: List[str],
        comparison_aspects: Optional[List[str]] = None,
    ) -> DocumentComparatorResponse:

        aspects = comparison_aspects or DEFAULT_ASPECTS

        # Retrieve representative chunks per document
        doc_contexts = {}
        for doc_id, filename in zip(document_ids, filenames):
            query = self.embedding_service.encode(["main contributions methodology results"])
            chunks = self.vector_store.search(
                query_vector=query,
                top_k=4,
                document_ids=[doc_id],
            )
            doc_contexts[doc_id] = {"filename": filename, "chunks": chunks}

        # Build per-document context strings
        doc_context_blocks = []
        doc_summaries = []
        for doc_id, info in doc_contexts.items():
            chunks = info["chunks"]
            filename = info["filename"]
            if chunks:
                text = "\n".join(f"[Page {c.page_number}] {c.text}" for c in chunks[:3])
                doc_context_blocks.append(f"=== Document: {filename} ===\n{text}")
                doc_summaries.append(DocumentComparisonItem(
                    document_id=doc_id,
                    filename=filename,
                    summary=chunks[0].text[:300] if chunks else "No indexed content.",
                ))
            else:
                doc_context_blocks.append(f"=== Document: {filename} ===\n[No indexed content available]")
                doc_summaries.append(DocumentComparisonItem(
                    document_id=doc_id,
                    filename=filename,
                    summary="No indexed content available. Run /embeddings/create first.",
                ))

        aspects_str = "\n".join(f"- {a}" for a in aspects)
        full_context = "\n\n".join(doc_context_blocks)

        prompt = f"""DOCUMENT CONTEXTS:
{full_context}

COMPARISON DIMENSIONS TO ANALYZE:
{aspects_str}

For each dimension listed above, write a detailed comparative analysis of all documents.
Format each section as:
DIMENSION: [name]
COMPARISON: [detailed comparison text]
"""
        raw = self.llm_provider.generate(prompt=prompt, system_prompt=SYSTEM_PROMPT)
        dimension_comparisons = self._parse_dimensions(raw, aspects)

        # Overall synthesis
        synthesis_prompt = f"""Given these document comparisons, write a concise 2-paragraph synthesis 
highlighting the most important similarities, differences, and what a researcher should take away 
when choosing between these papers.

Comparisons:
{raw[:1200]}
"""
        synthesis = self.llm_provider.generate(
            prompt=synthesis_prompt,
            system_prompt="You are a research synthesis expert. Be concise and insightful.",
        )

        return DocumentComparatorResponse(
            document_summaries=doc_summaries,
            dimension_comparisons=dimension_comparisons,
            overall_synthesis=synthesis,
            llm_provider_used=self.llm_provider.provider_name,
        )

    def _parse_dimensions(self, raw: str, aspects: List[str]) -> List[DimensionComparison]:
        comparisons = []
        try:
            sections = raw.split("\nDIMENSION:")
            for section in sections:
                if not section.strip():
                    continue
                lines = section.strip().splitlines()
                dimension = lines[0].strip()
                comparison_text = ""
                for line in lines[1:]:
                    if line.startswith("COMPARISON:"):
                        comparison_text = line.split(":", 1)[1].strip()
                    elif comparison_text:
                        comparison_text += " " + line.strip()
                comparisons.append(DimensionComparison(
                    dimension=dimension[:100],
                    comparison=comparison_text[:800] if comparison_text else section[:400],
                ))
        except Exception:
            # Fallback: split raw text into aspect-sized chunks
            chunk_size = max(1, len(raw) // max(1, len(aspects)))
            for i, aspect in enumerate(aspects):
                start = i * chunk_size
                comparisons.append(DimensionComparison(
                    dimension=aspect,
                    comparison=raw[start:start + chunk_size].strip() or "See full analysis above.",
                ))
        return comparisons if comparisons else [DimensionComparison(
            dimension="Overall Comparison",
            comparison=raw[:600],
        )]
