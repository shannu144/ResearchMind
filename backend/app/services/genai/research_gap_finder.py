from typing import List, Optional
from app.schemas.genai_schemas import ResearchGap, ResearchGapFinderResponse
from app.schemas.rag_schemas import SourceCitation
from app.services.embeddings.embedding_service import EmbeddingService
from app.services.embeddings.vector_store import VectorStoreFactory
from app.services.llm.provider import LLMProviderFactory
from app.core.logging import logger


SYSTEM_PROMPT = """You are a senior AI research analyst. Your task is to identify research gaps, 
open problems, and future directions from the provided collection of research paper excerpts.
Structure your output EXACTLY as follows (use these exact headings):
GAP 1: [Short Title]
Description: [2-3 sentences explaining the gap]
Evidence: [Cite the source using filename and page]

GAP 2: [Short Title]
Description: ...
Evidence: ...

Identify between 3 and 6 distinct gaps. Be precise and grounded strictly on the context provided."""


class ResearchGapFinder:
    """
    Identifies research gaps, open problems, and future research directions
    from a corpus of research paper chunks retrieved from the FAISS vector database.
    """

    def __init__(self):
        self.embedding_service = EmbeddingService()
        self.vector_store = VectorStoreFactory.get_vector_store()
        self.llm_provider = LLMProviderFactory.get_provider()

    def find_gaps(
        self,
        document_ids: Optional[List[int]] = None,
        focus_area: Optional[str] = None,
        top_k_chunks: int = 8,
    ) -> ResearchGapFinderResponse:

        focus = focus_area or "limitations, open problems, future work, and research gaps"
        query = f"What are the limitations, future work directions, and open research problems in {focus}?"

        query_vector = self.embedding_service.encode([query])
        chunks = self.vector_store.search(
            query_vector=query_vector,
            top_k=top_k_chunks,
            document_ids=document_ids,
        )

        # Also retrieve "conclusion" and "future work" specific chunks
        future_query = self.embedding_service.encode(["future work limitations conclusion open problems"])
        future_chunks = self.vector_store.search(
            query_vector=future_query,
            top_k=top_k_chunks // 2,
            document_ids=document_ids,
        )

        # Merge unique chunks
        seen_ids = set()
        all_chunks = []
        for c in chunks + future_chunks:
            if c.chunk_id not in seen_ids:
                all_chunks.append(c)
                seen_ids.add(c.chunk_id)

        if not all_chunks:
            return ResearchGapFinderResponse(
                focus_area=focus,
                identified_gaps=[ResearchGap(
                    gap_title="Insufficient Indexed Content",
                    description="No relevant document chunks were found. Please upload documents and generate embeddings first.",
                    evidence_sources=[],
                )],
                llm_provider_used=self.llm_provider.provider_name,
                sources_consulted=0,
            )

        context_text = "\n\n---\n\n".join(
            f"[Source: {c.filename}, Page {c.page_number}]\n{c.text}"
            for c in all_chunks
        )

        prompt = f"""RESEARCH DOCUMENT CONTEXT:
{context_text}

TASK:
Analyze the above research excerpts and identify significant research gaps, 
unresolved problems, and future research directions in: {focus}.
"""

        raw_answer = self.llm_provider.generate(prompt=prompt, system_prompt=SYSTEM_PROMPT)

        gaps = self._parse_gaps(raw_answer, all_chunks)

        return ResearchGapFinderResponse(
            focus_area=focus,
            identified_gaps=gaps,
            llm_provider_used=self.llm_provider.provider_name,
            sources_consulted=len(all_chunks),
        )

    def _parse_gaps(self, raw_text: str, chunks) -> List[ResearchGap]:
        """Parse structured GAP blocks from LLM output."""
        gaps: List[ResearchGap] = []
        source_names = list(dict.fromkeys(f"{c.filename} Page {c.page_number}" for c in chunks))

        try:
            sections = raw_text.split("\nGAP ")
            for section in sections:
                if not section.strip():
                    continue
                lines = section.strip().splitlines()
                title_line = lines[0].replace("GAP ", "").strip().lstrip("0123456789: ")
                description = ""
                evidence = []

                for line in lines[1:]:
                    if line.lower().startswith("description:"):
                        description = line.split(":", 1)[1].strip()
                    elif line.lower().startswith("evidence:"):
                        evidence = [line.split(":", 1)[1].strip()]

                if not description and len(lines) > 1:
                    description = " ".join(lines[1:3])

                gaps.append(ResearchGap(
                    gap_title=title_line[:120] if title_line else "Research Gap",
                    description=description[:500] if description else raw_text[:200],
                    evidence_sources=evidence if evidence else source_names[:2],
                ))
        except Exception as e:
            logger.warning(f"Gap parsing fallback: {e}")
            gaps.append(ResearchGap(
                gap_title="Synthesized Research Gaps",
                description=raw_text[:500],
                evidence_sources=source_names[:3],
            ))

        return gaps if gaps else [ResearchGap(
            gap_title="No Specific Gaps Identified",
            description=raw_text[:300],
            evidence_sources=source_names[:2],
        )]
