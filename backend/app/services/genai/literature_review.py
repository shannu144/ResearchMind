from typing import List, Optional
from app.schemas.genai_schemas import LiteratureReviewSection, LiteratureReviewResponse
from app.services.embeddings.embedding_service import EmbeddingService
from app.services.embeddings.vector_store import VectorStoreFactory
from app.services.llm.provider import LLMProviderFactory


SYSTEM_PROMPT = """You are an expert academic writer specializing in literature reviews.
Write a structured, formal literature review based STRICTLY on the provided research excerpts.
Your output MUST follow this exact structure:
ABSTRACT: [2-3 sentence overview]
SECTION: Introduction
CONTENT: [paragraph]
CITATIONS: [source1, source2]
SECTION: [Next section name]
CONTENT: ...
CITATIONS: ...
CONCLUSION: [1-2 paragraph synthesis]

Do NOT fabricate any papers or results not present in the provided context."""


class LiteratureReviewGenerator:
    """
    Auto-generates structured academic literature reviews from a document corpus
    using RAG-retrieved chunk context and LLM synthesis.
    """

    def __init__(self):
        self.embedding_service = EmbeddingService()
        self.vector_store = VectorStoreFactory.get_vector_store()
        self.llm_provider = LLMProviderFactory.get_provider()

    def generate(
        self,
        research_topic: str,
        document_ids: Optional[List[int]] = None,
        max_sections: int = 5,
    ) -> LiteratureReviewResponse:

        # Multi-angle retrieval: broad topic + methodology + results + limitations
        queries = [
            research_topic,
            f"{research_topic} methodology approach",
            f"{research_topic} results performance evaluation",
            f"{research_topic} limitations future work",
        ]

        seen_ids = set()
        all_chunks = []
        for q in queries:
            q_vec = self.embedding_service.encode([q])
            chunks = self.vector_store.search(
                query_vector=q_vec,
                top_k=5,
                document_ids=document_ids,
            )
            for c in chunks:
                if c.chunk_id not in seen_ids:
                    all_chunks.append(c)
                    seen_ids.add(c.chunk_id)

        if not all_chunks:
            return LiteratureReviewResponse(
                research_topic=research_topic,
                abstract="No indexed document content found. Please upload documents and generate embeddings first.",
                sections=[],
                conclusion="",
                llm_provider_used=self.llm_provider.provider_name,
                total_sources_used=0,
            )

        context_text = "\n\n---\n\n".join(
            f"[Source: {c.filename}, Page {c.page_number}]\n{c.text}"
            for c in all_chunks[:20]
        )

        prompt = f"""RESEARCH TOPIC: {research_topic}

SOURCE EXCERPTS FROM RESEARCH PAPERS:
{context_text}

Generate a structured literature review on the topic: "{research_topic}".
Include {max_sections} sections covering: background, related work, methodologies, 
comparative analysis, and future directions.
"""

        raw = self.llm_provider.generate(prompt=prompt, system_prompt=SYSTEM_PROMPT)
        abstract, sections, conclusion = self._parse_review(raw, all_chunks)

        return LiteratureReviewResponse(
            research_topic=research_topic,
            abstract=abstract,
            sections=sections[:max_sections],
            conclusion=conclusion,
            llm_provider_used=self.llm_provider.provider_name,
            total_sources_used=len(all_chunks),
        )

    def _parse_review(self, raw: str, chunks):
        source_list = list(dict.fromkeys(f"{c.filename} p.{c.page_number}" for c in chunks))
        abstract = ""
        sections: List[LiteratureReviewSection] = []
        conclusion = ""

        try:
            # Extract abstract
            if "ABSTRACT:" in raw:
                abstract_part = raw.split("ABSTRACT:")[1]
                abstract = abstract_part.split("\nSECTION:")[0].strip()[:600]

            # Extract sections
            if "SECTION:" in raw:
                section_blocks = raw.split("\nSECTION:")[1:]
                for block in section_blocks:
                    lines = block.strip().splitlines()
                    title = lines[0].strip()
                    content = ""
                    citations = []
                    for line in lines[1:]:
                        if line.startswith("CONTENT:"):
                            content = line.split(":", 1)[1].strip()
                        elif line.startswith("CITATIONS:"):
                            citations = [c.strip() for c in line.split(":", 1)[1].split(",")]
                        elif content and not line.startswith("SECTION:"):
                            content += " " + line.strip()
                    sections.append(LiteratureReviewSection(
                        section_title=title[:100],
                        content=content[:800] if content else block[:400],
                        citations=citations if citations else source_list[:2],
                    ))

            # Extract conclusion
            if "CONCLUSION:" in raw:
                conclusion = raw.split("CONCLUSION:")[1].strip()[:800]

        except Exception:
            pass

        if not abstract:
            abstract = raw[:400]
        if not sections:
            sections = [LiteratureReviewSection(
                section_title="Literature Review",
                content=raw[:800],
                citations=source_list[:3],
            )]
        if not conclusion:
            conclusion = raw[-300:] if len(raw) > 300 else raw

        return abstract, sections, conclusion
