import re
import datetime
from typing import List, Dict, Any, Optional
from app.schemas.genai_schemas import BibTeXEntry, ExportBibliographyResponse


class BibTeXExporter:
    """
    BibTeX & Academic Citation Exporter.
    Parses document titles, filenames, and extracted metadata into
    standard BibTeX entries for Overleaf/LaTeX integration.
    """

    def _sanitize_key(self, text: str) -> str:
        clean = re.sub(r"[^a-zA-Z0-9]", "", text)
        return clean.lower()[:20] if clean else "paper"

    def export_bibtex(
        self,
        documents: List[Dict[str, Any]],
    ) -> ExportBibliographyResponse:
        current_year = datetime.datetime.now().year
        entries: List[BibTeXEntry] = []
        bibtex_blocks: List[str] = []

        for idx, doc in enumerate(documents):
            title = doc.get("title") or doc.get("filename", f"Research Document {idx+1}")
            author = doc.get("author") or "ResearchMind Ingestion Pipeline"
            doc_id = doc.get("id", idx + 1)
            sanitized_title = self._sanitize_key(title)
            cite_key = f"{sanitized_title}{current_year}_{doc_id}"

            entry_type = "article" if doc.get("file_type") == "pdf" else "misc"

            bib_str = f"""@{entry_type}{{{cite_key},
  title = {{{title}}},
  author = {{{author}}},
  year = {{{current_year}}},
  note = {{Indexed in ResearchMind Platform. File: {doc.get('filename', 'doc')}}}
}}"""
            entries.append(
                BibTeXEntry(
                    key=cite_key,
                    entry_type=entry_type,
                    title=title,
                    author=author,
                    year=current_year,
                    raw_bibtex=bib_str,
                )
            )
            bibtex_blocks.append(bib_str)

        combined_bibtex = "\n\n".join(bibtex_blocks)

        return ExportBibliographyResponse(
            total_entries=len(entries),
            bibtex_string=combined_bibtex,
            entries=entries,
        )
