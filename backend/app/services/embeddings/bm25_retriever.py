import math
import re
from typing import List, Dict, Tuple, Optional
from app.schemas.embedding_schemas import ChunkMetadata, VectorSearchResult


class BM25Retriever:
    """
    Okapi BM25 Lexical Ranking Retriever.
    Provides sparse keyword matching for exact scientific terms, acronyms,
    and mathematical notation to complement dense semantic embeddings.
    """

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.corpus_chunks: List[ChunkMetadata] = []
        self.doc_lengths: List[int] = []
        self.avg_doc_len: float = 0.0
        self.term_freqs: List[Dict[str, int]] = []
        self.doc_freqs: Dict[str, int] = {}
        self.idf: Dict[str, float] = {}
        self.total_docs: int = 0

    def _tokenize(self, text: str) -> List[str]:
        """Tokenize text into lowercased terms."""
        return re.findall(r"\b[a-zA-Z0-9_\-\.]{2,}\b", text.lower())

    def index_chunks(self, chunks: List[ChunkMetadata]):
        """Index a list of document chunks into the BM25 inverted index."""
        self.corpus_chunks = chunks
        self.total_docs = len(chunks)
        if self.total_docs == 0:
            return

        self.doc_lengths = []
        self.term_freqs = []
        self.doc_freqs = {}

        total_length = 0
        for chunk in chunks:
            tokens = self._tokenize(chunk.text)
            length = len(tokens)
            self.doc_lengths.append(length)
            total_length += length

            tf: Dict[str, int] = {}
            for token in tokens:
                tf[token] = tf.get(token, 0) + 1
            self.term_freqs.append(tf)

            for token in set(tokens):
                self.doc_freqs[token] = self.doc_freqs.get(token, 0) + 1

        self.avg_doc_len = total_length / self.total_docs if self.total_docs > 0 else 0.0

        # Calculate IDF for all terms in vocabulary
        self.idf = {}
        for term, df in self.doc_freqs.items():
            # Robertson-Spärck Jones IDF formula with smoothing
            self.idf[term] = math.log(
                (self.total_docs - df + 0.5) / (df + 0.5) + 1.0
            )

    def search(
        self,
        query: str,
        top_k: int = 4,
        document_ids: Optional[List[int]] = None,
    ) -> List[VectorSearchResult]:
        """
        Rank indexed chunks using Okapi BM25 scoring formula.
        """
        if self.total_docs == 0:
            return []

        query_tokens = self._tokenize(query)
        if not query_tokens:
            return []

        scores: List[Tuple[int, float]] = []

        for idx, chunk in enumerate(self.corpus_chunks):
            # Optional document filter
            if document_ids and chunk.document_id not in document_ids:
                continue

            doc_len = self.doc_lengths[idx]
            tf_dict = self.term_freqs[idx]
            doc_score = 0.0

            for token in query_tokens:
                if token in tf_dict:
                    freq = tf_dict[token]
                    idf_val = self.idf.get(token, 0.0)
                    numerator = freq * (self.k1 + 1.0)
                    denominator = freq + self.k1 * (
                        1.0 - self.b + self.b * (doc_len / (self.avg_doc_len + 1e-9))
                    )
                    doc_score += idf_val * (numerator / denominator)

            if doc_score > 0.0:
                scores.append((idx, doc_score))

        # Sort by score descending
        scores.sort(key=lambda x: x[1], reverse=True)

        results: List[VectorSearchResult] = []
        # Normalize BM25 scores to [0.0, 1.0] range
        max_score = scores[0][1] if scores else 1.0

        for idx, raw_score in scores[:top_k]:
            chunk = self.corpus_chunks[idx]
            norm_score = round(raw_score / (max_score + 1e-9), 4)
            results.append(
                VectorSearchResult(
                    chunk_id=chunk.chunk_id,
                    document_id=chunk.document_id,
                    filename=chunk.filename,
                    page_number=chunk.page_number,
                    text=chunk.text,
                    word_count=chunk.word_count,
                    score=norm_score,
                )
            )

        return results
