import numpy as np
from collections import Counter
from typing import List, Dict, Any, Tuple
from app.models.document import Document
from app.models.document_page import DocumentPage
from app.schemas.data_science_schemas import (
    CorpusSummaryResponse,
    NGramAnalysisResponse,
    TermFrequencyItem,
    DocumentLengthDistribution,
)
from app.services.data_processing.text_processor import TextProcessor

text_processor = TextProcessor()


class CorpusAnalyzer:
    """
    Corpus Analytics Engine for Vocabulary, N-Gram frequencies, and Document Length Distributions.
    """

    def analyze_corpus(
        self, documents: List[Document], pages: List[DocumentPage]
    ) -> CorpusSummaryResponse:
        total_docs = len(documents)
        total_pages = len(pages)
        total_words = sum(doc.word_count for doc in documents)

        # Document type breakdown
        doc_types: Dict[str, int] = {}
        for doc in documents:
            doc_types[doc.file_type] = doc_types.get(doc.file_type, 0) + 1

        # Upload trends
        upload_trends: Dict[str, int] = {}
        for doc in documents:
            date_str = doc.created_at.strftime("%Y-%m-%d") if doc.created_at else "Unknown"
            upload_trends[date_str] = upload_trends.get(date_str, 0) + 1

        # Length distribution across documents
        doc_lengths = [doc.word_count for doc in documents] if documents else [0]
        length_array = np.array(doc_lengths, dtype=float)

        length_dist = DocumentLengthDistribution(
            min_words=int(np.min(length_array)),
            quantile_25=float(np.percentile(length_array, 25)),
            median_words=float(np.median(length_array)),
            quantile_75=float(np.percentile(length_array, 75)),
            max_words=int(np.max(length_array)),
            mean_words=float(np.mean(length_array)),
            std_words=float(np.std(length_array)) if len(length_array) > 1 else 0.0,
        )

        # Vocabulary and Top Terms Analysis
        all_tokens: List[str] = []
        for page in pages:
            raw_text = page.cleaned_text or page.raw_text
            res = text_processor.process_text(raw_text)
            all_tokens.extend(res.tokens)

        unique_vocab = set(all_tokens)
        vocab_size = len(unique_vocab)
        ttr = float(vocab_size / len(all_tokens)) if all_tokens else 0.0

        term_counts = Counter(all_tokens)
        top_terms = [
            TermFrequencyItem(term=t, frequency=c)
            for t, c in term_counts.most_common(20)
        ]

        return CorpusSummaryResponse(
            total_documents=total_docs,
            total_pages=total_pages,
            total_words=total_words,
            vocabulary_size=vocab_size,
            type_token_ratio=round(ttr, 4),
            doc_types_breakdown=doc_types,
            length_distribution=length_dist,
            top_terms=top_terms,
            upload_trends=upload_trends,
        )

    def generate_ngram_analysis(
        self, pages: List[DocumentPage], top_k: int = 20
    ) -> NGramAnalysisResponse:
        all_tokens: List[str] = []
        for page in pages:
            raw_text = page.cleaned_text or page.raw_text
            res = text_processor.process_text(raw_text)
            all_tokens.extend(res.tokens)

        # 1. Unigrams
        unigram_counts = Counter(all_tokens)

        # 2. Bigrams
        bigrams = [
            f"{all_tokens[i]} {all_tokens[i+1]}"
            for i in range(len(all_tokens) - 1)
        ]
        bigram_counts = Counter(bigrams)

        # 3. Trigrams
        trigrams = [
            f"{all_tokens[i]} {all_tokens[i+1]} {all_tokens[i+2]}"
            for i in range(len(all_tokens) - 2)
        ]
        trigram_counts = Counter(trigrams)

        return NGramAnalysisResponse(
            unigrams=[
                TermFrequencyItem(term=t, frequency=c)
                for t, c in unigram_counts.most_common(top_k)
            ],
            bigrams=[
                TermFrequencyItem(term=t, frequency=c)
                for t, c in bigram_counts.most_common(top_k)
            ],
            trigrams=[
                TermFrequencyItem(term=t, frequency=c)
                for t, c in trigram_counts.most_common(top_k)
            ],
        )
