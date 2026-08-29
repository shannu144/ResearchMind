import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from app.schemas.nlp_schemas import TextSimilarityResponse


class TextSimilarityService:
    """
    Text Similarity Engine computing Cosine Similarity between document texts.
    """

    def compute_similarity(self, text_a: str, text_b: str) -> TextSimilarityResponse:
        if not text_a.strip() or not text_b.strip():
            return TextSimilarityResponse(
                similarity_score=0.0,
                interpretation="One or both input texts are empty.",
            )

        vectorizer = TfidfVectorizer(stop_words="english", lowercase=True)
        try:
            tfidf_matrix = vectorizer.fit_transform([text_a, text_b])
            sim = float(cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0])
            sim = round(max(0.0, min(1.0, sim)), 4)
        except Exception:
            sim = 0.0

        if sim > 0.8:
            interp = "High Semantic Similarity (Nearly Identical or Highly Overlapping Topics)"
        elif sim > 0.4:
            interp = "Moderate Similarity (Shared Domain or Related Concepts)"
        elif sim > 0.1:
            interp = "Low Similarity (Minor Lexical Overlap)"
        else:
            interp = "No Significant Semantic Similarity Found"

        return TextSimilarityResponse(
            similarity_score=sim,
            metric="Cosine Similarity (TF-IDF Vector Space)",
            interpretation=interp,
        )
