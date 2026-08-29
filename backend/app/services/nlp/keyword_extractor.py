import re
from collections import Counter
from typing import List
from sklearn.feature_extraction.text import TfidfVectorizer
from app.schemas.nlp_schemas import KeywordItem, KeywordExtractionResponse


class KeywordExtractor:
    """
    Keyword and Keyphrase Extraction Engine using TF-IDF n-gram scoring.
    """

    def extract_keywords(
        self, text: str, top_k: int = 10, ngram_range: tuple = (1, 2)
    ) -> KeywordExtractionResponse:
        if not text or not text.strip():
            return KeywordExtractionResponse(keywords=[])

        try:
            vectorizer = TfidfVectorizer(
                max_features=100,
                ngram_range=ngram_range,
                stop_words="english",
                lowercase=True,
            )
            tfidf_matrix = vectorizer.fit_transform([text])
            feature_names = vectorizer.get_feature_names_out()
            scores = tfidf_matrix.toarray()[0]

            keyword_scores = [
                (feature_names[i], float(scores[i])) for i in range(len(feature_names))
            ]
            keyword_scores.sort(key=lambda x: x[1], reverse=True)

            items = [
                KeywordItem(keyword=kw, score=round(sc, 4))
                for kw, sc in keyword_scores[:top_k]
            ]
            return KeywordExtractionResponse(keywords=items)
        except Exception:
            # Fallback simple frequency count
            words = re.findall(r"\b\w{3,}\b", text.lower())
            counts = Counter(words).most_common(top_k)
            items = [KeywordItem(keyword=w, score=float(c)) for w, c in counts]
            return KeywordExtractionResponse(keywords=items)
