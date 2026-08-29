from typing import List
from sklearn.feature_extraction.text import TfidfVectorizer


class TFIDFVectorizerService:
    """
    TF-IDF Vectorizer Service for extracting term-frequency inverse-document-frequency features.
    """

    def __init__(self, max_features: int = 5000, ngram_range: tuple = (1, 2)):
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            stop_words="english",
            lowercase=True,
        )

    def fit_transform(self, texts: List[str]):
        return self.vectorizer.fit_transform(texts)

    def transform(self, texts: List[str]):
        return self.vectorizer.transform(texts)

    def get_feature_names(self) -> List[str]:
        return list(self.vectorizer.get_feature_names_out())
