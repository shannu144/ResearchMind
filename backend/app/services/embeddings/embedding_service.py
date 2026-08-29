import numpy as np
from typing import List
from app.core.config import settings
from app.core.logging import logger

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    SentenceTransformer = None


class EmbeddingService:
    """
    Sentence-Transformers Vector Embedding Service (`all-MiniLM-L6-v2`, 384 dimensions).
    Computes L2-normalized vector embeddings for text chunks and queries.
    """

    def __init__(self, model_name: str = None):
        self.model_name = model_name or settings.EMBEDDING_MODEL_NAME
        self.dimension = settings.EMBEDDING_DIMENSION
        self.model = None

    def _init_model(self):
        if self.model is None and SentenceTransformer is not None:
            try:
                self.model = SentenceTransformer(self.model_name)
                logger.info(f"Loaded SentenceTransformer model: {self.model_name}")
            except Exception as e:
                logger.warning(f"Failed to load SentenceTransformer {self.model_name}: {e}")
                self.model = False  # Mark as unavailable for fallback

    def encode(self, texts: List[str]) -> np.ndarray:
        """
        Encodes list of texts into L2-normalized float32 numpy vector array [N, dim].
        """
        self._init_model()

        if self.model:
            try:
                embeddings = self.model.encode(
                    texts, convert_to_numpy=True, normalize_embeddings=True
                )
                return embeddings.astype(np.float32)
            except Exception as e:
                logger.error(f"SentenceTransformer encoding error: {e}")

        # Fallback deterministic pseudo-embedding generation based on hash values
        embeddings = []
        for text in texts:
            vec = np.zeros(self.dimension, dtype=np.float32)
            words = text.lower().split()
            for i, word in enumerate(words):
                idx = abs(hash(word)) % self.dimension
                vec[idx] += 1.0
            norm = np.linalg.norm(vec)
            if norm > 0:
                vec = vec / norm
            embeddings.append(vec)

        return np.array(embeddings, dtype=np.float32)
