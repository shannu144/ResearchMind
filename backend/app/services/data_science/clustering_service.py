import re
import numpy as np
from typing import List, Optional, Dict, Tuple
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize
from app.core.logging import logger
from app.schemas.clustering_schemas import (
    TopicWord,
    TopicCluster,
    LDATopicModelingRequest,
    LDATopicModelingResponse,
    KMeansClusteringRequest,
    ClusterNode,
    KMeansClusteringResponse,
)


class DocumentClusteringService:
    """
    Document Clustering & Topic Modeling Service implementing:
    - Latent Dirichlet Allocation (LDA) for probabilistic topic discovery.
    - KMeans clustering on TF-IDF vectors for hard document partitioning.

    Both techniques are standard unsupervised ML approaches for understanding
    thematic structure in document corpora without labelled training data.
    """

    def _clean_text(self, text: str) -> str:
        text = re.sub(r"[^a-zA-Z\s]", " ", text.lower())
        text = re.sub(r"\s+", " ", text).strip()
        return text

    def _auto_label_topic(self, top_words: List[str]) -> str:
        """Generate a descriptive topic label from its top 3 words."""
        return " / ".join(w.capitalize() for w in top_words[:3])

    # ─── LDA Topic Modeling ───────────────────────────────────────────────────

    def run_lda(
        self,
        documents: List[Dict],   # [{"id": int, "filename": str, "text": str}]
        n_topics: int = 5,
        max_features: int = 1000,
        n_top_words: int = 10,
    ) -> LDATopicModelingResponse:
        """
        Latent Dirichlet Allocation — discovers latent topic distributions
        across a corpus. Each document is a mixture of topics; each topic
        is a distribution over words.
        """
        texts = [self._clean_text(d["text"]) for d in documents]
        doc_ids = [d["id"] for d in documents]
        doc_filenames = [d["filename"] for d in documents]

        # Guard: need at least n_topics documents
        if len(texts) < 2:
            return LDATopicModelingResponse(
                n_topics=n_topics,
                topics=[],
                model_perplexity=0.0,
                algorithm="LDA (Latent Dirichlet Allocation)",
            )

        n_topics_actual = min(n_topics, len(texts))

        # Bag-of-Words vectorization for LDA
        vectorizer = CountVectorizer(
            max_features=max_features,
            stop_words="english",
            min_df=1,
            ngram_range=(1, 2),
        )
        try:
            X = vectorizer.fit_transform(texts)
        except ValueError as e:
            logger.warning(f"LDA vectorization failed: {e}")
            return LDATopicModelingResponse(
                n_topics=n_topics_actual,
                topics=[],
                model_perplexity=0.0,
                algorithm="LDA (Latent Dirichlet Allocation)",
            )

        feature_names = vectorizer.get_feature_names_out()

        # LDA Model
        lda = LatentDirichletAllocation(
            n_components=n_topics_actual,
            max_iter=20,
            learning_method="online",
            random_state=42,
        )
        doc_topic_matrix = lda.fit_transform(X)  # shape: [n_docs, n_topics]

        perplexity = float(lda.perplexity(X))

        # Build topic clusters
        topic_clusters = []
        for topic_idx, topic_dist in enumerate(lda.components_):
            top_word_indices = topic_dist.argsort()[: -n_top_words - 1 : -1]
            top_words = [
                TopicWord(
                    word=feature_names[i],
                    weight=round(float(topic_dist[i] / topic_dist.sum()), 5),
                )
                for i in top_word_indices
            ]

            # Assign documents whose dominant topic = this topic
            dominant_topics = np.argmax(doc_topic_matrix, axis=1)
            topic_doc_indices = np.where(dominant_topics == topic_idx)[0].tolist()
            topic_doc_ids = [doc_ids[i] for i in topic_doc_indices]
            topic_doc_filenames = [doc_filenames[i] for i in topic_doc_indices]

            label = self._auto_label_topic([w.word for w in top_words])
            topic_clusters.append(
                TopicCluster(
                    topic_id=topic_idx,
                    label=label,
                    top_words=top_words,
                    document_ids=topic_doc_ids,
                    document_filenames=topic_doc_filenames,
                    coherence_score=round(float(topic_dist.max() / max(topic_dist.sum(), 1e-9)), 4),
                )
            )

        return LDATopicModelingResponse(
            n_topics=n_topics_actual,
            topics=topic_clusters,
            model_perplexity=round(perplexity, 2),
            algorithm="LDA (Latent Dirichlet Allocation)",
        )

    # ─── KMeans Clustering ────────────────────────────────────────────────────

    def run_kmeans(
        self,
        documents: List[Dict],
        n_clusters: int = 4,
    ) -> KMeansClusteringResponse:
        """
        KMeans Clustering on L2-normalized TF-IDF vectors.
        Partitions documents into k clusters by minimising within-cluster
        sum-of-squared distances (inertia).
        """
        texts = [self._clean_text(d["text"]) for d in documents]
        doc_ids = [d["id"] for d in documents]
        doc_filenames = [d["filename"] for d in documents]

        if len(texts) < 2:
            return KMeansClusteringResponse(
                n_clusters=n_clusters,
                clusters=[],
                cluster_assignments=[],
                inertia=0.0,
                algorithm="KMeans (TF-IDF Vectors)",
            )

        n_clusters_actual = min(n_clusters, len(texts))

        tfidf = TfidfVectorizer(
            max_features=500,
            stop_words="english",
            min_df=1,
            ngram_range=(1, 2),
        )
        try:
            X = tfidf.fit_transform(texts)
        except ValueError as e:
            logger.warning(f"KMeans vectorization failed: {e}")
            return KMeansClusteringResponse(
                n_clusters=n_clusters_actual,
                clusters=[],
                cluster_assignments=[],
                inertia=0.0,
                algorithm="KMeans (TF-IDF Vectors)",
            )

        X_norm = normalize(X, norm="l2")
        feature_names = tfidf.get_feature_names_out()

        km = KMeans(n_clusters=n_clusters_actual, random_state=42, n_init=10, max_iter=300)
        labels = km.fit_predict(X_norm)
        inertia = float(km.inertia_)

        # Per-cluster top terms from centroid
        cluster_nodes: List[ClusterNode] = []
        cluster_map: Dict[int, List[int]] = {i: [] for i in range(n_clusters_actual)}

        for doc_idx, cluster_id in enumerate(labels):
            cluster_map[int(cluster_id)].append(doc_idx)

        # Distance to centroid
        distances = km.transform(X_norm)   # [n_docs, n_clusters]
        for doc_idx, cluster_id in enumerate(labels):
            dist = float(distances[doc_idx, cluster_id])
            centroid_vec = km.cluster_centers_[cluster_id]
            top_term_indices = centroid_vec.argsort()[:-6:-1]
            top_terms = [feature_names[i] for i in top_term_indices]
            cluster_nodes.append(ClusterNode(
                document_id=doc_ids[doc_idx],
                filename=doc_filenames[doc_idx],
                cluster_id=int(cluster_id),
                distance_to_centroid=round(dist, 5),
                top_terms=top_terms,
            ))

        # Build TopicCluster objects per cluster
        topic_clusters = []
        for cluster_id in range(n_clusters_actual):
            centroid = km.cluster_centers_[cluster_id]
            top_idx = centroid.argsort()[:-n_clusters_actual * 3:-1][:10]
            top_words = [
                TopicWord(word=feature_names[i], weight=round(float(centroid[i]), 5))
                for i in top_idx
            ]
            member_indices = cluster_map[cluster_id]
            label = self._auto_label_topic([w.word for w in top_words])
            topic_clusters.append(TopicCluster(
                topic_id=cluster_id,
                label=label,
                top_words=top_words,
                document_ids=[doc_ids[i] for i in member_indices],
                document_filenames=[doc_filenames[i] for i in member_indices],
                coherence_score=round(1.0 / (1.0 + float(centroid.std())), 4),
            ))

        return KMeansClusteringResponse(
            n_clusters=n_clusters_actual,
            clusters=topic_clusters,
            cluster_assignments=cluster_nodes,
            inertia=round(inertia, 4),
            algorithm="KMeans (TF-IDF Vectors, L2-normalized)",
        )
