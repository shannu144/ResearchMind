from typing import List, Optional, Dict
from pydantic import BaseModel


class TopicWord(BaseModel):
    word: str
    weight: float


class TopicCluster(BaseModel):
    topic_id: int
    label: str                        # Auto-generated descriptive label
    top_words: List[TopicWord]
    document_ids: List[int]
    document_filenames: List[str]
    coherence_score: float


class LDATopicModelingRequest(BaseModel):
    n_topics: int = 5
    max_features: int = 1000
    n_top_words: int = 10
    document_ids: Optional[List[int]] = None


class LDATopicModelingResponse(BaseModel):
    n_topics: int
    topics: List[TopicCluster]
    model_perplexity: float
    algorithm: str


class KMeansClusteringRequest(BaseModel):
    n_clusters: int = 4
    document_ids: Optional[List[int]] = None


class ClusterNode(BaseModel):
    document_id: int
    filename: str
    cluster_id: int
    distance_to_centroid: float
    top_terms: List[str]


class KMeansClusteringResponse(BaseModel):
    n_clusters: int
    clusters: List[TopicCluster]
    cluster_assignments: List[ClusterNode]
    inertia: float
    algorithm: str
