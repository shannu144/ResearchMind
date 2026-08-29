import re
from typing import List, Dict, Set, Tuple
from app.core.logging import logger
from app.schemas.nlp_schemas import EntityItem, NERAnalysisResponse

# Research Domain Knowledge Base for Custom Entity Classification
ALGORITHMS_PATTERNS = [
    r"\b(gradient descent|adamw|adam|q-learning|backpropagation|random forest|support vector machine|svm|bilstm|lstm|transformer|self-attention|markov decision process|convolutions?|resnet|bert|gpt|logistic regression)\b"
]

DATASETS_PATTERNS = [
    r"\b(imagenet|mnist|cifar-10|cifar-100|squad|glue|superglue|wikitext|common crawl|pascal voc|coco|mimic-iii)\b"
]

TECHNOLOGIES_PATTERNS = [
    r"\b(pytorch|tensorflow|fastapi|scikit-learn|pandas|numpy|faiss|chromadb|spacy|nltk|hugging face|docker|postgresql|python|cuda)\b"
]

RESEARCH_CONCEPTS_PATTERNS = [
    r"\b(retrieval-augmented generation|rag|named entity recognition|ner|exploratory data analysis|eda|cosine similarity|type-token ratio|ttr|overfitting|underfitting|hyperparameter|embedding|vector database|quantization|immunotherapy|crispr|entanglement|superconductivity)\b"
]


class NamedEntityExtractor:
    """
    Named Entity Recognition (NER) Service for Research Papers and Technical Documents.
    Extracts People, Organizations, Technologies, Algorithms, Datasets, Locations, and Research Concepts.
    """

    def __init__(self):
        self.nlp = None
        try:
            import spacy
            try:
                self.nlp = spacy.load("en_core_web_sm")
            except Exception:
                pass
        except Exception as e:
            logger.debug(f"spaCy loading fallback: {e}")

    def extract_entities(self, text: str) -> NERAnalysisResponse:
        entities: List[EntityItem] = []
        seen_spans: Set[Tuple[int, int]] = set()

        # 1. spaCy General NER (PERSON, ORG, GPE, DATE, MONEY)
        if self.nlp:
            try:
                doc = self.nlp(text)
                for ent in doc.ents:
                    start, end = ent.start_char, ent.end_char
                    label = ent.label_
                    if label in ["PERSON", "ORG", "GPE", "NORP", "FAC"]:
                        mapped_label = "LOCATION" if label in ["GPE", "FAC"] else label
                        entities.append(
                            EntityItem(
                                text=ent.text,
                                label=mapped_label,
                                start_char=start,
                                end_char=end,
                                confidence=0.95,
                            )
                        )
                        seen_spans.add((start, end))
            except Exception as e:
                logger.error(f"spaCy NER execution failed: {e}")

        # 2. Rule-based Research Entity Recognition (Algorithms, Datasets, Technologies, Concepts)
        patterns_map = [
            ("ALGORITHM", ALGORITHMS_PATTERNS),
            ("DATASET", DATASETS_PATTERNS),
            ("TECH", TECHNOLOGIES_PATTERNS),
            ("CONCEPT", RESEARCH_CONCEPTS_PATTERNS),
        ]

        for label, patterns in patterns_map:
            for pat in patterns:
                for match in re.finditer(pat, text, re.IGNORECASE):
                    start, end = match.start(), match.end()
                    if (start, end) not in seen_spans:
                        entities.append(
                            EntityItem(
                                text=match.group(),
                                label=label,
                                start_char=start,
                                end_char=end,
                                confidence=1.0,
                            )
                        )
                        seen_spans.add((start, end))

        # Count frequencies by type
        counts: Dict[str, int] = {}
        for ent in entities:
            counts[ent.label] = counts.get(ent.label, 0) + 1

        return NERAnalysisResponse(
            entities=entities,
            entity_counts_by_type=counts,
            total_entities=len(entities),
        )
