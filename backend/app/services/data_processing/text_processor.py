import re
import string
from typing import List, Dict, Any, Set, Optional
from app.core.logging import logger
from app.schemas.document_schemas import TextPreprocessingConfig, TextPreprocessingResult

# Standard English stopwords set fallback
DEFAULT_STOPWORDS: Set[str] = {
    "a", "about", "above", "after", "again", "against", "all", "am", "an", "and", "any", "are", "aren't",
    "as", "at", "be", "because", "been", "before", "being", "below", "between", "both", "but", "by", "can't",
    "cannot", "could", "couldn't", "did", "didn't", "do", "does", "doesn't", "doing", "don't", "down", "during",
    "each", "few", "for", "from", "further", "had", "hadn't", "has", "hasn't", "have", "haven't", "having", "he",
    "he'd", "he'll", "he's", "her", "here", "here's", "hers", "herself", "him", "himself", "his", "how", "how's",
    "i", "i'd", "i'll", "i'm", "i've", "if", "in", "into", "is", "isn't", "it", "it's", "its", "itself", "let's",
    "me", "more", "most", "mustn't", "my", "myself", "no", "nor", "not", "of", "off", "on", "once", "only", "or",
    "other", "ought", "our", "ours", "ourselves", "out", "over", "own", "same", "shan't", "she", "she'd", "she'll",
    "she's", "should", "shouldn't", "so", "some", "such", "than", "that", "that's", "the", "their", "theirs",
    "them", "themselves", "then", "there", "there's", "these", "they", "they'd", "they'll", "they're", "they've",
    "this", "those", "through", "to", "too", "under", "until", "up", "very", "was", "wasn't", "we", "we'd", "we'll",
    "we're", "we've", "were", "weren't", "what", "what's", "when", "when's", "where", "where's", "which", "while",
    "who", "who's", "whom", "why", "why's", "with", "won't", "would", "wouldn't", "you", "you'd", "you'll",
    "you're", "you've", "your", "yours", "yourself", "yourselves"
}


class TextProcessor:
    """
    NLP Text Preprocessing Service.
    Cleans, tokenizes, segments sentences, handles stopwords, and normalizes text.
    """

    def __init__(self):
        self.stopwords = DEFAULT_STOPWORDS
        # Try loading NLTK stopwords if available
        try:
            import nltk
            from nltk.corpus import stopwords
            try:
                self.stopwords = set(stopwords.words("english"))
            except Exception:
                nltk.download("stopwords", quiet=True)
                self.stopwords = set(stopwords.words("english"))
        except Exception as e:
            logger.debug(f"NLTK stopwords fallback to default set: {e}")

    def segment_sentences(self, text: str) -> List[str]:
        """
        Segment text into individual sentences.
        """
        if not text:
            return []
        # Sentence splitting pattern looking for terminal punctuation followed by space/newline
        sentences = re.split(r"(?<=[.!?])\s+", text)
        return [s.strip() for s in sentences if s.strip()]

    def tokenize(self, text: str) -> List[str]:
        """
        Tokenize text into lower-case alphanumeric tokens.
        """
        if not text:
            return []
        tokens = re.findall(r"\b\w+\b", text.lower())
        return tokens

    def process_text(
        self,
        raw_text: str,
        document_id: int = 0,
        page_number: int = 1,
        config: Optional[TextPreprocessingConfig] = None,
    ) -> TextPreprocessingResult:
        """
        Processes text according to configuration options while keeping raw_text intact.
        """
        if config is None:
            config = TextPreprocessingConfig()

        sentences = self.segment_sentences(raw_text)
        
        # 1. Normalization
        text = raw_text
        if config.lowercase:
            text = text.lower()

        # 2. Tokenization
        tokens = re.findall(r"\b\w+\b", text)

        # 3. Punctuation handling
        if config.remove_punctuation:
            tokens = [t.translate(str.maketrans("", "", string.punctuation)) for t in tokens if t]

        # 4. Stopwords & minimum length
        filtered_tokens = []
        for token in tokens:
            if not token:
                continue
            if len(token) < config.min_word_length:
                continue
            if config.remove_stopwords and token.lower() in self.stopwords:
                continue
            filtered_tokens.append(token)

        cleaned_text = " ".join(filtered_tokens)
        unique_words = set(filtered_tokens)

        return TextPreprocessingResult(
            document_id=document_id,
            page_number=page_number,
            raw_text=raw_text,
            cleaned_text=cleaned_text,
            tokens=filtered_tokens,
            sentences=sentences,
            word_count=len(filtered_tokens),
            unique_word_count=len(unique_words),
        )
