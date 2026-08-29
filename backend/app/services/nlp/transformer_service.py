import re
from typing import Optional
from app.core.logging import logger
from app.schemas.nlp_schemas import SummarizationResponse

try:
    from transformers import pipeline as hf_pipeline
except ImportError:
    hf_pipeline = None


class TransformerPipelineService:
    """
    Hugging Face Transformer Pipeline Service for Abstractive Summarization and NLP Reasoning.
    """

    def __init__(self, model_name: str = "sshleifer/distilbart-cnn-6-6"):
        self.model_name = model_name
        self.summarizer = None
        # Lazy initialization to avoid startup delay if model weights are not pre-downloaded

    def _init_summarizer(self):
        if self.summarizer is None and hf_pipeline is not None:
            try:
                self.summarizer = hf_pipeline(
                    "summarization", model=self.model_name, device=-1
                )
                logger.info(f"Loaded HuggingFace Transformer model: {self.model_name}")
            except Exception as e:
                logger.warning(f"Failed to load Transformer model {self.model_name}: {e}")
                self.summarizer = False  # Mark as unavailable for fallback

    def summarize_text(
        self, text: str, max_length: int = 150, min_length: int = 40
    ) -> SummarizationResponse:
        words = text.split()
        original_count = len(words)

        if original_count <= 20:
            return SummarizationResponse(
                summary=text,
                original_word_count=original_count,
                summary_word_count=original_count,
                compression_ratio=1.0,
                model_used="Raw Text (Input too short for summarization)",
            )

        self._init_summarizer()

        # 1. Try Hugging Face Transformer Abstractive Summarizer
        if self.summarizer:
            try:
                # Truncate input text to ~1000 tokens for transformer context window
                truncated_text = " ".join(words[:600])
                res = self.summarizer(
                    truncated_text,
                    max_length=max_length,
                    min_length=min_length,
                    do_sample=False,
                )
                summary_text = res[0]["summary_text"]
                sum_count = len(summary_text.split())
                comp_ratio = round(sum_count / original_count, 4) if original_count > 0 else 1.0

                return SummarizationResponse(
                    summary=summary_text,
                    original_word_count=original_count,
                    summary_word_count=sum_count,
                    compression_ratio=comp_ratio,
                    model_used=f"HuggingFace Transformer ({self.model_name})",
                )
            except Exception as e:
                logger.error(f"Transformer summarization execution error: {e}")

        # 2. Heuristic Extractive Summarization Fallback (Top Salient Sentences)
        sentences = re.split(r"(?<=[.!?])\s+", text)
        if len(sentences) <= 3:
            summary_text = text
        else:
            # Select first sentence + middle key sentence + last sentence
            summary_text = f"{sentences[0]} {sentences[len(sentences)//2]} {sentences[-1]}"

        sum_count = len(summary_text.split())
        comp_ratio = round(sum_count / original_count, 4) if original_count > 0 else 1.0

        return SummarizationResponse(
            summary=summary_text,
            original_word_count=original_count,
            summary_word_count=sum_count,
            compression_ratio=comp_ratio,
            model_used="Heuristic Extractive Summarizer Engine (Fallback)",
        )
