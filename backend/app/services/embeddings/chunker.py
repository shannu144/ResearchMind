import re
from typing import List
from app.schemas.embedding_schemas import ChunkMetadata, DocumentChunk


class TextChunker:
    """
    Semantic Text Chunker splitting document page texts into overlapping chunks
    while preserving page numbers, document IDs, and chunk metadata.
    """

    def chunk_document(
        self,
        document_id: int,
        filename: str,
        pages_data: List[dict],  # list of dicts with keys: page_number, raw_text
        chunk_size: int = 500,
        chunk_overlap: int = 100,
    ) -> List[DocumentChunk]:
        chunks: List[DocumentChunk] = []
        global_chunk_idx = 1

        for page in pages_data:
            page_num = page["page_number"]
            raw_text = page["raw_text"].strip()
            if not raw_text:
                continue

            # Split into sentence units
            sentences = re.split(r"(?<=[.!?])\s+", raw_text)
            current_chunk_words: List[str] = []
            current_char_count = 0

            for sent in sentences:
                sent_words = sent.split()
                if not sent_words:
                    continue

                sent_char_count = len(sent)
                if current_char_count + sent_char_count > chunk_size and current_chunk_words:
                    # Emit current chunk
                    chunk_text = " ".join(current_chunk_words)
                    chunk_id = f"doc_{document_id}_p{page_num}_c{global_chunk_idx}"
                    meta = ChunkMetadata(
                        chunk_id=chunk_id,
                        document_id=document_id,
                        filename=filename,
                        page_number=page_num,
                        text=chunk_text,
                        word_count=len(current_chunk_words),
                    )
                    chunks.append(DocumentChunk(chunk_id=chunk_id, metadata=meta))
                    global_chunk_idx += 1

                    # Retain overlap words from the end of current chunk
                    overlap_words: List[str] = []
                    overlap_char_count = 0
                    for w in reversed(current_chunk_words):
                        if overlap_char_count + len(w) <= chunk_overlap:
                            overlap_words.insert(0, w)
                            overlap_char_count += len(w) + 1
                        else:
                            break

                    current_chunk_words = overlap_words + sent_words
                    current_char_count = sum(len(w) + 1 for w in current_chunk_words)
                else:
                    current_chunk_words.extend(sent_words)
                    current_char_count += sent_char_count + 1

            # Emit remaining words on page
            if current_chunk_words:
                chunk_text = " ".join(current_chunk_words)
                chunk_id = f"doc_{document_id}_p{page_num}_c{global_chunk_idx}"
                meta = ChunkMetadata(
                    chunk_id=chunk_id,
                    document_id=document_id,
                    filename=filename,
                    page_number=page_num,
                    text=chunk_text,
                    word_count=len(current_chunk_words),
                )
                chunks.append(DocumentChunk(chunk_id=chunk_id, metadata=meta))
                global_chunk_idx += 1

        return chunks
