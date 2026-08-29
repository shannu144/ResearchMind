import re
from typing import List, Dict, Set, Optional


class Vocabulary:
    """
    Word-to-Index & Index-to-Word mapping Vocabulary for PyTorch NLP models.
    """

    PAD_TOKEN = "<PAD>"
    UNK_TOKEN = "<UNK>"

    def __init__(self, max_vocab_size: Optional[int] = 5000):
        self.max_vocab_size = max_vocab_size
        self.word2idx: Dict[str, int] = {self.PAD_TOKEN: 0, self.UNK_TOKEN: 1}
        self.idx2word: Dict[int, str] = {0: self.PAD_TOKEN, 1: self.UNK_TOKEN}
        self.pad_idx = 0
        self.unk_idx = 1

    def build_vocab(self, texts: List[str]) -> None:
        word_freq: Dict[str, int] = {}
        for text in texts:
            tokens = re.findall(r"\b\w+\b", text.lower())
            for token in tokens:
                word_freq[token] = word_freq.get(token, 0) + 1

        sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
        if self.max_vocab_size:
            sorted_words = sorted_words[: self.max_vocab_size - 2]

        idx = 2
        for word, _ in sorted_words:
            if word not in self.word2idx:
                self.word2idx[word] = idx
                self.idx2word[idx] = word
                idx += 1

    def encode(self, text: str, max_seq_len: int = 64) -> List[int]:
        tokens = re.findall(r"\b\w+\b", text.lower())
        indices = [self.word2idx.get(t, self.unk_idx) for t in tokens]

        # Truncate or Pad sequence to max_seq_len
        if len(indices) >= max_seq_len:
            return indices[:max_seq_len]
        else:
            return indices + [self.pad_idx] * (max_seq_len - len(indices))

    def __len__(self) -> int:
        return len(self.word2idx)
