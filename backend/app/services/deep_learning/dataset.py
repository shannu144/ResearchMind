from typing import List
import torch
from torch.utils.data import Dataset
from app.services.deep_learning.vocab import Vocabulary


class TextDataset(Dataset):
    """
    PyTorch Dataset returning padded sequence tensors and label indices.
    """

    def __init__(
        self,
        texts: List[str],
        labels: List[int],
        vocab: Vocabulary,
        max_seq_len: int = 64,
    ):
        self.texts = texts
        self.labels = labels
        self.vocab = vocab
        self.max_seq_len = max_seq_len

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int):
        encoded = self.vocab.encode(self.texts[idx], max_seq_len=self.max_seq_len)
        return (
            torch.tensor(encoded, dtype=torch.long),
            torch.tensor(self.labels[idx], dtype=torch.long),
        )
