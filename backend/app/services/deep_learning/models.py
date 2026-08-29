import torch
import torch.nn as nn


class BiLSTMClassifier(nn.Module):
    """
    PyTorch Bidirectional LSTM (BiLSTM) Neural Network for Text Sequence Classification.
    
    Architecture:
    Input Token Tensors [B, L]
          ↓
    Embedding Layer (vocab_size -> embed_dim) [B, L, E]
          ↓
    BiLSTM Layers (num_layers=2, bidirectional=True) -> Forward & Backward Hidden States [B, L, H*2]
          ↓
    Mean Pooling across time dimension -> [B, H*2]
          ↓
    Dropout (0.3)
          ↓
    Linear Output Layer (H*2 -> num_classes) [B, C]
    """

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 128,
        hidden_dim: int = 128,
        num_classes: int = 5,
        num_layers: int = 2,
        dropout: float = 0.3,
        pad_idx: int = 0,
    ):
        super(BiLSTMClassifier, self).__init__()
        self.embedding = nn.Embedding(
            vocab_size, embed_dim, padding_idx=pad_idx
        )
        self.bilstm = nn.LSTM(
            embed_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.fc = nn.Linear(hidden_dim * 2, num_classes)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: [batch_size, seq_len]
        embedded = self.embedding(x)  # [batch_size, seq_len, embed_dim]
        bilstm_out, _ = self.bilstm(embedded)  # [batch_size, seq_len, hidden_dim * 2]

        # Mean pooling across sequence length dimension
        pooled = torch.mean(bilstm_out, dim=1)  # [batch_size, hidden_dim * 2]
        pooled = self.dropout(pooled)

        logits = self.fc(pooled)  # [batch_size, num_classes]
        return logits
