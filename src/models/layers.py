import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple

class PositionalEncoding(nn.Module):
    """
    Positional encoding for transformer models.
    Adds information about the position of each element in the sequence.
    """
    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Add positional encoding to input tensor
        
        Args:
            x: Input tensor of shape [seq_len, batch_size, d_model]
            
        Returns:
            Tensor with positional encoding added
        """
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

class TemporalAttention(nn.Module):
    """
    Multi-head self-attention mechanism focusing on temporal relationships
    """
    def __init__(self, d_model: int, nhead: int, dropout: float = 0.1):
        super(TemporalAttention, self).__init__()
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(
        self, 
        query: torch.Tensor, 
        key: torch.Tensor, 
        value: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply temporal attention mechanism
        
        Args:
            query: Query tensor
            key: Key tensor
            value: Value tensor
            attn_mask: Optional attention mask
            
        Returns:
            Tuple of (output tensor, attention weights)
        """
        # Multi-head attention
        attn_output, attn_weights = self.multihead_attn(
            query, key, value, attn_mask=attn_mask
        )
        
        # Add & norm
        output = self.norm(query + self.dropout(attn_output))
        
        return output, attn_weights

class CrossAssetAttention(nn.Module):
    """
    Attention mechanism to model relationships between different assets
    """
    def __init__(self, d_model: int, nhead: int, dropout: float = 0.1):
        super(CrossAssetAttention, self).__init__()
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self, 
        assets: torch.Tensor,
        asset_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply cross-asset attention
        
        Args:
            assets: Tensor of asset representations [batch_size, num_assets, d_model]
            asset_mask: Optional mask for assets
            
        Returns:
            Tuple of (enriched representations, attention weights)
        """
        # Cross-asset attention
        attn_output, attn_weights = self.multihead_attn(
            assets, assets, assets, attn_mask=asset_mask
        )
        
        # Add & norm
        output = self.norm(assets + self.dropout(attn_output))
        
        return output, attn_weights

class FeedForward(nn.Module):
    """
    Feed-forward network used in transformer blocks
    """
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super(FeedForward, self).__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.norm = nn.LayerNorm(d_model)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply feed-forward network
        
        Args:
            x: Input tensor
            
        Returns:
            Processed tensor
        """
        ff_output = self.linear2(self.dropout(F.relu(self.linear1(x))))
        output = self.norm(x + self.dropout(ff_output))
        return output

class RegimeDetectionModule(nn.Module):
    """
    Module to detect market regimes
    """
    def __init__(self, d_model: int, num_regimes: int = 3):
        super(RegimeDetectionModule, self).__init__()
        self.regime_detector = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(d_model // 2, num_regimes),
            nn.Softmax(dim=-1)
        )
        self.num_regimes = num_regimes
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Detect market regime probabilities
        
        Args:
            x: Input tensor representing market state
            
        Returns:
            Regime probability distribution
        """
        # x shape: [batch_size, d_model]
        regime_probs = self.regime_detector(x)
        return regime_probs  # [batch_size, num_regimes]
