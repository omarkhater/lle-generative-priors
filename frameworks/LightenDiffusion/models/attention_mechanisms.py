import math
import torch
import torch.nn as nn
from torch.nn import functional as F
from einops import rearrange
from .operators import DepthConv

class Self_Attention(nn.Module):
    """
    Implements self-attention on spatial features using convolutions and normalization.
    """
    def __init__(self, dim: int, num_heads: int, bias: bool) -> None:
        super(Self_Attention, self).__init__()
        self.num_heads = num_heads
        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=(1, 1), bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim * 3, dim * 3, kernel_size=(3, 3), stride=(1, 1),
                                    padding=1, groups=dim * 3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=(1, 1), bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply self-attention mechanism.
        
        Args:
            x (torch.Tensor): Input tensor.
        
        Returns:
            torch.Tensor: Output tensor after applying self-attention.
        """
        b, c, h, w = x.shape

        qkv = self.qkv_dwconv(self.qkv(x))
        q, k, v = qkv.chunk(3, dim=1)

        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1))
        attn = attn.softmax(dim=-1)

        out = (attn @ v)

        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        out = self.project_out(out)
        return out


class Cross_Attention(nn.Module):
    """
    Computes cross-attention between a query and a context using depthwise separable convolutions.
    """
    def __init__(self, dim: int, num_heads: int, dropout: float = 0.) -> None:
        super(Cross_Attention, self).__init__()
        if dim % num_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention heads (%d)" % (dim, num_heads)
            )
        self.num_heads = num_heads
        self.attention_head_size = int(dim / num_heads)

        self.query = DepthConv(in_channels=dim, out_channels=dim)
        self.key = DepthConv(in_channels=dim, out_channels=dim)
        self.value = DepthConv(in_channels=dim, out_channels=dim)

        self.dropout = nn.Dropout(dropout)

    def transpose_for_scores(self, x: torch.Tensor) -> torch.Tensor:
        """
        Reshape and permute tensor for attention score computation.
        
        Args:
            x (torch.Tensor): Input tensor.
        
        Returns:
            torch.Tensor: Transposed tensor.
        """
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states: torch.Tensor, ctx: torch.Tensor) -> torch.Tensor:
        """
        Apply cross-attention mechanism.
        
        Args:
            hidden_states (torch.Tensor): Query tensor.
            ctx (torch.Tensor): Context tensor for key and value.
        
        Returns:
            torch.Tensor: Output tensor after cross-attention.
        """
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(ctx)
        mixed_value_layer = self.value(ctx)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)

        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        attention_probs = self.dropout(attention_probs)

        ctx_layer = torch.matmul(attention_probs, value_layer)
        ctx_layer = ctx_layer.permute(0, 2, 1, 3).contiguous()

        return ctx_layer
