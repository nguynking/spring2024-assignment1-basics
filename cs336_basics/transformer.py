import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from einops import einsum, rearrange, reduce, repeat

class RMSNorm(nn.Module):
    def __init__(self, d_model, eps=1e-5, device=None):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d_model, device=device))
        self.eps = eps

    def set_weights_from_dict(self, d):
        # manually assign the weights, if the keys match, we can use load_state_dict() instead
        self.weight.data = d['weight']

    def forward(self, x):
        """
        a: (..., d_model)
        """
        return x * self.weight / torch.sqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)

def gelu(x):
    return x * 0.5 * (1.0 + torch.erf(x / sqrt(2.0)))

class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.w1 = nn.Linear(d_model, d_ff, bias=False)
        self.w2 = nn.Linear(d_ff, d_model, bias=False)
        self.gelu = gelu

    def set_weights_from_dict(self, d):
        self.w1.weight.data = d['w1.weight']
        self.w2.weight.data = d['w2.weight']

    def forward(self, x):
        return self.w2(self.gelu(self.w1(x)))

def softmax(x, dim):
    x_adjusted = x - x.max(dim=dim, keepdim=True).values
    x_exp = torch.exp(x_adjusted)
    return x_exp / x_exp.sum(dim=dim, keepdim=True)

def scaled_dot_product_attention(K, Q, V, mask=None, pdrop=None):
    """
    Args
        query: (B, ..., S, K)
        key:   (B, ..., S, K)
        value: (B, ..., S, V)
        mask:  (S, S)
        pdrop: float

    Returns
        output: (B, ..., S, V)
    """
    d_c, d_k = Q.shape[-2:]
    
    # (B, ..., S, K) @ (B, ..., K, S) -> (B, ..., S, S)
    QK_T = einsum(Q, K, '... s1 k, ... s2 k -> ... s1 s2') / math.sqrt(d_k)
    # QK_T = Q @ K.transpose(-1, -2) / math.sqrt(d_k)

    # apply mask
    if mask is not None:
        QK_T = QK_T.masked_fill(mask, -torch.inf)

    scores = softmax(QK_T, dim=-1)
    if pdrop is not None:
        scores = F.dropout(scores, p=pdrop)
        
    return scores @ V

class CausalMultiheadAttention(nn.Module):
    """Given the key, query, and value projection weights of a naive unbatched
    implementation of multi-head attention, return the output of an optimized batched
    implementation. This implementation should handle the key, query, and value projections
    for all heads in a single matrix multiply.
    See section 3.2.2 of Vaswani et al., 2017.

    Args:
        d_model: int
            Dimensionality of the feedforward input and output.
        num_heads: int
            Number of heads to use in multi-headed attention.
        attn_pdrop: float
            Drop-out the attention probabilities (the softmax-normalized
            attention scores) with this rate.
        weights: dict[str, torch.FloatTensor]
            State dict of our reference implementation.
            The keys of this dictionary are:
            - `q_heads.{N}.weight`, `q_heads.{N}.weight`:
                Weights for the query projection heads.
                N is an integer from 0 to `num_heads - 1`.
                Shape of each tensor is (d_key, d_model).
            - `k_heads.{N}.weight`, `k_heads.{N}.weight`:
                Weights for the key projection heads.
                N is an integer from 0 to `num_heads - 1`.
                Shape of each tensor is (d_key, d_model).
            - `v_heads.{N}.weight`, `v_heads.{N}.weight`:
                Weights for the value projection heads.
                N is an integer from 0 to `num_heads - 1`.
                Shape of each tensor is (d_value, d_model).
            - `output_proj.weight`:
                Weight of the output projection
                (W^{O} in the original Transformer paper)
                Shape of (d_model, d_value * num_heads).
        in_features: torch.FloatTensor
            Tensor to run your implementation on.

    Returns:
        torch.FloatTensor with the output of running your optimized, batched multi-headed attention
        implementation with the given QKV projection weights and input features.
    """
    def __init__(self, d_model, num_heads, attn_pdrop, max_seq_len=512):
        """d_k = d_v = d_model / h"""
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.attn_pdrop = attn_pdrop
        d_k = d_model // num_heads
        self.d_k = d_k
        
        self.w_qkv = nn.Parameter(torch.empty(3, num_heads, d_k, d_model))
        self.w_o = nn.Linear(d_model, d_model, bias=False)

    def set_weights_from_dict(self, d):
        # q_heads.{N}.weight
        for qkvi, qkv_name in enumerate('qkv'):
            for head in range(self.num_heads):
                self.w_qkv.data[qkvi, head] = d[f'{qkv_name}_heads.{head}.weight']
        self.w_o.weight.data = d['output_proj.weight']
        
    def forward(self, x):
        """
        b: batch size
        s: sequence length
        m: model dimension
        h: number of heads
        k: key dimension
        """
        # x: (..., s, m)
        seq_len = x.size(-2)

        qkv_heads = einsum(x, self.w_qkv, '... s m , qkv h k m -> ... qkv h s k')
        Q, K, V = (qkv_heads[..., i, :, :, :] for i in range(3))

        # attn_out: ( ..., k)
        mask = torch.triu(torch.ones(seq_len, seq_len, dtype=torch.bool), diagonal=1)
        attn_out = scaled_dot_product_attention(K, Q, V, mask=mask, pdrop=self.attn_pdrop)
        
        # concat all heads
        attn_concatenated = rearrange(attn_out, '... h s k -> ... s (h k)')

        # final linear layer
        out = einsum(attn_concatenated, self.w_o.weight, '... s dim_out, m dim_out -> ... s m')
        
        return out