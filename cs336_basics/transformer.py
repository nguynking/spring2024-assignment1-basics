import torch
import torch.nn as nn
from math import sqrt

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
        self.w1 = nn.Linear(d_ff, d_model, bias=False)
        self.w2 = nn.Linear(d_model, d_ff, bias=False)
        self.gelu = gelu

    def set_weights_from_dict(self, d):
        self.w1.weight.data = d['w1.weight']
        self.w2.weight.data = d['w2.weight']

    def forward(self, x):
        return self.w2(self.gelu(self.w1(x)))