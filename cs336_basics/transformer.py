import torch
import torch.nn as nn

class RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-5, device=None):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size, device=device))
        self.eps = eps

    def set_weights_from_dict(self, d):
        self.weight.data = d['weight']

    def forward(self, x):
        """
        a: (..., d_model)
        """
        return x * self.weight / torch.sqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps) 