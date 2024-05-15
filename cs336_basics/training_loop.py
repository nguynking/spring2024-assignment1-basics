import numpy as np
import random
import torch

def get_batch(ids, batch_size, context_length, device):
    """
    Args:
        ids (np.array): token ids
        batch_size: int
        context_length: int
        device: str

    Returns:
        X: tensor of shape (batch_size, context_length)
        y: tensor of shape (batch_size, context_length)
    """
    n = len(ids) - context_length
    idxs = random.sample(range(n), batch_size)
    X, y = [], []
    for idx in idxs:
        X.append(ids[idx: idx + context_length])
        y.append(ids[idx + 1: idx + context_length + 1])
    X = torch.tensor(np.array(X), device=device)
    y = torch.tensor(np.array(y), device=device)
    return (X, y)