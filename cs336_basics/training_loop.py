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
    X = torch.tensor(X, device=device)
    y = torch.tensor(y, device=device)
    return (X, y)

def save_checkpoint(model, optimizer, iteration, out):
    """Dump model, optimizer and iteration into file-like object out
    Args:
        model: torch.nn.Module
        optimizer: torch.optim.Optimizer
        iteration: int
        out: str | os.Pathlike | typing.BinaryIO | typing.IO[bytes]
    """
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'iteration': iteration
    }, out)
    
def load_checkpoint(src, model, optimizer):
    """Load a checkpoint from src
    Args:
        src: str | os.Pathlike | typing.BinaryIO | typing.IO[bytes]
        model: torch.nn.Module
        optimizer: torch.optim.Optimizer 
    """
    checkpoint = torch.load(src)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    iteration = checkpoint['iteration']
    return iteration