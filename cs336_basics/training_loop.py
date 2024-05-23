import numpy as np
import random
import torch

def get_batch(ids, batch_size, context_length, steps, device, start_idx, seed=40):
    """
    Args:
        ids (np.array): token ids
        batch_size: int
        context_length: int
        steps: int
        device: str
        start_idx: int
        seed: int = 40

    Returns:
        X: tensor of shape (batch_size, context_length)
        y: tensor of shape (batch_size, context_length)
    """
    n = len(ids) - context_length
    indices = random.choices(range(n), k=steps * batch_size)
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
    for i in range(start_idx * batch_size, steps * batch_size, batch_size):
        X_batch = [ids[idx    :idx + context_length    ] for idx in indices[i:i + batch_size]]
        y_batch = [ids[idx + 1:idx + context_length + 1] for idx in indices[i:i + batch_size]]
        X = torch.tensor(X_batch, device=device)
        y = torch.tensor(y_batch, device=device)
        yield int(i / batch_size), (X, y)

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