from collections.abc import Callable, Iterable
from typing import Optional
import torch
import math

class AdamW(torch.optim.Optimizer):
    def __init__(self, params, lr=0.001, betas=(0.9, 0.999), weight_decay=0.01, eps=1e-8):
        defaults = dict(
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
        )
        super().__init__(params, defaults)

    def step(self, closure: Optional[Callable] = None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            weight_decay = group["weight_decay"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]
                grad = p.grad
                t = state.get("t", 1) # Get number of iterations, else return 1
                m = state.get("m", torch.zeros_like(p)) # get the first moment vector
                v = state.get("v", torch.zeros_like(p)) # get the second moment vector

                # doing gradient descent
                m = beta1 * m + (1 - beta1) * grad
                v = beta2 * v + (1 - beta2) * grad**2
                alpha_t = lr * math.sqrt(1 - beta2**t) / (1 - beta1**t)
                p.data -= alpha_t * m / (torch.sqrt(v) + eps)
                p.data -= lr * weight_decay * p.data

                # update state
                state["t"] = t + 1
                state["m"] = m
                state["v"] = v
        return loss


def lr_cosine_schedule(t, lr_max, lr_min, warm_up, cosine_annealing):
    if t < warm_up:
        lr_t = (t / warm_up) * lr_max
    elif t >= warm_up and t <= cosine_annealing:
        lr_t = lr_min + 0.5 * (1 + math.cos((t - warm_up) * math.pi / (cosine_annealing - warm_up))) * (lr_max - lr_min)
    else:
        lr_t = lr_min
    return lr_t