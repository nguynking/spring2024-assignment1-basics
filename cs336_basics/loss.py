import torch

def cross_entropy(logits, targets):
    # logits: (batch_size, vocab_size)
    # targets: (batch_size)
    scaled_logits = logits - logits.max(dim=-1, keepdim=True).values # scaled_logits: (batch_size, vocab_size)
    # -log(exp(o_i) / sum(exp(o_i))) = -o_i + log(sum(exp(o_i))
    loss = -scaled_logits.gather(dim=-1, index=targets.unsqueeze(-1)).squeeze() + scaled_logits.exp().sum(-1).log()
    return loss.mean()