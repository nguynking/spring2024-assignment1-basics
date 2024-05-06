import regex as re
from collections import Counter
from typing import Tuple
GPT2_PATTERN = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""


def get_pretoken_stats(input_path: str, special_tokens: list[str]):
    """
    Args:
        input_path: str
        special_tokens: list(str)

    Returns:
        pretoken_stats: dict[tuple[bytes]: int]
    """
    with open(input_path, "r", encoding="utf-8") as f:
        text = f.read()

    # remove all special tokens from the corpus
    for token in special_tokens:
        text = text.replace(token, "")

    # dict[str: int], e.g. {'low': 3}
    pretokens = Counter(re.findall(GPT2_PATTERN, text))
    
    str_to_tuple_bytes = lambda pretoken: tuple(bytes([b]) for b in pretoken.encode("utf-8"))

    # dict[tuple[bytes]: int], e.g. {('l', 'o', 'm'): 3}
    pretoken_stats = {str_to_tuple_bytes(pretoken): freq for pretoken, freq in pretokens.items()}
    return pretoken_stats


def get_pair_stats(pretoken_stats):
    """
    Args:
        pretoken_stats: dict[tuple[bytes]: int]
        stats: Counter
    
    Returns:
        pair_stats: dict[tuple[bytes, bytes]: int]
    """
    pair_stats =  Counter()
    for ids, count in pretoken_stats.items():
        for pair in zip(ids, ids[1:]):
            pair_stats[pair] += count
    return pair_stats


def merge(pretoken_tuple, max_pair, i):
    """
    Args:
        pretoken_tuple: tuple[bytes]
        i: int

    Returns:
        updated_pretoken_tuple: tuple[bytes]
        prefix: tuple[bytes]
        suffix: tuple[bytes]
    """
    prefix = pretoken_tuple[:i]
    suffix = pretoken_tuple[i + 2:]
    updated_pretoken_tuple = prefix + (b"".join(max_pair),) + suffix
    return updated_pretoken_tuple, prefix, suffix

def update_stats(pretoken_stats, pair_stats, max_pair, vocab, idx):
    # update pretoken_stats and pair_stats
    updated_pretoken_stats = Counter()
    for pretoken_tuple, freq in pretoken_stats.items():
        i = 0
        while i < len(pretoken_tuple) - 1:
            pair = pretoken_tuple[i:i + 2]
            if pair == max_pair:
                pretoken_tuple, prefix, suffix = merge(pretoken_tuple, max_pair, i)

                # update pair_stats
                if prefix:
                    add_pair = (prefix[-1], vocab[idx])
                    del_pair = (prefix[-1], max_pair[0])
                    pair_stats[add_pair] += freq
                    pair_stats[del_pair] -= freq

                if suffix:
                    add_pair = (vocab[idx] , suffix[0])
                    del_pair = (max_pair[1], suffix[0])
                    pair_stats[add_pair] += freq
                    pair_stats[del_pair] -= freq

                # always delete the max frequent pair
                pair_stats[max_pair] -= freq
            i += 1

        # update the pretoken_stats
        updated_pretoken_stats[pretoken_tuple] = freq
    return updated_pretoken_stats, pair_stats

def train_bpe(input_path: str, vocab_size: int, special_tokens: list[str]):
    """
    Args:
        input_path (str): path to the text file
        vocab_size (int): the desired size of the vocab
        special_tokens (list[str]): all special tokens while training

    Returns:
        vocab (dict[int, bytes]): the obtained vocab after training
        merges (list[tuple[bytes, bytes]]): all pair that are merged during training
    """
    # initialize the vocab, including special tokens
    vocab = { i: bytes([i]) for i in range(256) } # dict[int: bytes]
    for i, token in enumerate(special_tokens):
        vocab[256 + i] = token.encode("utf-8")

    # pre-tokenization
    pretoken_stats = get_pretoken_stats(input_path, special_tokens) # dict[tuple[bytes]: int], e.g. {(l, o, w): 5}

    # get the pair stats
    pair_stats = get_pair_stats(pretoken_stats) # dict[tuple(bytes): int], e.g. {(l, o): 1, (o, w): 2}

    merges = []
    while len(vocab) < vocab_size:
        # get max frequent pair, if there are multiple pairs, choose the lexicographically greater pair
        max_pair = max(pair_stats, key=lambda pair: (pair_stats[pair], pair)) # tuple(bytes)

        # update vocab
        idx = len(vocab)
        vocab[idx] = b"".join(max_pair)

        # update merges
        merges.append(max_pair)

        # update pretoken_stats and pair_stats
        pretoken_stats, pair_stats = update_stats(pretoken_stats, pair_stats, max_pair, vocab, idx)
        
    return vocab, merges