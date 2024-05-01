import regex as re
import time

def get_pretoken_stats(text_chunks):
    """text_chunks -> ids_chunks
    Example: ["hi", "hi", "my", "name"] -> {"hi": 2, "my": 1, "name": 1} -> {(104, 105): 2, (109, 121): 1, (110, 97, 109, 101): 1}
    """
    counts = {}
    for text in text_chunks:
        ids = tuple(text.encode("utf-8", errors="replace"))
        counts[ids] = counts.get(ids, 0) + 1
    return counts

def get_stats(pretoken_stats, stats=None):
    # pretoken_stats = {tuple(text.encode("utf-8", errors="replace")): count for text, count in chunk_stats.items()}
    stats = stats if not None else {}
    for ids, count in pretoken_stats.items():
        for pair in zip(ids, ids[1:]):
            stats[pair] = stats.get(pair, 0) + count
    return stats

def merge(pretoken_stats, pair, idx):
    newpstats = {}
    for ids, count in pretoken_stats.items():
        newids = []
        i = 0
        while i < len(ids):
            if i < len(ids) - 1 and ids[i] == pair[0] and ids[i + 1] == pair[1]:
                newids.append(idx)
                i += 2
            else:
                newids.append(ids[i])
                i += 1
        newpstats.update({tuple(newids): count})
    return newpstats        

def train_bpe(input_path: str, vocab_size: int, special_tokens: list[str]):
    """Function to train a BPE tokenizer

    Args:
    - input_path (str): Path to a text file with BPE tokenizer training data.
    - vocab_size (int): A non-negative integer that defines the maximum final vocabulary size (including the initial byte vocabulary,
        vocabulary items produced from merging, and any special tokens).
    - special_tokens (list[str]): A list of strings to add to the vocabulary. These special tokens do not otherwise affect BPE training.

    Returns:
    - vocab (dict[int, bytes]): The tokenizer vocabulary, a mapping from int (token ID in the vocabulary) to bytes (token bytes).
    - merges (list[tuple[bytes, bytes]]): A list of BPE merges produced from training. Each list item is a tuple of bytes (<token1>, <token2>),
        representing that <token1> is merged with <token2>. The merges should be ordered by order of creation.
    """
    with open(input_path, "r", encoding="utf-8") as f:
        s = f.read()
    # initialize the vocab
    # special_tokens = ["<|endoftext|>"]
    vocab = { i:bytes([i]) for i in range(256) } # int -> byte
    # pre-tokenization
    PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    text_chunks = re.findall(PAT, s)# s.split()
    # count pretoken
    pretoken_stats = get_pretoken_stats(text_chunks)
    # merges
    num_merges = vocab_size - len(vocab) - len(special_tokens)
    merges = []
    for i in range(num_merges):
        stats = {} # {byte pair: count}
        get_stats(pretoken_stats, stats)
        max_count = max(stats.values())
        max_stats = {pair:(vocab[pair[0]], vocab[pair[1]]) for pair, count in stats.items() if count == max_count}
        pair = max(max_stats, key=max_stats.get)
        idx = 256 + i
        pretoken_stats = merge(pretoken_stats, pair, idx)
        vocab[idx] = vocab[pair[0]] + vocab[pair[1]]
        merges.append((vocab[pair[0]], vocab[pair[1]]))
        # print(f"merge {i+1}/{num_merges}: {pair} -> {idx} ({vocab[idx]}) had {stats[pair]} occurrences")
    lvocab = len(vocab)
    vocab.update({i + lvocab: token.encode("utf-8") for i, token in enumerate(special_tokens)})
    return vocab, merges