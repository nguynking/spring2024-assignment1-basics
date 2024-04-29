import regex as re

def get_stats(ids, counts=None):
    counts = counts if not None else {}
    for pair in zip(ids, ids[1:]):
        counts[pair] = counts.get(pair, 0) + 1
    return counts

def merge(ids, pair, idx):
    newids = []
    i = 0
    while i < len(ids):
        if i < len(ids) - 1 and ids[i] == pair[0] and ids[i + 1] == pair[1]:
            newids.append(idx)
            i += 2
        else:
            newids.append(ids[i])
            i += 1
    return newids

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
    num_stokens = len(special_tokens)
    # Vocabulary initialization
    vocab = {k: stoken.encode("utf-8") for k, stoken in enumerate(special_tokens)}
    vocab.update({b + num_stokens: bytes([b]) for b in range(256)})
    
    assert vocab_size > len(vocab), f"Increase your vocab size to at least {len(vocab) + 1}!"
    num_merges = vocab_size - len(vocab)

    # Read contents
    with open(input_path, "r", encoding="utf-8") as f:
        text = f.read()

    for token in special_tokens:
        text = text.replace(token, "")
    
    # Pre-tokenization
    PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    text_chunks = re.findall(PAT, text) # str -> list[str]
    ids = [text.encode("utf-8") for text in text_chunks]
    # print(ids)

    # Compute BPR merges
    merges = []
    for i in range(num_merges):
        stats = {}
        for chunk_ids in ids:
            stats = get_stats(chunk_ids, stats) # get counts
        if not stats:
            break
        pair = max(stats, key=stats.get) # extract the maximum frequent pair
        idx = len(vocab)
        ids = [merge(chunk_ids, pair, idx) for chunk_ids in ids] # merge each chunk with the max pair
        vocab[idx] = vocab[pair[0]] + vocab[pair[1]]
        merges.append((vocab[pair[0]], vocab[pair[1]]))
        # print(f"merge {i+1}/{num_merges}: {pair} -> {idx} ({vocab[idx]}) had {stats[pair]} occurrences")
    
    return vocab, merges