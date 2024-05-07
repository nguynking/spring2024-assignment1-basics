import regex as re
from collections import Counter
from typing import Tuple
from functools import wraps
import time
import heapq
from collections import defaultdict
from tqdm import tqdm
import logging


logger = logging.getLogger(__name__)
GPT2_PATTERN = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""


def timing(f):
    @wraps(f)
    def wrap(*args, **kw):
        ts = time.time()
        result = f(*args, **kw)
        te = time.time()
        print(f'func:{f.__name__} args:[{args}, {kw}] took: {te-ts:.4f} sec')
        # print(f'func:{f.__name__} took: {te-ts:.4f} sec')
        return result
    return wrap


def get_pretoken_stats(input_path: str, special_tokens: list[str], logger=None):
    """
    Args:
        input_path: str
        special_tokens: list(str)

    Returns:
        pretoken_stats: dict[tuple[bytes]: int]
    """
    # with timed_block("loading file", logger):
    with open(input_path, "r", encoding="utf-8") as f:
        text = f.read()

    # with timed_block("remove special tokens", logger):
    # remove all special tokens from the corpus
    for token in special_tokens:
        text = text.replace(token, "")

    # with timed_block("count pretokens file", logger):
    # dict[str: int], e.g. {'low': 3}
    pretokens = Counter(re.findall(GPT2_PATTERN, text, concurrent=True))
    
    str_to_tuple_bytes = lambda pretoken: tuple(bytes([b]) for b in pretoken.encode("utf-8"))

    # with timed_block("changing pretokens to tuple", logger):
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
    pair_stats_heap = [(-freq, pair) for pair, freq in pair_stats.items()]
    heapq.heapify(pair_stats_heap)
    return pair_stats, pair_stats_heap


def get_bytes_2_pretokens(pretoken_stats):
    # indexing bytes to pretokens
    bytes_2_pretokens = defaultdict(set)
    for pretoken in pretoken_stats:
        for byte in pretoken:
            bytes_2_pretokens[byte].add(pretoken)
    return bytes_2_pretokens


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


@timing
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
    pretoken_stats = get_pretoken_stats(input_path, special_tokens, logger) # dict[tuple[bytes]: int], e.g. {(l, o, w): 5}

    # indexing bytes to pretokens
    bytes_2_pretokens = get_bytes_2_pretokens(pretoken_stats)

    # get the pair stats
    pair_stats, pair_stats_heap = get_pair_stats(pretoken_stats) # dict[tuple(bytes): int], e.g. {(l, o): 1, (o, w): 2}

    merges = []
    # with timed_block("training BPE", logger):
    for _ in tqdm(range(vocab_size - len(vocab))):

        # step 1: get max frequent pair, if there are multiple pairs, choose the lexicographically greater pair
        if len(pair_stats_heap) == 0:
            break

        max_pair, max_freq = None, None
        repush_pairs = []
        
        while len(pair_stats_heap) > 0:
            freq, pair = heapq.heappop(pair_stats_heap)

            # ignore inaccurate frequencies from lazy updates
            if pair_stats[pair] != -freq:
                continue

            # update on first iteration
            if max_pair is None:
                max_pair, max_freq = pair, freq
                continue

            # break if the frequency is not the same as the top frequency
            if freq != max_freq:
                heapq.heappush(pair_stats_heap, (freq, pair))
                break

            # update top token if lexicographically larger
            if pair > max_pair:
                repush_pairs.append(max_pair)
                max_pair = pair
            else:
                repush_pairs.append(pair)

        for pair in repush_pairs:
            heapq.heappush(pair_stats_heap, (max_freq, pair))

        if max_pair is None:
            break

        # step 2: update vocab
        idx = len(vocab)
        vocab[idx] = max_pair[0] + max_pair[1]

        # step 3: update merges
        merges.append(max_pair)

        # step 4: update pretoken_stats, pair_stats and pair_stats_heap
        changed_keys = set()
        
        # we will only check pretokens that have both bytes in the max frequent pair
        exist_pretokens = (bytes_2_pretokens[max_pair[0]] & bytes_2_pretokens[max_pair[1]])
        for pretoken in exist_pretokens:
            # check if the pretoken exist because we dont delete links between bytes with the old pretokens
            if pretoken not in pretoken_stats:
                continue
            
            pretoken_count = pretoken_stats[pretoken]
            del pretoken_stats[pretoken]

            i = 0
            while i < len(pretoken) - 1:
                pair = pretoken[i:i + 2]
                if pair == max_pair:
                    pretoken, prefix, suffix = merge(pretoken, max_pair, i)

                    # update pair_stats
                    if prefix:
                        add_pair = (prefix[-1], vocab[idx])
                        del_pair = (prefix[-1], max_pair[0])
                        pair_stats[add_pair] += pretoken_count
                        pair_stats[del_pair] -= pretoken_count
                        changed_keys.update([add_pair, del_pair])
                        
                    if suffix:
                        add_pair = (vocab[idx] , suffix[0])
                        del_pair = (max_pair[1], suffix[0])
                        pair_stats[add_pair] += pretoken_count
                        pair_stats[del_pair] -= pretoken_count
                        changed_keys.update([add_pair, del_pair])
                        
                i += 1  
            pretoken_stats[pretoken] = pretoken_count

            # we just need to add, dont need to delete links between bytes and old pretokens
            for byte in pretoken:
                bytes_2_pretokens[byte].add(pretoken)
                
        del pair_stats[max_pair]
        for key in list(changed_keys):
            heapq.heappush(
                pair_stats_heap, (-pair_stats[key], key)
            )

    return vocab, merges