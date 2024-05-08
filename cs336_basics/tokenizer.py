from typing import Iterable
import regex as re
from collections import defaultdict
import os
import json
from cs336_basics import merge

class Tokenizer:
    def __init__(self, vocab, merges, special_tokens=None):
        """
        Args
            vocab: dict[int, bytes]
            merges: list[tuple[bytes, bytes]]
            special_tokens: list[str] | None = None
        """
        self.vocab = vocab
        self.inv_vocab = {b:i for i, b in self.vocab.items()}
        self.merges = merges
        self.special_tokens = special_tokens if special_tokens else None
        self.compiled_pattern = re.compile(r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")

    @classmethod
    def from_files(cls, vocab_filepath, merges_filepath, special_tokens=None):
        """Use trained GPT-2 tokenizer
        Args
            vocab_filepath: str
            merges_filepath: str
            special_tokens: list[str] | None = None
        """
        with open(vocab_filepath, "r", encoding="utf-8") as f:
            vocab = json.load(f)
            vocab = {idx: ch.encode("utf-8") for ch, idx in vocab.items()}
        
        with open(merges_filepath, "r", encoding="utf-8") as f:
            merges_data = f.read()
            merges = [tuple(ch.encode("utf-8") for ch in merge_str.split()) for merge_str in merges_data.split("\n")[1:-1]]

        # for some reasons, there are some missing bytes in the gpt-2 vocab
        idx = len(vocab)
        for i in range(256):
            if bytes([i]) not in vocab.values():
                vocab[idx] = bytes([i])
                idx += 1
            
        if special_tokens:
            for i, token in enumerate(special_tokens):
                if token.encode("utf-8") not in vocab.values():
                    vocab[i + len(vocab)] = token.encode("utf-8")
        
        return cls(vocab, merges, special_tokens)

    def merge_idx(byte1, byte2):
        try:
            return self.merges.index((byte1, byte2))
        except:
            return float("inf")
        
    def encode(self, text: str) -> list[int]:
        pretokens = re.findall(self.compiled_pattern, text)
        print(pretokens)
        ids = []
        for pretoken in pretokens:
            text_bytes = tuple(bytes([b]) for b in pretoken.encode("utf-8"))
            while len(text_bytes) >= 2:
                pairs = [pair for pair in zip(text_bytes, text_bytes[1:])]
                pair = min(pairs, key=self.merge_idx)
                if pair not in self.merges:
                    break
                idx = pairs.index(pair)
                text_bytes, _, _ = merge(text_bytes, pair, idx)
            ids.extend([self.inv_vocab[b] for b in text_bytes])
        return ids

    def encode_iterable(self, iterable: Iterable[str]) -> Iterable[int]:
        for text in iterable:
            for i in self.encode(text):
                yield i

    def decode(self, ids: list[int]) -> str:
        text_bytes = b"".join([self.vocab[i] for i in ids])
        text = text_bytes.decode("utf-8", errors="replace")
        return text