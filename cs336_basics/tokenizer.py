from typing import Iterable
import regex as re
from collections import defaultdict
import os
import json
from cs336_basics import merge
from tests.common import gpt2_bytes_to_unicode

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
        self.merges = {merge: i for i, merge in enumerate(merges)}
        self.special_tokens = special_tokens if special_tokens else None
        self.compiled_pattern = re.compile(r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")

    @classmethod
    def from_files(cls, vocab_path, merges_path, special_tokens=None):
        """Use trained GPT-2 tokenizer
        Args
            vocab_filepath: str
            merges_filepath: str
            special_tokens: list[str] | None = None
        """
        gpt2_byte_decoder = {v: k for k, v in gpt2_bytes_to_unicode().items()}
        with open(vocab_path) as vocab_f:
            gpt2_vocab = json.load(vocab_f)
        gpt2_bpe_merges = []
        with open(merges_path) as f:
            for line in f:
                cleaned_line = line.rstrip()
                if cleaned_line and len(cleaned_line.split(" ")) == 2:
                    gpt2_bpe_merges.append(tuple(cleaned_line.split(" ")))
        # The GPT-2 tokenizer uses a remapped unicode encoding for bytes. Let's
        # just return the original bytes, so we don't force students to use
        # any particular encoding scheme.
        vocab = {
            gpt2_vocab_index: bytes([gpt2_byte_decoder[token] for token in gpt2_vocab_item])
            for gpt2_vocab_item, gpt2_vocab_index in gpt2_vocab.items()
        }
        # If any of the special tokens don't exist in the vocab, append them to the vocab.
        if special_tokens:
            for special_token in special_tokens:
                byte_encoded_special_token = special_token.encode("utf-8")
                if byte_encoded_special_token not in set(vocab.values()):
                    vocab[len(vocab)] = byte_encoded_special_token
    
        merges = [
            (
                bytes([gpt2_byte_decoder[token] for token in merge_token_1]),
                bytes([gpt2_byte_decoder[token] for token in merge_token_2]),
            )
            for merge_token_1, merge_token_2 in gpt2_bpe_merges
        ]
        return cls(vocab, merges, special_tokens)
        
    def encode_ordinary(self, text: str) -> list[int]:
        pretokens = re.findall(self.compiled_pattern, text)
        # print(pretokens)
        ids = []
        for pretoken in pretokens:
            text_bytes = tuple(bytes([b]) for b in pretoken.encode("utf-8"))
            while len(text_bytes) >= 2:
                pairs = [pair for pair in zip(text_bytes, text_bytes[1:])]
                pair = min(pairs, key=lambda p: self.merges.get(p, float("inf")))
                # print(f"merge before {pair}")
                if pair not in self.merges:
                    break
                # print(f"merge after {pair}")
                idx = pairs.index(pair)
                text_bytes, _, _ = merge(text_bytes, pair, idx)
            ids.extend([self.inv_vocab[b] for b in text_bytes])
        return ids

    def encode(self, text):
        if not self.special_tokens:
            return self.encode_ordinary(text)
            
        special_pattern = re.compile("(" + "|".join(map(re.escape, sorted(self.special_tokens, key=lambda x: len(x), reverse=True))) + ")")
        special_chunks = re.split(special_pattern, text)
        ids = []
        for part in special_chunks:
            if part in self.special_tokens:
                ids.append(self.inv_vocab[part.encode("utf-8")])
            else:
                ids.extend(self.encode_ordinary(part))
        return ids

    def encode_iterable(self, iterable: Iterable[str]) -> Iterable[int]:
        for text in iterable:
            for i in self.encode(text):
                yield i

    def decode(self, ids: list[int]) -> str:
        text_bytes = b"".join([self.vocab[i] for i in ids])
        text = text_bytes.decode("utf-8", errors="replace")
        return text