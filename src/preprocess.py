#!/usr/bin/env python
"""src/preprocess.py
Dataset loading and preprocessing utilities for language-model experiments.
"""

from __future__ import annotations

import itertools
from pathlib import Path
from typing import Dict, Tuple, List

import torch
from torch.utils.data import Dataset, DataLoader

# HuggingFace
from datasets import load_dataset as hf_load_dataset
from transformers import GPT2TokenizerFast

# --------------------------------------------------------------------------------------
# Synthetic dummy dataset (kept for smoke-tests)
# --------------------------------------------------------------------------------------


class DummyLanguageModelingDataset(Dataset):
    """Creates random token sequences for next-token prediction."""

    def __init__(self, num_samples: int, seq_length: int, vocab_size: int):
        super().__init__()
        g = torch.Generator().manual_seed(0)
        self.data = torch.randint(1, vocab_size, (num_samples, seq_length + 1), generator=g)
        self.vocab_size = vocab_size

    def __len__(self):
        return self.data.size(0)

    def __getitem__(self, idx):
        seq = self.data[idx]
        return seq[:-1], seq[1:]  # input, target


# --------------------------------------------------------------------------------------
# Real language-modeling datasets (WikiText-2 / 103)
# --------------------------------------------------------------------------------------


class PackedLMSequenceDataset(Dataset):
    """Packs a flat token stream into fixed-length sequences for next-token prediction.

    Each item returns (input_ids, target_ids) where target_ids = input_ids shifted by +1.
    """

    def __init__(self, token_ids: torch.Tensor, seq_length: int):
        super().__init__()
        # Drop the tail that doesn’t fit a full (seq_length+1) window
        window = seq_length + 1
        usable_len = (token_ids.size(0) // window) * window
        self.tokens = token_ids[:usable_len]
        self.seq_length = seq_length
        self.num_sequences = usable_len // window

    def __len__(self):
        return self.num_sequences

    def __getitem__(self, idx):
        window = self.seq_length + 1
        start = idx * window
        end = start + window
        chunk = self.tokens[start:end]
        return chunk[:-1], chunk[1:]


# --------------------------------------------------------------------------------------
# Helper functions
# --------------------------------------------------------------------------------------


def _get_tokenizer(tokenizer_name: str = "gpt2") -> GPT2TokenizerFast:
    tokenizer = GPT2TokenizerFast.from_pretrained(tokenizer_name)
    # GPT-2 doesn’t have pad token by default – assign eos as pad for batching.
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def _load_wikitext(name: str, seq_length: int) -> Tuple[Dataset, Dataset, int]:
    """Load WikiText-2 or ‑103 from HF Datasets and build Packed datasets."""

    variant_map = {
        "wikitext-2": "wikitext-2-raw-v1",
        "wikitext-103": "wikitext-103-raw-v1",
    }
    if name not in variant_map:
        raise ValueError(f"Unsupported WikiText dataset: {name}")

    raw_ds = hf_load_dataset("wikitext", variant_map[name])
    tokenizer = _get_tokenizer()

    def _tokenise(split_name: str) -> torch.Tensor:
        texts = raw_ds[split_name]["text"]
        # Filter empty lines for cleanliness
        texts = [t for t in texts if len(t.strip()) > 0]
        # Encode and concatenate with EOS delimiters
        eos_id = tokenizer.eos_token_id
        ids: List[int] = list(
            itertools.chain.from_iterable(
                [(tokenizer.encode(t, add_special_tokens=False) + [eos_id]) for t in texts]
            )
        )
        return torch.tensor(ids, dtype=torch.long)

    train_tokens = _tokenise("train")
    val_tokens = _tokenise("validation")

    train_set = PackedLMSequenceDataset(train_tokens, seq_length)
    val_set = PackedLMSequenceDataset(val_tokens, seq_length)
    vocab_size = tokenizer.vocab_size
    return train_set, val_set, vocab_size


# --------------------------------------------------------------------------------------
# Public API
# --------------------------------------------------------------------------------------


def load_dataset(cfg: Dict, split_ratio: float = 0.9) -> Tuple[DataLoader, DataLoader, int]:
    """Create DataLoaders according to *cfg* and return (train_loader, val_loader, vocab_size)."""

    dscfg = cfg["dataset"]
    batch_size = cfg["training"]["batch_size"]
    seq_length = dscfg.get("seq_length", 1024)

    if dscfg["name"] == "dummy":
        dataset = DummyLanguageModelingDataset(
            num_samples=dscfg.get("num_samples", 1024),
            seq_length=dscfg.get("seq_length", 32),
            vocab_size=dscfg.get("vocab_size", 256),
        )
        # Simple random split
        n_train = int(len(dataset) * split_ratio)
        n_val = len(dataset) - n_train
        train_set, val_set = torch.utils.data.random_split(dataset, [n_train, n_val])
        vocab_size = dscfg.get("vocab_size", 256)
    elif dscfg["name"].startswith("wikitext"):
        train_set, val_set, vocab_size = _load_wikitext(dscfg["name"], seq_length)
    else:
        raise NotImplementedError(f"Dataset '{dscfg['name']}' is not supported.")

    # ------- DataLoaders -------
    def collate(batch):
        inputs, targets = zip(*batch)
        return torch.stack(inputs), torch.stack(targets)

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        collate_fn=collate,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        collate_fn=collate,
        pin_memory=True,
    )

    return train_loader, val_loader, vocab_size