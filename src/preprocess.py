#!/usr/bin/env python
"""src/preprocess.py
Dataset loading & preprocessing utilities (fully implemented).
"""

from __future__ import annotations

import random
from functools import partial
from pathlib import Path
from typing import Dict, Tuple, List, Any

import torch
from torch.utils.data import DataLoader, Dataset

# Optional heavy dependencies – import lazily
try:
    from datasets import load_dataset
    from transformers import AutoTokenizer
except ImportError:  # pragma: no cover – handled via project optional deps
    load_dataset = None  # type: ignore
    AutoTokenizer = None  # type: ignore

# --------------------------------------------------------------------------------------
# Dummy dataset (used for smoke tests)
# --------------------------------------------------------------------------------------


class DummyLanguageModelingDataset(Dataset):
    """Creates random token sequences for next-token prediction."""

    def __init__(self, num_samples: int, seq_length: int, vocab_size: int):
        super().__init__()
        self.num_samples = num_samples
        self.seq_length = seq_length
        self.vocab_size = vocab_size
        rng = random.Random(0)
        self.data = [
            torch.tensor([rng.randint(1, vocab_size - 1) for _ in range(seq_length)])
            for _ in range(num_samples)
        ]

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        x = self.data[idx]
        return x[:-1], x[1:]


# --------------------------------------------------------------------------------------
# CNN/DailyMail summarisation dataset
# --------------------------------------------------------------------------------------


def _shift_right(labels: torch.Tensor, pad_id: int, start_id: int) -> torch.Tensor:
    shifted = labels.new_full(labels.shape, pad_id)
    shifted[..., 1:] = labels[..., :-1]
    shifted[..., 0] = start_id
    return shifted


class CNNDailyMailDataset(Dataset):
    """Pre-tokenised CNN/DailyMail split (train/val)."""

    def __init__(
        self,
        split: str,
        tokenizer_name: str,
        version: str = "3.0.0",
        max_source_length: int = 512,
        max_target_length: int = 128,
        min_article_words: int = 50,
        noise_frac: float = 0.0,
    ) -> None:
        if load_dataset is None:
            raise ImportError("datasets & transformers must be installed for real datasets")

        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=True)
        # Load HF dataset
        ds = load_dataset("cnn_dailymail", version, split=split)
        # Filter short articles
        ds = ds.filter(lambda x: len(x["article"].split()) >= min_article_words)
        self.ds = ds
        self.max_source_length = max_source_length
        self.max_target_length = max_target_length
        self.noise_frac = noise_frac
        self.pad_id = self.tokenizer.pad_token_id
        # decoder_start_token_id fallbacks to BOS if not defined
        self.start_id = (
            self.tokenizer.convert_tokens_to_ids(self.tokenizer.bos_token)
            if self.tokenizer.bos_token_id is None
            else self.tokenizer.bos_token_id
        )

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx: int):
        example = self.ds[int(idx)]
        article = example["article"]
        summary = example["highlights"]

        # Optional noise injection: sample random summary from within the batch
        if self.noise_frac > 0 and random.random() < self.noise_frac:
            rand_idx = random.randint(0, len(self.ds) - 1)
            summary = self.ds[rand_idx]["highlights"]

        tok = self.tokenizer
        model_inputs = tok(
            article,
            max_length=self.max_source_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        with tok.as_target_tokenizer():
            labels_enc = tok(
                summary,
                max_length=self.max_target_length,
                truncation=True,
                padding="max_length",
                return_tensors="pt",
            )
        labels = labels_enc["input_ids"].squeeze(0)
        decoder_input_ids = _shift_right(labels, self.pad_id, self.start_id)
        model_inputs = {
            "input_ids": model_inputs["input_ids"].squeeze(0),
            "attention_mask": model_inputs["attention_mask"].squeeze(0),
            "decoder_input_ids": decoder_input_ids,
        }
        return model_inputs, labels


# --------------------------------------------------------------------------------------
# Public API
# --------------------------------------------------------------------------------------

def load_dataset(cfg: Dict, split_ratio: float = 0.9) -> Tuple[DataLoader, DataLoader, int]:
    """Returns (train_loader, val_loader, vocab_size)."""

    dscfg = cfg["dataset"]
    batch_size = cfg["training"]["batch_size"]
    task_type = cfg["task_type"].lower()

    # ------------------------------------------------------------------
    # Language modelling dummy task (used for CI / smoke)
    # ------------------------------------------------------------------
    if dscfg["name"] == "dummy":
        dataset = DummyLanguageModelingDataset(
            num_samples=dscfg.get("num_samples", 1024),
            seq_length=dscfg.get("seq_length", 32),
            vocab_size=dscfg.get("vocab_size", 256),
        )
        vocab_size = dscfg.get("vocab_size", 256)
        n_train = int(len(dataset) * split_ratio)
        n_val = len(dataset) - n_train
        train_set, val_set = torch.utils.data.random_split(dataset, [n_train, n_val])

        def collate(batch):
            inputs, targets = zip(*batch)
            return torch.stack(inputs), torch.stack(targets)

    # ------------------------------------------------------------------
    # CNN/DailyMail summarisation (seq2seq)
    # ------------------------------------------------------------------
    elif dscfg["name"].lower() == "cnn_dailymail":
        tokenizer_name = dscfg.get("tokenizer_name", cfg["model"]["name"])
        train_set = CNNDailyMailDataset(
            split="train",
            tokenizer_name=tokenizer_name,
            version=dscfg.get("version", "3.0.0"),
            max_source_length=dscfg.get("max_source_length", 512),
            max_target_length=dscfg.get("max_target_length", 128),
            min_article_words=dscfg.get("min_article_words", 50),
            noise_frac=dscfg.get("noise_frac", 0.0),
        )
        val_set = CNNDailyMailDataset(
            split="validation",
            tokenizer_name=tokenizer_name,
            version=dscfg.get("version", "3.0.0"),
            max_source_length=dscfg.get("max_source_length", 512),
            max_target_length=dscfg.get("max_target_length", 128),
            min_article_words=dscfg.get("min_article_words", 50),
            noise_frac=0.0,  # no noise for validation
        )
        vocab_size = train_set.tokenizer.vocab_size

        def collate(batch: List[Any]):
            inputs_list, labels_list = zip(*batch)
            batch_inputs = {k: torch.stack([d[k] for d in inputs_list]) for k in inputs_list[0].keys()}
            batch_labels = torch.stack(labels_list)
            return batch_inputs, batch_labels

    else:
        raise NotImplementedError(f"Dataset '{dscfg['name']}' not implemented.")

    # ------------------------------------------------------------------
    # DataLoaders
    # ------------------------------------------------------------------
    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        collate_fn=collate,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        collate_fn=collate,
    )

    return train_loader, val_loader, vocab_size
