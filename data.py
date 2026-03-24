"""
Penn Treebank data loader
Port of Facebook's char-rnn data.lua
"""

import torch
import numpy as np
from pathlib import Path

ptb_path = "./data/"

vocab_idx = 0
vocab_map = {}


def load_data(fname):
    """Load and tokenize a text file"""
    global vocab_idx, vocab_map

    with open(fname, "r", encoding="utf-8") as f:
        data = f.read()

    # Replace newlines with <eos>
    data = data.replace("\n", "<eos>")

    # Split into words
    words = data.split()

    print(f"Loading {fname}, size of data = {len(words)}")

    # Map words to indices
    x = torch.zeros(len(words), dtype=torch.long)
    for i, word in enumerate(words):
        if word not in vocab_map:
            vocab_idx += 1
            vocab_map[word] = vocab_idx
        x[i] = vocab_map[word]

    return x


def replicate(x_inp, batch_size):
    """
    Stacks replicated, shifted versions of x_inp into a single matrix
    of size (len(x_inp) // batch_size) x batch_size
    """
    s = x_inp.size(0)
    new_size = s // batch_size

    # Create output tensor
    x = torch.zeros(new_size, batch_size, dtype=x_inp.dtype)

    for i in range(batch_size):
        start = int(round((i * s) / batch_size))
        finish = start + new_size
        x[:, i] = x_inp[start:finish]

    return x


def testdataset(batch_size):
    """
    Intentionally we repeat dimensions without offsetting.
    Pass over this batch corresponds to the fully sequential processing.
    """
    x = load_data(ptb_path + "ptb.test.txt")

    # Reshape: (len, 1) -> expand to (len, batch_size)
    x = x.unsqueeze(1).expand(x.size(0), batch_size)

    return x


def traindataset(batch_size):
    """Load training data with replication"""
    x = load_data(ptb_path + "ptb.train.txt")
    x = replicate(x, batch_size)
    return x


def validdataset(batch_size):
    """Load validation data with replication"""
    x = load_data(ptb_path + "ptb.valid.txt")
    x = replicate(x, batch_size)
    return x


def get_vocab_map():
    """Return the vocabulary mapping"""
    return vocab_map


if __name__ == "__main__":
    # Test loading
    train_data = traindataset(20)
    print(f"Train data shape: {train_data.shape}")

    valid_data = validdataset(20)
    print(f"Valid data shape: {valid_data.shape}")

    test_data = testdataset(20)
    print(f"Test data shape: {test_data.shape}")

    print(f"Vocab size: {vocab_idx + 1}")
