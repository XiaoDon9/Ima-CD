"""Random seeds, checkpoint I/O and classification metrics."""
import random

import numpy as np
import torch


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def save_checkpoint(state, path):
    # A file handle supports non-ASCII Windows paths in older PyTorch versions.
    with open(path, "wb") as stream:
        torch.save(state, stream)


def load_checkpoint(path, map_location="cpu"):
    with open(path, "rb") as stream:
        return torch.load(stream, map_location=map_location, weights_only=True)


def confusion_matrix(truth, predicted, num_classes):
    matrix = np.zeros((num_classes, num_classes), dtype=np.int64)
    for label, prediction in zip(truth, predicted):
        matrix[int(label), int(prediction)] += 1
    return matrix


def macro_f1(matrix):
    denominator = matrix.sum(0) + matrix.sum(1)
    scores = np.divide(2 * np.diag(matrix), denominator,
                       out=np.zeros(len(matrix), dtype=float), where=denominator != 0)
    return float(scores.mean())
