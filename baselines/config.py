"""Recorded baseline protocol (window normalization, Adam LR 0.001)."""
from contextlib import contextmanager

from imacd.config import (ANGLE_N_PER_REV, ANGLE_REVS_PER_SEG, CLASSES, DATA_ROOT,
                          DEFAULT_BATCH_SIZE, DEFAULT_MAX_SEGMENTS_PER_FILE,
                          DEFAULT_NUM_WORKERS, DEFAULT_SEED, DEFAULT_SEG_DOMAIN,
                          NUM_CLASSES, SEG_LEN, class_to_idx)

METHODS = ("source_only", "dann", "coral", "msdal")
TARGET_MODES = ("none", "unlabeled")
PROTOCOL_TRAIN_SPEEDS = [20, 25, 30, 35]
PROTOCOL_VAL_SPEEDS = [40]
PROTOCOL_TEST_SPEEDS = [45, 50, 55]
DEFAULT_TRAIN_INSTALLATIONS = [1]
DEFAULT_TARGET_MODE = "unlabeled"


@contextmanager
def data_settings(data_root):
    global DATA_ROOT
    previous = DATA_ROOT
    DATA_ROOT = str(data_root)
    try:
        yield
    finally:
        DATA_ROOT = previous
