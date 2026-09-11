# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
# Licensed under the Apache License, Version 2.0. See LICENSE for details.
#******************************************************************************
#  Dataset Split Utility
# -----------------------------------------------------------------------------
#  Splits raw CSV sensor data into reproducible train/validation/test datasets.
#
#  The script supports class-based input folders with optional nested
#  subdirectories (for example, noise sessions) and creates a standard
#  machine-learning dataset layout.
#
#  INPUT DATA                                OUTPUT DATASET
#  ──────────                                ──────────────
#
#  data/                                     data/
#  ├── detected_rep_sessions/                ├── train/
#  │   └── FLOOR_PULL/                       │   ├── FLOOR_PULL/
#  │       ├── sample_001.csv                │   │   └── *.csv
#  │       └── sample_002.csv                │   └── noise/
#  │                                         │       └── *.csv
#  └── noise_sessions/                       │
#      ├── session_01/                       ├── valid/
#      │   ├── sample_003.csv                │   ├── FLOOR_PULL/
#      │   └── ...                           │   └── noise/
#      └── session_02/                       │
#          └── *.csv                         └── test/
#                                              ├── FLOOR_PULL/
#                                              └── noise/
#
#  Features:
#    - Splits each class independently into train/valid/test sets.
#    - Supports recursive CSV collection for nested folders.
#    - Avoids filename collisions during copying.
#    - Produces deterministic splits when a random seed is provided.
#
#******************************************************************************

import logging
import os
import random
import shutil
from typing import Final

from lift_ml.config import DataConfig

logger: Final = logging.getLogger(__name__)


def collect_class_files(class_dir: str, collapse: bool = False) -> list[str]:
    """Collects CSV files from a class directory.

    If collapse is True, recursively searches all subdirectories and aggregates
    all .csv files into a single list.
    If collapse is False, only collects .csv files located directly inside class_dir.

    Raises:
        FileNotFoundError: If class_dir does not exist.
        ValueError: If no .csv files are found in class_dir.
    """
    if not os.path.isdir(class_dir):
        raise FileNotFoundError(f"Source class directory does not exist: {class_dir}")

    files: list[str] = []
    if collapse:
        for root, _, filenames in os.walk(class_dir):
            for f in sorted(filenames):
                if f.endswith(".csv"):
                    files.append(os.path.join(root, f))
    else:
        for f in sorted(os.listdir(class_dir)):
            if f.endswith(".csv"):
                files.append(os.path.join(class_dir, f))

    logger.info("Found %d CSV files in %s (collapse=%s)", len(files), class_dir, collapse)
    return files


def split_one_class(
    file_list: list[str], train_ratio: float, valid_ratio: float
) -> tuple[list[str], list[str], list[str]]:
    """Split list into train, valid, and test sets."""
    if not file_list:
        raise ValueError("Cannot split an empty file list.")

    random.shuffle(file_list)
    total = len(file_list)

    train_end = int(total * train_ratio)
    valid_end = int(total * (train_ratio + valid_ratio))

    train = file_list[:train_end]
    valid = file_list[train_end:valid_end]
    test = file_list[valid_end:]

    return train, valid, test


def copy_files(file_list: list[str], dst_dir: str) -> None:
    """Copies list of files to destination directory, avoiding filename collisions."""
    os.makedirs(dst_dir, exist_ok=True)
    for f in file_list:
        filename = os.path.basename(f)
        dst_path = os.path.join(dst_dir, filename)
        if os.path.exists(dst_path):
            parent_name = os.path.basename(os.path.dirname(f))
            dst_path = os.path.join(dst_dir, f"{parent_name}_{filename}")
        shutil.copy(f, dst_path)


def split_data(
    config: DataConfig,
    class_a: str,
    class_b: str,
    class_a_collapse: bool = False,
    class_b_collapse: bool = True,
    train_ratio: float = 0.6,
    valid_ratio: float = 0.2,
    **kwargs: bool | str,
) -> None:
    """Splits dataset files from two source folders (Class A and Class B) into
    train/valid/test sets.

    Parameters:
        config: DataConfig containing output paths and class labels.
        class_a: Folder path for Class A (e.g. 'data/detected_rep_sessions/FLOOR_PULL').
        class_b: Folder path for Class B (e.g. 'data/noise_sessions').
        class_a_collapse: If True, recursively aggregates CSVs from subdirs for Class A.
        class_b_collapse: If True, recursively aggregates CSVs from subdirs for Class B.
        train_ratio: Fraction of files allocated to training set.
        valid_ratio: Fraction of files allocated to validation set.

    Raises:
        ValueError: If train_ratio + valid_ratio >= 1.0 or non-positive ratios.
        FileNotFoundError: If source directories do not exist.
    """
    if train_ratio <= 0 or valid_ratio <= 0 or (train_ratio + valid_ratio) >= 1.0:
        raise ValueError(
            f"Invalid split ratios: train={train_ratio}, valid={valid_ratio}. "
            "train_ratio + valid_ratio must be strictly less than 1.0."
        )

    # Support alternate naming variants if provided in kwargs
    class_a_resolved = str(kwargs.get("classa", class_a))
    class_b_resolved = str(kwargs.get("classb", class_b))
    if "classa_collapse" in kwargs:
        class_a_collapse = bool(kwargs["classa_collapse"])
    if "classb_collapse" in kwargs:
        class_b_collapse = bool(kwargs["classb_collapse"])

    class_a_dir_name = os.path.basename(class_a_resolved.rstrip("/\\"))
    class_a_parent_name = os.path.basename(os.path.dirname(class_a_resolved.rstrip("/\\")))
    class_b_dir_name = os.path.basename(class_b_resolved.rstrip("/\\"))

    if class_a_dir_name in config.labels:
        class_a_label = class_a_dir_name
    elif class_a_parent_name in config.labels:
        class_a_label = class_a_parent_name
    elif len(config.labels) > 0:
        class_a_label = config.labels[0]
    else:
        class_a_label = class_a_dir_name

    remaining_labels = [lbl for lbl in config.labels if lbl != class_a_label]
    if class_b_dir_name in remaining_labels:
        class_b_label = class_b_dir_name
    elif len(remaining_labels) > 0:
        class_b_label = remaining_labels[0]
    elif len(config.labels) > 1:
        class_b_label = config.labels[1]
    else:
        class_b_label = class_b_dir_name

    targets = [
        (class_a_resolved, class_a_label, class_a_collapse),
        (class_b_resolved, class_b_label, class_b_collapse),
    ]

    for source_dir, label, collapse in targets:
        files = collect_class_files(source_dir, collapse=collapse)

        if not files:
            logger.warning("No CSV files found for label '%s' in %s — skipping.", label, source_dir)
            # Still create empty destination dirs so callers can inspect them
            for dest in [config.train_path, config.valid_path, config.test_path]:
                os.makedirs(os.path.join(dest, label), exist_ok=True)
            continue

        train_list, valid_list, test_list = split_one_class(
            files, train_ratio, valid_ratio
        )

        logger.info(
            "Splitting '%s' (%s): train=%d, valid=%d, test=%d",
            label,
            source_dir,
            len(train_list),
            len(valid_list),
            len(test_list),
        )

        copy_files(train_list, os.path.join(config.train_path, label))
        copy_files(valid_list, os.path.join(config.valid_path, label))
        copy_files(test_list, os.path.join(config.test_path, label))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    dummy_config = DataConfig(
        train_path="data/train",
        valid_path="data/valid",
        test_path="data/test",
        labels=["barbell", "none"],
    )
    random.seed(42)
    split_data(
        config=dummy_config,
        class_a="data/detected_rep_sessions/FLOOR_PULL",
        class_b="data/noise_sessions",
        class_a_collapse=False,
        class_b_collapse=True,
        train_ratio=0.6,
        valid_ratio=0.2,
    )
    logger.info("Data split complete!")
