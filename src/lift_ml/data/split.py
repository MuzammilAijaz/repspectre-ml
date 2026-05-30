# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
# Licensed under the Apache License, Version 2.0. See LICENSE for details.
#******************************************************************************
#  Split raw CSV sensor data into train/validation/test sets.
# -----------------------------------------------------------------------------
#  This script organizes time-series CSV files stored in a class-based
#  directory structure and partitions them into reproducible dataset splits 
#  for machine learning.
#  
#      Input structure:              |           Output structure:
#      ---------------                           ----------------
#          raw_data/                 |               data/
#              class_a/              |                   train/
#                  *.csv             |                       class_a/
#              class_b/              |                       class_b/
#                  *.csv             |                   valid/
#                                    |                       class_a/
#                                    |                       class_b/
#                                    |                   test/
#                                    |                       class_a/
#                                    |                       class_b/
#
#  Key behavior:
#  ------------
#   - Reads all CSV files under each class folder defined in `DataConfig.labels`
#   - Randomly shuffles and splits files per class (not globally)
#   - Ensures class balance is preserved across splits
#   - Copies files into train/valid/test directories
#   - Creates missing output directories automatically
#******************************************************************************

import os
import random
import shutil
from typing import List, Dict
from lift_ml.config import DataConfig

def list_files(root_dir: str, classes: List[str]) -> Dict[str, List[str]]:
    """Load CSV files grouped by class labels."""
    data = {cls: [] for cls in classes}

    for cls in classes:
        class_dir = os.path.join(root_dir, cls)
        if not os.path.isdir(class_dir):
            print(f"Warning: Missing folder: {class_dir}")
            continue

        for f in os.listdir(class_dir):
            if f.endswith(".csv"):
                data[cls].append(os.path.join(class_dir, f))

        print(f"{cls}: {len(data[cls])} files found")

    return data


def split_one_class(file_list: List[str], train_ratio: float, valid_ratio: float):
    """Split list into train, valid, and test."""
    random.shuffle(file_list)
    total = len(file_list)

    train_end = int(total * train_ratio)
    valid_end = int(total * (train_ratio + valid_ratio))

    train = file_list[:train_end]
    valid = file_list[train_end:valid_end]
    test  = file_list[valid_end:]

    return train, valid, test


def copy_files(file_list: List[str], dst_dir: str):
    os.makedirs(dst_dir, exist_ok=True)
    for f in file_list:
        shutil.copy(f, dst_dir)


def split_data(
    config: DataConfig,
    raw_root: str = "raw_data",
    train_ratio: float = 0.6,
    valid_ratio: float = 0.2
):
    """
    Splits data from raw_root into train, valid, and test directories defined in config.
    """
    data = list_files(raw_root, config.labels)

    for cls in config.labels:
        train_list, valid_list, test_list = split_one_class(
            data[cls], train_ratio, valid_ratio
        )

        print(f"\nSplitting {cls}:")
        print(f"  Train: {len(train_list)}")
        print(f"  Valid: {len(valid_list)}")
        print(f"  Test : {len(test_list)}")

        # Copy into class folders as specified in config
        copy_files(train_list, os.path.join(config.train_path, cls))
        copy_files(valid_list, os.path.join(config.valid_path, cls))
        copy_files(test_list, os.path.join(config.test_path, cls))


if __name__ == "__main__":
    # Example usage (would typically be called from a higher level script or CLI)
    from lift_ml.config import Config
    # Assuming a default config exists or providing a dummy one
    dummy_config = DataConfig(
        train_path="data/train",
        valid_path="data/valid",
        test_path="data/test",
        labels=["barbell", "none"]
    )
    random.seed(42)  # reproducibility
    split_data(
        config=dummy_config,
        raw_root="raw_data",
        train_ratio=0.6,
        valid_ratio=0.2
    )

    print("\nDONE — Data split complete!")
