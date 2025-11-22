# split_csv_data.py
"""
=====================================================================================
Split CSV sensor data into train / valid / test sets.

Expected input directory structure:

raw_data/
    onBarbell/
        *.csv
    notOnBarbell/
        *.csv

Output directory structure:

data/
    train/
        onBarbell/
        notOnBarbell/
    valid/
        onBarbell/
        notOnBarbell/
    test/
        onBarbell/
        notOnBarbell/
=====================================================================================
"""

import os
import random
import shutil

# MODIFY LABELS IF NEEDED
CLASSES = ["barbell", "none"]

def list_files(root_dir):
    """Load CSV files grouped by class labels."""
    data = {cls: [] for cls in CLASSES}

    for cls in CLASSES:
        class_dir = os.path.join(root_dir, cls)
        if not os.path.isdir(class_dir):
            raise ValueError(f"Missing folder: {class_dir}")

        for f in os.listdir(class_dir):
            if f.endswith(".csv"):
                data[cls].append(os.path.join(class_dir, f))

        print(f"{cls}: {len(data[cls])} files found")

    return data


def split_one_class(file_list, train_ratio, valid_ratio):
    """Split list into train, valid, and test."""
    random.shuffle(file_list)
    total = len(file_list)

    train_end = int(total * train_ratio)
    valid_end = int(total * (train_ratio + valid_ratio))

    train = file_list[:train_end]
    valid = file_list[train_end:valid_end]
    test  = file_list[valid_end:]

    return train, valid, test


def copy_files(file_list, dst_dir):
    os.makedirs(dst_dir, exist_ok=True)
    for f in file_list:
        shutil.copy(f, dst_dir)


def split_data(
    raw_root="raw_data",
    output_root="data",
    train_ratio=0.6,
    valid_ratio=0.2
):
    data = list_files(raw_root)

    # Output directories
    train_dir = os.path.join(output_root, "train")
    valid_dir = os.path.join(output_root, "valid")
    test_dir  = os.path.join(output_root, "test")

    for cls in CLASSES:
        train_list, valid_list, test_list = split_one_class(
            data[cls], train_ratio, valid_ratio
        )

        print(f"\nSplitting {cls}:")
        print(f"  Train: {len(train_list)}")
        print(f"  Valid: {len(valid_list)}")
        print(f"  Test : {len(test_list)}")

        # Copy into class folders
        copy_files(train_list, os.path.join(train_dir, cls))
        copy_files(valid_list, os.path.join(valid_dir, cls))
        copy_files(test_list, os.path.join(test_dir, cls))


if __name__ == "__main__":
    random.seed(42)  # reproducibility
    split_data(
        raw_root="./../raw_data",
        output_root="./../data",
        train_ratio=0.6,
        valid_ratio=0.2
    )

    print("\nDONE — Data split complete!")

