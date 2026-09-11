# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
# Licensed under the Apache License, Version 2.0. See LICENSE for details.
#******************************************************************************
#  Data Loading and Preprocessing Module
# -----------------------------------------------------------------------------
#  Responsible for loading time-series CSV datasets from a directory structure,
#  mapping class labels, and preparing data for model training.
#
#  Main responsibilities:
#    - Load CSV files from class-based folder structure
#    - Select specified axes or slice columns dynamically up to data_dimension
#    - Convert raw sequences into numpy arrays
#    - Apply optional data augmentation to training set
#    - Pad sequences to fixed length
#    - Convert datasets into tf.data.Dataset format
#
#  Output:
#    - Train / validation / test datasets ready for TensorFlow models
#******************************************************************************

# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnknownArgumentType=false, reportMissingTypeStubs=false

import logging
import os
from typing import Any, Final

import numpy as np
import pandas as pd
import tensorflow as tf
from numpy.typing import NDArray

from lift_ml.config import DataConfig
from lift_ml.data.augment import augment_data

logger: Final = logging.getLogger(__name__)


class DataLoader:
    """Loads CSV data and prepares for training."""

    def __init__(self, config: DataConfig, augment_train: bool = True) -> None:
        self.config = config
        self.axes: list[str] | None = config.axes
        # If axes are explicitly specified in config, dimension is dynamically set to len(axes)
        self.dim = len(self.axes) if self.axes is not None else config.data_dimension

        self.seq_length: int = config.seq_length
        self.label2id: dict[str, int] = {label: i for i, label in enumerate(config.labels)}

        # Load CSV data
        self.train_data, self.train_label, self.train_len = self.load_csv_folder(config.train_path)
        self.valid_data, self.valid_label, self.valid_len = self.load_csv_folder(config.valid_path)
        self.test_data, self.test_label, self.test_len = self.load_csv_folder(config.test_path)

        # Augment training data if requested
        if augment_train:
            self.train_data, self.train_label = augment_data(self.train_data, self.train_label)
            self.train_len = len(self.train_label)
            logger.info("After augmentation, train_data_length: %d", self.train_len)

    def load_csv_folder(self, root_path: str) -> tuple[list[Any], list[str], int]:
        """Load CSV files from a folder with subfolders for each class.

        Expected Folder Structure:
        --------------------------
         root_path/
            class1/
                file1.csv
            class2/
                file2.csv
        """
        data: list[Any] = []
        labels: list[str] = []

        if not os.path.exists(root_path):
            raise FileNotFoundError(f"Dataset path does not exist: {root_path}")

        for label_name in sorted(os.listdir(root_path)):
            class_path = os.path.join(root_path, label_name)
            if not os.path.isdir(class_path):
                continue

            if label_name not in self.label2id:
                logger.warning(
                    "Skipping unknown class folder '%s' in %s (not in config labels: %s)",
                    label_name,
                    root_path,
                    self.config.labels,
                )
                continue

            csv_files = [f for f in sorted(os.listdir(class_path)) if f.endswith(".csv")]
            if not csv_files:
                raise ValueError(f"No CSV files found in class folder: {class_path}")

            for file in csv_files:
                file_path = os.path.join(class_path, file)
                df = pd.read_csv(file_path)

                if df.empty:
                    raise ValueError(f"Encountered empty CSV file: {file_path}")

                # Select columns based on explicit axes or slice up to self.dim
                if self.axes is not None:
                    missing_axes = [ax for ax in self.axes if ax not in df.columns]
                    if missing_axes:
                        raise ValueError(
                            f"CSV {file_path} is missing specified axes {missing_axes}. "
                            f"Available columns: {df.columns.tolist()}."
                        )
                    df_selected = df[self.axes]
                else:
                    if df.shape[1] < self.dim:
                        raise ValueError(
                            f"Invalid CSV format in {file_path}. Expected at least {self.dim} "
                            f"columns, got {df.shape[1]}."
                        )
                    df_selected = df.iloc[:, : self.dim]

                arr = df_selected.to_numpy(dtype=np.float64)
                if np.isnan(arr).any() or np.isinf(arr).any():
                    raise ValueError(f"CSV {file_path} contains NaN or Inf values.")

                data.append(arr)
                labels.append(label_name)

        if not labels:
            raise ValueError(
                f"No valid CSV samples found in {root_path} for classes {self.config.labels}."
            )

        logger.info("Loaded %d samples from %s", len(labels), root_path)
        return data, labels, len(labels)

    def pad(
        self, data: NDArray[np.float64], seq_length: int, dim: int
    ) -> list[NDArray[np.float64]]:
        noise_level = 20.0
        padded_data: list[NDArray[np.float64]] = []

        # Before-padding
        tmp = (np.random.rand(seq_length, dim) - 0.5) * noise_level + data[0]
        tmp[(seq_length - min(len(data), seq_length)) :] = data[: min(len(data), seq_length)]
        padded_data.append(tmp)

        # After-padding
        tmp = (np.random.rand(seq_length, dim) - 0.5) * noise_level + data[-1]
        tmp[: min(len(data), seq_length)] = data[: min(len(data), seq_length)]
        padded_data.append(tmp)

        return padded_data

    def _format_support_func(
        self, padded_num: int, length: int, data: list[Any], label: list[str]
    ) -> tuple[int, Any]:
        length *= padded_num
        features = np.zeros((length, self.seq_length, self.dim))
        labels = np.zeros(length)

        for idx, (d, lbl) in enumerate(zip(data, label, strict=False)):
            padded_data = self.pad(d, self.seq_length, self.dim)
            for num in range(padded_num):
                features[idx * padded_num + num] = padded_data[num]
                labels[idx * padded_num + num] = self.label2id[lbl]

        ds = tf.data.Dataset.from_tensor_slices((features, labels.astype("int32")))
        return length, ds

    def format(self) -> None:
        padded_num = 2
        self.train_len, self.train_data = self._format_support_func(
            padded_num, self.train_len, self.train_data, self.train_label
        )

        self.valid_len, self.valid_data = self._format_support_func(
            padded_num, self.valid_len, self.valid_data, self.valid_label
        )

        self.test_len, self.test_data = self._format_support_func(
            padded_num, self.test_len, self.test_data, self.test_label
        )

