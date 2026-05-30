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
#    - Convert raw sequences into numpy arrays
#    - Apply optional data augmentation to training set
#    - Pad sequences to fixed length
#    - Convert datasets into tf.data.Dataset format
#  
#  Output:
#    - Train / validation / test datasets ready for TensorFlow models
#******************************************************************************

import os
import numpy as np
import pandas as pd
import tensorflow as tf

from lift_ml.data.augment import augment_data
from lift_ml.config import DataConfig

class DataLoader(object):
    """Loads CSV data and prepares for training."""

    def __init__(self, config: DataConfig, augment_train=True):
        self.config = config
        self.dim = config.data_dimension
        self.seq_length = config.seq_length
        self.label2id = {label: i for i, label in enumerate(config.labels)}

        # Load CSV data
        self.train_data, self.train_label, self.train_len = self.load_csv_folder(config.train_path)
        self.valid_data, self.valid_label, self.valid_len = self.load_csv_folder(config.valid_path)
        self.test_data,  self.test_label,  self.test_len  = self.load_csv_folder(config.test_path)

        # Augment training data if requested
        if augment_train:
            self.train_data, self.train_label = augment_data(self.train_data, self.train_label)
            self.train_len = len(self.train_label)
            print(f"After augmentation, train_data_length: {self.train_len}")

    def load_csv_folder(self, root_path):
        """ 
        Load CSV files from a folder with subfolders for each class.

        Expected Folder Structure:
        --------------------------
         root_path/
            class1/
                file1.csv
            class2/
                file2.csv
        """

        data = []
        labels = []

        if not os.path.exists(root_path):
            print(f"Warning: Path {root_path} does not exist.")
            return data, labels, 0

        for label_name in os.listdir(root_path):
            class_path = os.path.join(root_path, label_name)
            if not os.path.isdir(class_path):
                continue
            
            if label_name not in self.label2id:
                print(f"Warning: Label {label_name} in {root_path} not found in config labels {self.config.labels}. Skipping.")
                continue

            for file in os.listdir(class_path):
                if file.endswith(".csv"):
                    file_path = os.path.join(class_path, file)
                    df = pd.read_csv(file_path)

                    # Expecting columns matching config.data_dimension
                    if df.shape[1] != self.dim:
                        raise ValueError(f"Invalid CSV format in {file_path}. Expected {self.dim} columns, got {df.shape[1]}.")

                    # Convert to numpy [seq_len, dim]
                    data.append(df.values)
                    labels.append(label_name)

        print(f"Loaded {len(labels)} samples from {root_path}")
        return data, labels, len(labels)

    def pad(self, data, seq_length, dim):
        noise_level = 20
        padded_data = []

        # Before-padding
        tmp = (np.random.rand(seq_length, dim) - 0.5) * noise_level + data[0]
        tmp[(seq_length - min(len(data), seq_length)):] = data[:min(len(data), seq_length)]
        padded_data.append(tmp)

        # After-padding
        tmp = (np.random.rand(seq_length, dim) - 0.5) * noise_level + data[-1]
        tmp[:min(len(data), seq_length)] = data[:min(len(data), seq_length)]
        padded_data.append(tmp)

        return padded_data

    def _format_support_func(self, padded_num, length, data, label):
        length *= padded_num
        features = np.zeros((length, self.seq_length, self.dim))
        labels = np.zeros(length)

        for idx, (d, l) in enumerate(zip(data, label)):
            padded_data = self.pad(d, self.seq_length, self.dim)
            for num in range(padded_num):
                features[idx * padded_num + num] = padded_data[num]
                labels[idx * padded_num + num] = self.label2id[l]

        ds = tf.data.Dataset.from_tensor_slices((features, labels.astype("int32")))
        return length, ds

    def format(self):
        padded_num = 2
        self.train_len, self.train_data = self._format_support_func(
            padded_num, self.train_len, self.train_data, self.train_label)

        self.valid_len, self.valid_data = self._format_support_func(
            padded_num, self.valid_len, self.valid_data, self.valid_label)

        self.test_len, self.test_data = self._format_support_func(
            padded_num, self.test_len, self.test_data, self.test_label)
