"""
=====================================================================================
|                                  CLASS FILE 
-------------------------------------------------------------------------------------
Loads CSV data and prepares training/validation/testing datasets for the model.
=====================================================================================
"""

import os
import numpy as np
import pandas as pd
import tensorflow as tf

from data_augmentation import augment_data

# --------------------------------------------
# LABELS (Modify if needed)
# --------------------------------------------
label1 = "barbell"
label2 = "none"
DIMENSION = 2 


class DataLoader(object):
    """Loads CSV data and prepares for training."""

    def __init__(self, train_path, valid_path, test_path, seq_length, augment_train=True):
        self.dim = DIMENSION
        self.seq_length = seq_length
        self.label2id = {label1: 0, label2: 1}

        # ----------------------------
        # Load CSV data
        # ----------------------------
        self.train_data, self.train_label, self.train_len = self.load_csv_folder(train_path)
        self.valid_data, self.valid_label, self.valid_len = self.load_csv_folder(valid_path)
        self.test_data,  self.test_label,  self.test_len  = self.load_csv_folder(test_path)

        # ----------------------------
        # Augment training data if requested
        # ----------------------------
        if augment_train:
            self.train_data, self.train_label = augment_data(self.train_data, self.train_label)
            self.train_len = len(self.train_label)
            print(f"After augmentation, train_data_length: {self.train_len}")

    # ------------------------------------------------------------
    # Load CSV files from a folder with subfolders for each class
    # ------------------------------------------------------------
    def load_csv_folder(self, root_path):
        data = []
        labels = []

        # Expected structure:
        # root_path/
        #    onBarbell/
        #        file1.csv
        #    notOnBarbell/
        #        file2.csv
        #
        for label_name in os.listdir(root_path):
            class_path = os.path.join(root_path, label_name)
            if not os.path.isdir(class_path):
                continue

            for file in os.listdir(class_path):
                if file.endswith(".csv"):
                    file_path = os.path.join(class_path, file)
                    df = pd.read_csv(file_path)

                    # Expecting columns: ax, ay, az, gx, gy, gz
                    if df.shape[1] != DIMENSION:
                        raise ValueError(f"Invalid CSV format in {file_path}")

                    # Convert to numpy [seq_len, 6]
                    data.append(df.values)
                    labels.append(label_name)

        print(f"Loaded {len(labels)} samples from {root_path}")
        return data, labels, len(labels)

    # ------------------------------------------------------------
    # Padding (same as original)
    # ------------------------------------------------------------
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

    # ------------------------------------------------------------
    # Shared padding + dataset creation
    # ------------------------------------------------------------
    def format_support_func(self, padded_num, length, data, label):
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

    # ------------------------------------------------------------
    # Format datasets (same as original)
    # ------------------------------------------------------------
    def format(self):
        padded_num = 2
        self.train_len, self.train_data = self.format_support_func(
            padded_num, self.train_len, self.train_data, self.train_label)

        self.valid_len, self.valid_data = self.format_support_func(
            padded_num, self.valid_len, self.valid_data, self.valid_label)

        self.test_len, self.test_data = self.format_support_func(
            padded_num, self.test_len, self.test_data, self.test_label)

