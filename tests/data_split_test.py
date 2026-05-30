# Lint as: python3
# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import os
import shutil
import unittest
import pandas as pd
import numpy as np
from lift_ml.data.split import split_data
from lift_ml.config import DataConfig

class TestDataSplit(unittest.TestCase):

    def setUp(self):
        self.test_dir = "test_split_env"
        self.raw_root = os.path.join(self.test_dir, "raw_data")
        self.output_root = os.path.join(self.test_dir, "data")
        
        self.labels = ["barbell", "none"]
        
        # Create raw data
        for label in self.labels:
            label_dir = os.path.join(self.raw_root, label)
            os.makedirs(label_dir, exist_ok=True)
            for i in range(10):
                df = pd.DataFrame(np.random.rand(5, 2), columns=["ax", "ay"])
                df.to_csv(os.path.join(label_dir, f"file_{i}.csv"), index=False)

        self.config = DataConfig(
            train_path=os.path.join(self.output_root, "train"),
            valid_path=os.path.join(self.output_root, "valid"),
            test_path=os.path.join(self.output_root, "test"),
            labels=self.labels
        )

    def tearDown(self):
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)

    def test_split_data_distribution(self):
        """ 60% train, 20% valid, 20% test. """
        split_data(self.config, raw_root=self.raw_root, train_ratio=0.6, valid_ratio=0.2)

        for label in self.labels:
            train_files = os.listdir(os.path.join(self.config.train_path, label))
            valid_files = os.listdir(os.path.join(self.config.valid_path, label))
            test_files = os.listdir(os.path.join(self.config.test_path, label))

            self.assertEqual(len(train_files), 6)
            self.assertEqual(len(valid_files), 2)
            self.assertEqual(len(test_files), 2)

    def test_split_data_empty_class_results_in_zero_output_files(self):
        # Add an empty class folder
        empty_label = "empty"
        os.makedirs(os.path.join(self.raw_root, empty_label), exist_ok=True)
        
        # Update config to include empty label
        new_labels = self.labels + [empty_label]
        self.config.labels = new_labels

        split_data(self.config, raw_root=self.raw_root, train_ratio=0.6, valid_ratio=0.2)

        train_files = os.listdir(os.path.join(self.config.train_path, empty_label))
        self.assertEqual(len(train_files), 0)

    def test_split_data_missing_folder_does_not_crash_and_processes_existing_classes(self):
        # Label in config but missing in raw_root
        missing_label = "missing"
        self.config.labels = self.labels + [missing_label]

        # Should not crash, just print warning
        split_data(self.config, raw_root=self.raw_root, train_ratio=0.6, valid_ratio=0.2)
        
        # Check that it still split the other classes correctly
        for label in self.labels:
            train_files = os.listdir(os.path.join(self.config.train_path, label))
            self.assertEqual(len(train_files), 6)

if __name__ == "__main__":
    unittest.main()
