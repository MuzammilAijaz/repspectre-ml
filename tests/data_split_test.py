# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnknownArgumentType=false, reportMissingTypeStubs=false

import os
import shutil
import unittest

import numpy as np
import pandas as pd

from lift_ml.config import DataConfig
from lift_ml.data.split import collect_class_files, split_data


class TestDataSplit(unittest.TestCase):
    def setUp(self) -> None:
        self.test_dir = "test_split_env"
        self.output_root = os.path.join(self.test_dir, "data")

        # Class A: Single lift directory with flat CSVs (e.g. detected_rep_sessions/FLOOR_PULL)
        self.class_a_dir = os.path.join(
            self.test_dir, "source", "detected_rep_sessions", "FLOOR_PULL"
        )
        os.makedirs(self.class_a_dir, exist_ok=True)
        for i in range(10):
            df = pd.DataFrame(np.random.rand(5, 2), columns=["ax", "ay"])
            df.to_csv(os.path.join(self.class_a_dir, f"rep_{i}.csv"), index=False)

        # Class B: Noise directory with multiple subcategories (e.g. BARBELL_IMPACT, SENSOR_DRIFT)
        self.class_b_dir = os.path.join(self.test_dir, "source", "noise_sessions")
        for subcat in ["BARBELL_IMPACT", "SENSOR_DRIFT"]:
            subcat_dir = os.path.join(self.class_b_dir, subcat)
            os.makedirs(subcat_dir, exist_ok=True)
            for i in range(5):
                df = pd.DataFrame(np.random.rand(5, 2), columns=["ax", "ay"])
                df.to_csv(os.path.join(subcat_dir, f"{subcat}_{i}.csv"), index=False)

        self.labels = ["barbell", "none"]
        self.config = DataConfig(
            train_path=os.path.join(self.output_root, "train"),
            valid_path=os.path.join(self.output_root, "valid"),
            test_path=os.path.join(self.output_root, "test"),
            labels=self.labels,
        )

    def tearDown(self) -> None:
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)

    def test_collect_class_files_with_and_without_collapse(self) -> None:
        # Class A (flat): collapse=False gets all 10 files
        files_a = collect_class_files(self.class_a_dir, collapse=False)
        self.assertEqual(len(files_a), 10)

        # Class B (nested subdirectories):
        # collapse=False on parent gets 0 files (since they are in subdirs)
        files_b_flat = collect_class_files(self.class_b_dir, collapse=False)
        self.assertEqual(len(files_b_flat), 0)

        # collapse=True collects all 10 files from both subdirs
        files_b_collapsed = collect_class_files(self.class_b_dir, collapse=True)
        self.assertEqual(len(files_b_collapsed), 10)

    def test_split_data_distribution_with_collapsed_noise(self) -> None:
        """60% train (6), 20% valid (2), 20% test (2) for both classes."""
        split_data(
            config=self.config,
            class_a=self.class_a_dir,
            class_b=self.class_b_dir,
            class_a_collapse=False,
            class_b_collapse=True,
            train_ratio=0.6,
            valid_ratio=0.2,
        )

        for label in self.labels:
            train_files = os.listdir(os.path.join(self.config.train_path, label))
            valid_files = os.listdir(os.path.join(self.config.valid_path, label))
            test_files = os.listdir(os.path.join(self.config.test_path, label))

            self.assertEqual(len(train_files), 6)
            self.assertEqual(len(valid_files), 2)
            self.assertEqual(len(test_files), 2)

    def test_split_data_empty_or_missing_folder_handles_gracefully(self) -> None:
        empty_dir = os.path.join(self.test_dir, "empty_dir")
        os.makedirs(empty_dir, exist_ok=True)

        split_data(
            config=self.config,
            class_a=self.class_a_dir,
            class_b=empty_dir,
            class_a_collapse=False,
            class_b_collapse=True,
            train_ratio=0.6,
            valid_ratio=0.2,
        )

        # Class A has 6 train files
        train_a = os.listdir(os.path.join(self.config.train_path, self.labels[0]))
        self.assertEqual(len(train_a), 6)

        # Class B has 0 train files
        train_b = os.listdir(os.path.join(self.config.train_path, self.labels[1]))
        self.assertEqual(len(train_b), 0)


if __name__ == "__main__":
    unittest.main()
