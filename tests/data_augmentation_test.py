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
# pylint: disable=g-bad-import-order

"""Test for data_augmentation.py."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import unittest

import numpy as np

from lift_ml.data.augment import augment_data
from lift_ml.data.augment import _time_wrapping

class TestAugmentation(unittest.TestCase):
    # The number of samples generated per original sample when all augmentations succeed:
    # 1 (original) + 5 (shift) + 5 (noise) + 7 (time warp) + 7 (amplification) = 25
    FULL_AUGMENTATION_COUNT = 25

    def test_time_wrapping_preserves_feature_dimensions(self):
        original_data = np.random.rand(10, 3).tolist()
        wrapped_data = _time_wrapping(4, 5, original_data)
        self.assertEqual(len(wrapped_data), int(len(original_data) / 4 - 1) * 5)
        self.assertEqual(len(wrapped_data[0]), len(original_data[0]))

    def test_time_wrapping_returns_none_when_sequence_too_short(self):
        data = np.random.rand(3, 20).tolist()
        self.assertIsNone(_time_wrapping(4, 5, data))

    def test_augment_data_returns_expected_number_of_samples(self):
        original_data = [
                np.random.rand(128, 3).tolist(),
                np.random.rand(66, 2).tolist(),
                np.random.rand(50, 1).tolist()
        ]
        original_label = ["data", "augmentation", "test"]
        augmented_data, augmented_label = augment_data(original_data,
                                                       original_label)
        self.assertEqual(self.FULL_AUGMENTATION_COUNT * len(original_data), len(augmented_data))
        self.assertIsInstance(augmented_data, list)
        self.assertEqual(self.FULL_AUGMENTATION_COUNT * len(original_label), len(augmented_label))
        self.assertIsInstance(augmented_label, list)
        for i in range(len(original_label)):
            self.assertEqual(augmented_label[self.FULL_AUGMENTATION_COUNT * i], original_label[i])

    def test_augment_data_short_sequence_returns_fewer_than_full_count(self):
        # Length 3 is too short for all time warping fractions in augment_data
        original_data = [np.random.rand(3, 3).tolist()]
        original_label = ["short"]
        augmented_data, _ = augment_data(original_data, original_label)
        # Expected: 1 (original) + 5 (shift) + 5 (noise) + 0 (warp) + 7 (amp) = 18
        self.assertLess(len(augmented_data), self.FULL_AUGMENTATION_COUNT)
        self.assertNotEqual(len(augmented_data) % self.FULL_AUGMENTATION_COUNT, 0)

    def test_augment_data_mixed_lengths_is_not_multiple_of_full_count(self):
        original_data = [
            np.random.rand(100, 3).tolist(), # Should give FULL_AUGMENTATION_COUNT
            np.random.rand(5, 3).tolist()    # Should give < FULL_AUGMENTATION_COUNT
        ]
        original_label = ["long", "short"]
        augmented_data, _ = augment_data(original_data, original_label)
        expected_full_total = self.FULL_AUGMENTATION_COUNT * len(original_data)
        self.assertNotEqual(len(augmented_data), expected_full_total)
        self.assertNotEqual(len(augmented_data) % self.FULL_AUGMENTATION_COUNT, 0)

    # TODO: test augmentation of data

if __name__ == "__main__":
    unittest.main()
