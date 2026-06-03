# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
# Licensed under the Apache License, Version 2.0. See LICENSE for details.
#******************************************************************************
#  Data Augmentation Module
# -----------------------------------------------------------------------------
#  Applies augmentation techniques to motion sequence data to improve model
#  generalization during training.
# 
#  Techniques included:
#    - Sequence shift (global offset)
#    - Random noise injection
#    - Time warping (speed variation)
#    - Movement amplification
# 
#  Used during dataset preparation to expand training samples. @see data_load.py
#******************************************************************************

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import random
import numpy as np


def _time_wrapping(molecule, denominator, data):
  """Generate (molecule/denominator)x speed data.

  where molecule and denominator define the speed ratio.

  If molecule > denominator → the sequence gets faster (fewer steps = compression).
  If molecule < denominator → the sequence gets slower (more steps = stretching).

  Return
  ------
   - Time warped data with same column size but different row sizes 
   based on molecule/denominator.
   - Returns None if sequence is too short.
  """
  assert(molecule > 0)
  assert(denominator >= 0)

  # creates 2D array with 0s
  rows = (int(len(data) / molecule) - 1) * denominator
  
  # if the sequence is not long enough to time warp
  if (rows <= 0):
    return None

  columns = len(data[0])
  tmp_data = [[0 for i in range(columns)] for j in range(rows)]

  # apply time warp 
  for i in range(int(len(data) / molecule) - 1):
    for j in range(len(data[i])):
      for k in range(denominator):
        tmp_data[denominator * i +
                 k][j] = (data[molecule * i + k][j] * (denominator - k) +
                          data[molecule * i + k + 1][j] * k) / denominator
  return tmp_data


def augment_data(original_data, original_label):
  """Perform data augmentation."""
  new_data = []
  new_label = []

  # ------------------ Go over all points ------------------ 

  for idx, (data, label) in enumerate(zip(original_data, original_label)):  # pylint: disable=unused-variable
    # Original data
    new_data.append(data)
    new_label.append(label)

    # Sequence shift - Add random global offset to simulate sensor bias or motion baseline shifts
    for num in range(5):  # pylint: disable=unused-variable
      new_data.append((np.array(data, dtype=np.float32) +
                       (random.random() - 0.5) * 200).tolist())
      new_label.append(label)

    # Random noise - to simulate natural movements which are often different
    for num in range(5):
      tmp_data = [[0 for i in range(len(data[0]))] for j in range(len(data))]
      for i in range(len(tmp_data)):
        for j in range(len(tmp_data[i])):
          tmp_data[i][j] = data[i][j] + 5 * random.random()
      new_data.append(tmp_data)
      new_label.append(label)

    # Time warping - to take into account the speed of the motion.
    fractions = [(3, 2), (5, 3), (2, 3), (3, 4), (9, 5), (6, 5), (4, 5)]
    for molecule, denominator in fractions:
      warped = _time_wrapping(molecule, denominator, data)
      if warped is not None:
        new_data.append(warped)
        new_label.append(label)

    # Movement amplification - to take into account the degree of motion 
    for molecule, denominator in fractions:
      new_data.append(
          (np.array(data, dtype=np.float32) * molecule / denominator).tolist())
      new_label.append(label)

  return new_data, new_label

