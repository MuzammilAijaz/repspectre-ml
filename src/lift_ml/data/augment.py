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

# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnknownArgumentType=false, reportArgumentType=false

import random
from typing import Any

import numpy as np


def _time_wrapping(
    molecule: int, denominator: int, data: list[list[float]]
) -> list[list[float]] | None:
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
    assert molecule > 0
    assert denominator >= 0

    # creates 2D array with 0s
    rows = (int(len(data) / molecule) - 1) * denominator

    # if the sequence is not long enough to time warp
    if rows <= 0:
        return None

    columns = len(data[0])
    tmp_data: list[list[float]] = [[0.0 for _ in range(columns)] for _ in range(rows)]

    # apply time warp
    for i in range(int(len(data) / molecule) - 1):
        for j in range(len(data[i])):
            for k in range(denominator):
                val = (
                    data[molecule * i + k][j] * (denominator - k)
                    + data[molecule * i + k + 1][j] * k
                ) / denominator
                tmp_data[denominator * i + k][j] = float(val)
    return tmp_data


def augment_data(
    original_data: list[Any], original_label: list[Any]
) -> tuple[list[Any], list[Any]]:
    """Perform data augmentation."""
    new_data: list[Any] = []
    new_label: list[Any] = []

    # ------------------ Go over all points ------------------

    for _idx, (data, label) in enumerate(zip(original_data, original_label, strict=False)):
        # Original data
        new_data.append(data)
        new_label.append(label)

        # Sequence shift - Add random global offset to simulate sensor bias
        # or motion baseline shifts
        for _num in range(5):
            shifted = (np.array(data, dtype=np.float32) + (random.random() - 0.5) * 200).tolist()
            new_data.append(shifted)
            new_label.append(label)

        # Random noise - to simulate natural movements which are often different
        for _num in range(5):
            num_cols = len(data[0])
            num_rows = len(data)
            tmp_data: list[list[float]] = [[0.0 for _ in range(num_cols)] for _ in range(num_rows)]
            for i in range(len(tmp_data)):
                for j in range(len(tmp_data[i])):
                    tmp_data[i][j] = float(data[i][j]) + 5.0 * random.random()
            new_data.append(tmp_data)
            new_label.append(label)

        # Time warping - to take into account the speed of the motion.
        fractions: list[tuple[int, int]] = [(3, 2), (5, 3), (2, 3), (3, 4), (9, 5), (6, 5), (4, 5)]
        for molecule, denominator in fractions:
            warped = _time_wrapping(molecule, denominator, data)
            if warped is not None:
                new_data.append(warped)
                new_label.append(label)

        # Movement amplification - to take into account the degree of motion
        for molecule, denominator in fractions:
            amplified = (np.array(data, dtype=np.float32) * molecule / denominator).tolist()
            new_data.append(amplified)
            new_label.append(label)

    return new_data, new_label


