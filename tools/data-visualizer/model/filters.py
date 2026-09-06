# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnknownArgumentType=false, reportMissingTypeStubs=false

import logging
from typing import Final

import numpy as np
from scipy import ndimage, signal

logger: Final = logging.getLogger("ui.panes.filter_selector_pane")


def apply_moving_average(data: np.ndarray, window_size: int) -> np.ndarray:
    """Uniform 1D moving average filter."""
    if window_size <= 1 or len(data) == 0:
        return data
    return ndimage.uniform_filter1d(data, size=window_size, mode="nearest")


def apply_butterworth_lowpass(
    data: np.ndarray, cutoff_hz: float, fs: float = 100.0, order: int = 2,
) -> np.ndarray:
    """Zero-phase Butterworth lowpass filter."""
    if len(data) == 0 or cutoff_hz <= 0:
        return data

    nyquist = 0.5 * fs
    normal_cutoff = min(cutoff_hz / nyquist, 0.99)

    sos = signal.butter(order, normal_cutoff,
            btype="low", output="sos",)

    return signal.sosfiltfilt(sos, data)


def apply_median_filter(data: np.ndarray, kernel_size: int) -> np.ndarray:
    """1D Median filter for removing impulse noise using scipy.signal."""
    if kernel_size <= 1 or len(data) == 0:
        return data
    # Ensure odd kernel size
    if kernel_size % 2 == 0:
        kernel_size += 1
    return signal.medfilt(data, kernel_size=kernel_size)


def apply_savgol_filter(
    data: np.ndarray, window_length: int, polyorder: int
) -> np.ndarray:
    """Savitzky-Golay polynomial smoothing filter using scipy.signal."""
    if len(data) == 0:
        return data
    # Ensure window_length is odd and greater than polyorder
    if window_length % 2 == 0:
        window_length += 1
    if window_length <= polyorder:
        window_length = polyorder + 2 if (polyorder + 2) % 2 != 0 else polyorder + 3
    if len(data) <= window_length:
        return data

    return np.asarray(
        signal.savgol_filter(data, window_length=window_length, polyorder=polyorder),
        dtype=np.float64,
    )

