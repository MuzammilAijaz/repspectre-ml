from dataclasses import dataclass
from typing import cast

import numpy as np
import pandas as pd
from numpy.typing import NDArray


@dataclass
class RepDetectionResult:
    start_idx: int | None
    end_idx: int | None
    signal: NDArray[np.float64]
    baseline: float
    start_threshold: float
    end_threshold: float
    model_start_idx: int | None
    model_end_idx: int | None

def mad(x: NDArray[np.float64]) -> np.float64:
    """Median absolute deviation."""
    med_val: float = float(np.median(x))
    return np.float64(np.median(np.abs(x - med_val)))

def detect_rep_axis(
    df: pd.DataFrame,
    axis: str = "ay",
    fs: int = 130,
    baseline_seconds: float = 1.0,
    k_start: float = 4.0,   # multiplier for start threshold
    k_end: float = 2.0,     # multiplier for end threshold
    smooth_window: int = 5,
    min_duration: float = 0.15,
) -> RepDetectionResult:
    """
    Detect start and end index (samples) of first rep on a chosen axis.
    Also returns model input start and end indices based on a pre/post window.
    """
    series = cast("pd.Series", df[axis])
    ys: NDArray[np.float64] = np.asarray(series, dtype=np.float64)

    # 1) smooth
    if smooth_window > 1:
        s = pd.Series(ys).rolling(smooth_window, center=True, min_periods=1).mean()  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]
        ys_s: NDArray[np.float64] = np.asarray(s, dtype=np.float64)
    else:
        ys_s = ys

    # 2) baseline
    baseline_count = int(max(1, baseline_seconds * fs))
    baseline_segment: NDArray[np.float64] = ys_s[:baseline_count]

    base_med: float = float(np.median(baseline_segment))
    noise_mad: float = float(mad(baseline_segment)) + 1e-12

    # 3) thresholds
    dist: NDArray[np.float64] = np.abs(ys_s - base_med)
    start_th: float = k_start * noise_mad
    end_th: float = k_end * noise_mad

    # 4) detect start
    min_samples = int(np.ceil(min_duration * fs))
    above_start: NDArray[np.bool_] = dist > start_th

    start_idx: int | None = None
    for i in range(len(above_start)):
        if not above_start[i]:
            continue
        end_check_idx = i + min_samples
        if end_check_idx > len(dist):
            break
        window = dist[i:end_check_idx]
        if np.any(window > end_th):
            start_idx = i
            break

    if start_idx is None:
        return RepDetectionResult(
            start_idx=None,
            end_idx=None,
            signal=ys_s,
            baseline=base_med,
            start_threshold=start_th,
            end_threshold=end_th,
            model_start_idx=None,
            model_end_idx=None,
        )

    # 5) detect end
    below_end: NDArray[np.bool_] = dist < end_th
    end_idx: int | None = None
    for j in range(start_idx + 1, len(below_end)):
        if below_end[j]:
            j_end = j + min_samples
            if j_end > len(below_end):
                end_idx = len(below_end) - 1
                break
            if np.all(below_end[j:j_end]):
                end_idx = j
                break
    if end_idx is None:
        end_idx = len(dist) - 1

    # 6) model input window
    pre_sec = 0.3
    post_sec = 0.5
    model_start_idx = max(0, start_idx - int(pre_sec * fs))
    model_end_idx   = min(len(ys_s) - 1, start_idx + int(post_sec * fs))

    return RepDetectionResult(
        start_idx=start_idx,
        end_idx=end_idx,
        signal=ys_s,
        baseline=base_med,
        start_threshold=start_th,
        end_threshold=end_th,
        model_start_idx=model_start_idx,
        model_end_idx=model_end_idx,
    )

