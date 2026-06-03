import numpy as np
import pandas as pd

def mad(x):
    """Median absolute deviation."""
    return np.median(np.abs(x - np.median(x)))

def detect_rep_axis(df, axis='ay', fs=130,
                    baseline_seconds=1.0,
                    k_start=4.0,   # multiplier for start threshold
                    k_end=2.0,     # multiplier for end threshold
                    smooth_window=5,
                    min_duration=0.15
                   ):
    """Detect start and end index (samples) of first rep on a chosen axis.
       Also returns model input start and end indices based on a pre/post window.
    """
    ys = df[axis].values.astype(np.float64)

    # 1) smooth
    if smooth_window > 1:
        ys_s = pd.Series(ys).rolling(smooth_window, center=True, min_periods=1).mean().values
    else:
        ys_s = ys

    # 2) baseline
    baseline_count = int(max(1, baseline_seconds * fs))
    baseline_segment = ys_s[:baseline_count]

    base_med = np.median(baseline_segment)
    noise_mad = mad(baseline_segment) + 1e-12

    # 3) thresholds
    dist = np.abs(ys_s - base_med)
    start_th = k_start * noise_mad
    end_th = k_end * noise_mad

    # 4) detect start
    min_samples = int(np.ceil(min_duration * fs))
    above_start = dist > start_th

    start_idx = None
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
        return None, None, ys_s, base_med, start_th, end_th, None, None

    # 5) detect end
    below_end = dist < end_th
    end_idx = None
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

    return start_idx, end_idx, ys_s, base_med, start_th, end_th, model_start_idx, model_end_idx
