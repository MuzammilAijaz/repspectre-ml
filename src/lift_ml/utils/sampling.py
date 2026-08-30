# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnknownArgumentType=false, reportMissingTypeStubs=false

import pandas as pd


def calculate_sampling_rate(
    df: pd.DataFrame,
    timestamp_col: str = "timestampUs",
) -> float:
    """Calculates the exact sampling rate (Hz) from microsecond timestamps in a DataFrame.

    Parameters:
        df: DataFrame containing sensor data and timestamp column.
        timestamp_col: Column name containing microsecond timestamps (default: 'timestampUs').

    Returns:
        Exact sampling rate in Hertz (Hz).

    Raises:
        ValueError: If df has fewer than 2 rows, timestamp column is missing,
                    or timestamps yield non-positive duration.
    """
    if len(df) < 2:
        raise ValueError(
            f"DataFrame has fewer than 2 samples ({len(df)} samples). "
            "Cannot calculate sampling rate."
        )

    if timestamp_col not in df.columns:
        raise ValueError(
            f"DataFrame lacks '{timestamp_col}' column required to calculate sampling rate."
        )

    start_us = float(df[timestamp_col].iloc[0])
    end_us = float(df[timestamp_col].iloc[-1])
    duration_s = (end_us - start_us) / 1_000_000.0

    if duration_s <= 0:
        raise ValueError(
            f"Invalid duration from '{timestamp_col}': start={start_us} us, "
            f"end={end_us} us (duration={duration_s} s). Timestamps must be monotonic."
        )

    fs = (len(df) - 1) / duration_s
    if fs <= 0:
        raise ValueError(f"Calculated non-positive sampling rate: {fs} Hz.")

    return fs
