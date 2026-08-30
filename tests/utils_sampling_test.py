# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnknownArgumentType=false, reportMissingTypeStubs=false, reportAttributeAccessIssue=false

import unittest

import numpy as np
import pandas as pd

from lift_ml.utils.sampling import calculate_sampling_rate


class TestSamplingRateCalculation(unittest.TestCase):
    def test_calculate_sampling_rate_from_dataframe(self) -> None:
        # 101 samples spanning 1,000,000 us (1.0 sec) -> exact 100.0 Hz
        timestamps = np.linspace(0, 1_000_000, 101, dtype=np.int64)
        df = pd.DataFrame({"ax": np.zeros(101), "timestampUs": timestamps})
        fs = calculate_sampling_rate(df)
        self.assertAlmostEqual(fs, 100.0, places=2)

    def test_calculate_sampling_rate_with_custom_column_name(self) -> None:
        timestamps = np.linspace(0, 2_000_000, 201, dtype=np.int64)
        df = pd.DataFrame({"ax": np.zeros(201), "custom_ts": timestamps})
        fs = calculate_sampling_rate(df, timestamp_col="custom_ts")
        self.assertAlmostEqual(fs, 100.0, places=2)

    def test_calculate_sampling_rate_raises_on_missing_column(self) -> None:
        df = pd.DataFrame({"ax": np.zeros(50)})
        with self.assertRaises(ValueError) as ctx:
            calculate_sampling_rate(df)
        self.assertIn("lacks 'timestampUs' column", str(ctx.exception))

    def test_calculate_sampling_rate_raises_on_non_positive_duration(self) -> None:
        df = pd.DataFrame({
            "ax": np.zeros(5),
            "timestampUs": [1000, 1000, 1000, 1000, 1000],
        })
        with self.assertRaises(ValueError) as ctx:
            calculate_sampling_rate(df)
        self.assertIn("Invalid duration", str(ctx.exception))

    def test_calculate_sampling_rate_raises_on_fewer_than_two_samples(self) -> None:
        df = pd.DataFrame({"ax": [1.0], "timestampUs": [1000]})
        with self.assertRaises(ValueError) as ctx:
            calculate_sampling_rate(df)
        self.assertIn("fewer than 2 samples", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
