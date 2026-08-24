import sys
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import numpy as np
import pandas as pd

# Ensure src/ is on the path for lift_ml imports
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
_SRC_DIR = _PROJECT_ROOT / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from lift_ml.utils.rep_detection import RepDetectionResult, detect_rep_axis  # noqa: E402


@dataclass
class Session:
    path: Path
    axis_data: np.ndarray
    detection: RepDetectionResult | None


class SessionRepository:

    # WARN: hardcoded session type
    _BASE_DIR: Path = _PROJECT_ROOT / "data/raw_sessions"
    DATA_DIR: Path  = _BASE_DIR / "FLOOR_PULL"

    def __init__(self) -> None:
        super().__init__()

        self.csv_files: list[Path] = sorted(
            self.DATA_DIR.glob("*.csv"), key=lambda p: int(p.stem)
        )
        self.sessions_count = len(self.csv_files)

    def load_session(self, idx: int) -> Session:
        """Load csv file, convert to dataframe and perform rep detection."""
        path = self.csv_files[idx]
        df   = pd.read_csv(path)

        detection: RepDetectionResult | None = detect_rep_axis(
            df,
            axis="az",
            fs=130,
            baseline_seconds=1.0,
            k_start=24.0,
            k_end=5.0,
            smooth_window=5,
            min_duration=0.12,
        )

        axis_data = cast("np.ndarray", np.asarray(df["az"], dtype=np.float64))

        return Session(
            path=path,
            axis_data=axis_data,
            detection=detection,
        )

    def get_sessions_count(self) -> int:
        return self.sessions_count


