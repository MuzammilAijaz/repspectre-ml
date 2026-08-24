# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnknownArgumentType=false, reportMissingTypeStubs=false

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

    DATA_ROOT: Path = _PROJECT_ROOT / "data"
    DEFAULT_DATA_DIR: Path = DATA_ROOT / "raw_sessions" / "FLOOR_PULL"
    AXIS: str = "az"
    FS: int = 130

    def __init__(self, initial_data_dir: Path | None = None) -> None:
        super().__init__()
        self.current_data_dir: Path = initial_data_dir or self.DEFAULT_DATA_DIR
        self.csv_files: list[Path] = []
        self.sessions_count: int = 0
        self.refresh_csv_files()

    def scan_available_datasets(self) -> list[Path]:
        """Find all subdirectories under data/ that contain .csv files."""
        datasets: list[Path] = []
        if not self.DATA_ROOT.exists():
            return datasets

        for p in sorted(self.DATA_ROOT.rglob("*")):
            if p.is_dir():
                # Check if directory directly contains any .csv files
                csv_matches = list(p.glob("*.csv"))
                if csv_matches:
                    datasets.append(p)
        return datasets

    def set_data_dir(self, data_dir: Path) -> None:
        """Switch current data directory and reload available CSV files."""
        self.current_data_dir = data_dir
        self.refresh_csv_files()

    def refresh_csv_files(self) -> None:
        """Reload sorted CSV file list from current_data_dir."""
        if self.current_data_dir.exists() and self.current_data_dir.is_dir():
            def sort_key(p: Path) -> int | str:
                try:
                    return int(p.stem)
                except ValueError:
                    return p.stem

            self.csv_files = sorted(self.current_data_dir.glob("*.csv"), key=sort_key)
        else:
            self.csv_files = []
        self.sessions_count = len(self.csv_files)

    def load_session(self, idx: int) -> Session | None:
        """Load csv file, convert to dataframe and perform rep detection."""
        if not self.csv_files or idx < 0 or idx >= len(self.csv_files):
            return None

        path = self.csv_files[idx]
        df = pd.read_csv(path)

        detection: RepDetectionResult | None = detect_rep_axis(
            df,
            axis=self.AXIS,
            fs=self.FS,
            baseline_seconds=1.0,
            k_start=24.0,
            k_end=5.0,
            smooth_window=5,
            min_duration=0.12,
        )

        axis_data = cast("np.ndarray", np.asarray(df[self.AXIS], dtype=np.float64))

        return Session(
            path=path,
            axis_data=axis_data,
            detection=detection,
        )

    def get_sessions_count(self) -> int:
        return self.sessions_count



