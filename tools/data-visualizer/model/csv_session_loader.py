# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnknownArgumentType=false, reportMissingTypeStubs=false

import logging
import sys
from pathlib import Path
from typing import Final, cast

import numpy as np
import pandas as pd

from model.visualizer_session import VisualizerSession

# Ensure src/ is on the path for lift_ml imports
_PROJECT_ROOT = Path(__file__).resolve().parents[3]
_SRC_DIR = _PROJECT_ROOT / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from lift_ml.utils.rep_detection import RepDetectionResult, detect_rep_axis  # noqa: E402
from lift_ml.utils.sampling import calculate_sampling_rate  # noqa: E402

logger: Final = logging.getLogger("model.loader")


class CsvSessionLoader:
    """Loads and navigates CSV session datasets for visualization and rapid exploration."""

    DATA_ROOT: Path = _PROJECT_ROOT / "data"
    DEFAULT_DATA_DIR: Path = DATA_ROOT / "rep_sessions" / "FLOOR_PULL"
    DEFAULT_AXIS: str = "ay"
    DEFAULT_FS: int = 130

    def __init__(
        self,
        initial_data_dir: Path | None = None,
        axis: str = DEFAULT_AXIS,
        default_fs: int = DEFAULT_FS,
    ) -> None:
        self.axis: str = axis
        self.default_fs: int = default_fs
        self.csv_files: list[Path] = []
        self.sessions_count: int = 0

        # Choose initial data directory
        if initial_data_dir is not None:
            self.current_data_dir: Path = initial_data_dir
        elif self.DEFAULT_DATA_DIR.exists():
            self.current_data_dir = self.DEFAULT_DATA_DIR
        else:
            # Fall back to first available dataset if default does not exist
            available = self.scan_available_datasets()
            self.current_data_dir = available[0] if available else self.DEFAULT_DATA_DIR

        logger.info("Initialized CsvSessionLoader with DATA_ROOT: %s", self.DATA_ROOT)
        logger.info("Active dataset directory: %s", self.current_data_dir)
        self.refresh_csv_files()

    def scan_available_datasets(self, data_root: Path | None = None) -> list[Path]:
        """Find all subdirectories under data/ that contain .csv files."""
        root = data_root or self.DATA_ROOT
        datasets: list[Path] = []
        if not root.exists():
            logger.warning("Data root does not exist: %s", root)
            return datasets

        for p in sorted(root.rglob("*")):
            if p.is_dir():
                csv_matches = list(p.glob("*.csv"))
                if csv_matches:
                    datasets.append(p)

        logger.info("Scanned %s: found %d dataset folders with CSVs", root, len(datasets))
        for d in datasets:
            try:
                rel = d.relative_to(root)
            except ValueError:
                rel = d
            logger.debug("  - %s (%d files)", rel, len(list(d.glob("*.csv"))))

        return datasets

    def set_data_dir(self, data_dir: Path) -> None:
        """Switch current data directory and reload available CSV files."""
        logger.info("Switching data directory to: %s", data_dir)
        self.current_data_dir = data_dir
        self.refresh_csv_files()

    def set_axis(self, axis: str) -> None:
        """Change the active plotting axis."""
        logger.info("Changed plotting axis to: %s", axis)
        self.axis = axis

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
        logger.info(
            "Refreshed CSV files for %s: found %d session files",
            self.current_data_dir.name,
            self.sessions_count,
        )

    def load_session(self, idx: int) -> VisualizerSession | None:
        """Load CSV file at idx, compute sampling frequency directly from DataFrame,
        and perform rep detection.
        """
        if not self.csv_files or idx < 0 or idx >= len(self.csv_files):
            logger.warning("Cannot load session at index %d (total: %d)", idx, len(self.csv_files))
            return None

        path = self.csv_files[idx]
        df = pd.read_csv(path)

        try:
            fs = calculate_sampling_rate(df)
        except ValueError as e:
            logger.debug(
                "Could not calculate sampling rate for %s (%s). Using default %d Hz",
                path.name,
                e,
                self.default_fs,
            )
            fs = float(self.default_fs)

        detection: RepDetectionResult | None = None
        if self.axis in df.columns:
            detection = detect_rep_axis(
                df,
                axis=self.axis,
                fs=int(round(fs)),
                baseline_seconds=1.0,
                k_start=24.0,
                k_end=5.0,
                smooth_window=5,
                min_duration=0.12,
            )
            axis_data = cast("np.ndarray", np.asarray(df[self.axis], dtype=np.float64))
        else:
            logger.warning(
                "Axis '%s' not found in %s (columns: %s)",
                self.axis,
                path.name,
                df.columns.tolist(),
            )
            axis_data = np.zeros(len(df), dtype=np.float64)

        try:
            session_id: int | str = int(path.stem)
        except ValueError:
            session_id = path.stem

        logger.info(
            "Loaded session [%d/%d] '%s' (%d rows, %.1f Hz, rep_detected=%s)",
            idx + 1,
            len(self.csv_files),
            path.name,
            len(df),
            fs,
            detection is not None,
        )

        return VisualizerSession(
            path=path,
            sensor_data=df,
            axis_data=axis_data,
            detection=detection,
            sampling_rate=fs,
            session_id=session_id,
        )

    def get_sessions_count(self) -> int:
        return self.sessions_count
