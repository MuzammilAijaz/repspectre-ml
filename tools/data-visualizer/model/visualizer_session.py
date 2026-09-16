import sys
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

# Ensure src/ is on the path for lift_ml imports
_PROJECT_ROOT = Path(__file__).resolve().parents[3]
_SRC_DIR = _PROJECT_ROOT / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from lift_ml.utils.rep_detection import RepDetectionResult  # noqa: E402


@dataclass
class VisualizerSession:
    """Container for visualizer session data and rep detection result."""

    path: Path
    sensor_data: pd.DataFrame
    detection: RepDetectionResult | None
    sampling_rate: float
    session_id: int | str
    modified_data: pd.DataFrame

