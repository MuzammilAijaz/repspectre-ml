#*****************************************************************************
# Interactive sensor-session viewer for visualizing signals and rep detection
#*****************************************************************************

# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnknownArgumentType=false, reportMissingTypeStubs=false, reportAttributeAccessIssue=false

import sys
from pathlib import Path

import pandas as pd
import pyqtgraph as pg
from pyqtgraph.Qt import QtWidgets

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

# Custom modules
from lift_ml.utils.rep_detection import RepDetectionResult, detect_rep_axis  # noqa: E402

# ----- Session config -------------------------------------------------

DATA_DIR = PROJECT_ROOT / "data/raw_sessions/FLOOR_PULL"
AXIS = "az"
FS = 130

# Sorted list of all CSV files in the directory
csv_files = sorted(DATA_DIR.glob("*.csv"), key=lambda p: int(p.stem))
current_idx = 0


# ----- Pyqtgraph Setup ------------------------------------------------

app = pg.mkQApp()
main_window = QtWidgets.QMainWindow()
central_widget = QtWidgets.QWidget()
layout = QtWidgets.QVBoxLayout()

main_window.setWindowTitle("Sensor Data Visualizer")
main_window.resize(800, 800)
main_window.setCentralWidget(central_widget)

central_widget.setLayout(layout)

plot_widget = pg.PlotWidget()
layout.addWidget(plot_widget)

# Navigation bar
nav_layout = QtWidgets.QHBoxLayout()
label_file = QtWidgets.QLabel()
btn_prev = QtWidgets.QPushButton("<- Prev.")
btn_next = QtWidgets.QPushButton("Next ->")
nav_layout.addWidget(label_file)
nav_layout.addStretch()
nav_layout.addWidget(btn_prev)
nav_layout.addWidget(btn_next)
layout.addLayout(nav_layout)

main_window.show()


#----- Session loader -------------------------------------------------

# Plot items kept so we can clear & redraw on each load
_az_curve: pg.PlotDataItem | None = None
_region: pg.LinearRegionItem | None = None


def load_session(idx: int) -> None:
    """Load CSV at csv_files[idx], run detection, and refresh the plot."""
    global _az_curve, _region

    path = csv_files[idx]
    label_file.setText(f"[{idx + 1}/{len(csv_files)}]  {path.name}")

    df = pd.read_csv(path)
    az_data = df[AXIS].to_numpy()

    res: RepDetectionResult | None = detect_rep_axis(
        df,
        axis=AXIS,
        fs=FS,
        baseline_seconds=1.0,
        k_start=24.0,
        k_end=5.0,
        smooth_window=5,
        min_duration=0.12,
    )

    # Clear previous items
    if _az_curve is not None:
        plot_widget.removeItem(_az_curve)
    if _region is not None:
        plot_widget.removeItem(_region)

    _az_curve = plot_widget.plot(az_data, clickable=True)
    if _az_curve is not None:
        _az_curve.curve.setClickable(True)
        _az_curve.setPen("w")  ## white pen

    if res is not None:
        _region = pg.LinearRegionItem(
            [res.model_start_idx, res.model_end_idx], orientation="vertical"
        )
        plot_widget.addItem(_region)
    else:
        _region = None


#----- Navigation -----------------------------------------------------

def on_next() -> None:
    global current_idx
    current_idx = (current_idx + 1) % len(csv_files)
    load_session(current_idx)


def on_prev() -> None:
    global current_idx
    current_idx = (current_idx - 1) % len(csv_files)
    load_session(current_idx)


btn_next.clicked.connect(on_next)
btn_prev.clicked.connect(on_prev)


#----- Initial load ---------------------------------------------------

load_session(current_idx)

if __name__ == "__main__":
    app.exec()

