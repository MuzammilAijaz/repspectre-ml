# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnknownArgumentType=false, reportMissingTypeStubs=false, reportAttributeAccessIssue=false

#*****************************************************************************
#  Classification Window
# -----------------------------------------------------------------------------
#  Displays multi-axis IMU time-series sensor data (ax, ay, az, gx, gy, gz)
#  directly capturing modified_data from the active session.
#*****************************************************************************

import logging
from typing import Final

import numpy as np
import pandas as pd
import pyqtgraph as pg
from model.visualizer_session import VisualizerSession
from pyqtgraph.Qt import QtWidgets

from ui.view.session_viewmodel import SessionViewModel

logger: Final = logging.getLogger("ui.windows.classification_window")


# Axes to display: (column_key, display_title, grid_row, grid_col, pen_color)
AXIS_CONFIGS: Final = [
    ("ax", "Accel X (ax)", 0, 0, "#e74c3c"),
    ("ay", "Accel Y (ay)", 1, 0, "#2ecc71"),
    ("az", "Accel Z (az)", 2, 0, "#3498db"),
    ("gx", "Gyro X (gx)", 0, 1, "#e67e22"),
    ("gy", "Gyro Y (gy)", 1, 1, "#9b59b6"),
    ("gz", "Gyro Z (gz)", 2, 1, "#f1c40f"),
]


class ClassificationWindow(QtWidgets.QMainWindow):
    """Multi-graph window capturing and displaying sensor channels from modified_data."""

    def __init__(
        self,
        view_model: SessionViewModel,
        parent: QtWidgets.QWidget | None = None,
    ) -> None:
        super().__init__(parent)

        self.setWindowTitle("Data Classification - Multi-Axis View")
        self.resize(1100, 850)

        self.view_model = view_model

        central = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(central)

        # Plot layout widget containing multiple linked graphs
        self.glw: pg.GraphicsLayoutWidget = pg.GraphicsLayoutWidget()
        layout.addWidget(self.glw)

        # Build plots and curves for all 6 axes with linked X axis
        self.plots: dict[str, pg.PlotItem] = {}
        self.curves: dict[str, pg.PlotDataItem] = {}

        first_plot: pg.PlotItem | None = None
        for axis_key, title, row, col, color in AXIS_CONFIGS:
            p: pg.PlotItem = self.glw.addPlot(row=row, col=col, title=title)
            p.showGrid(x=True, y=True, alpha=0.3)
            p.setMouseEnabled(x=False, y=False)

            if first_plot is None:
                first_plot = p
            else:
                p.setXLink(first_plot)

            self.plots[axis_key] = p
            self.curves[axis_key] = p.plot(pen=pg.mkPen(color=color, width=1.5))

        self.setCentralWidget(central)

        # Connect session updates
        self.view_model.session_changed.connect(self.on_session_changed)

        # Initial render
        if self.view_model.current_session is not None:
            self.update_graphs(self.view_model.current_session.modified_data)
        else:
            self.clear_graphs()

    #===== Callbacks ==============================================================

    def on_session_changed(self, session: VisualizerSession | None) -> None:
        """Called when active session changes."""
        if session is None:
            logger.warning("No session to display (session is None)")
            self.clear_graphs()
            return

        logger.info("Classification window session changed: %s", session.path.name)
        self.update_graphs(session.modified_data)

    #===== Graph Updating =========================================================

    def clear_graphs(self) -> None:
        """Clears all curve data across all plots."""
        for curve in self.curves.values():
            curve.clear()

    def update_graphs(self, modified_data: pd.DataFrame) -> None:
        """Captures and plots all 6 sensor axes from session.modified_data."""
        df = modified_data

        for axis_key, _title, _row, _col, _color in AXIS_CONFIGS:
            if axis_key not in df.columns:
                logger.warning("Axis '%s' missing from session modified_data", axis_key)
                continue

            data = np.asarray(df[axis_key], dtype=np.float64)
            curve = self.curves.get(axis_key)
            if curve is not None:
                curve.setData(data)
