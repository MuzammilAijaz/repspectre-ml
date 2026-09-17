# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnknownArgumentType=false, reportMissingTypeStubs=false, reportAttributeAccessIssue=false

#*****************************************************************************
#  Classification Window
#-----------------------------------------------------------------------------
#  Displays multi-axis IMU time-series sensor data (ax, ay, az, gx, gy, gz)
#  with a synchronized, fixed-width line region across all graphs.
#
#----------------------------------------------------------------------------
# FIXME: linear regions can be positioned outside when cursor outisde graph
#        However, clamping works for selected_start and selected_end
#
#*****************************************************************************

import logging
from typing import Final, cast

import numpy as np
import pandas as pd
import pyqtgraph as pg
from model.visualizer_session import VisualizerSession
from pyqtgraph.Qt import QtCore, QtWidgets

from ui.view.session_viewmodel import SessionViewModel

logger: Final = logging.getLogger("ui.windows.classification_window")

# Default classification model window length
# NOTE: This would determine the window length of all classes of the model as well!
#       Also affecting things like inference frequency etc.
# TODO: show user the time period (calcuAted from the samples and FS).
# TODO: make intent of this variable more clear to user.
DEFAULT_REGION_WIDTH: Final[int] = 50

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
    """
    Multi-graph window capturing sensor channels with a synchronized fixed-size selection region.
    """

    def __init__(
        self,
        view_model: SessionViewModel,
        parent: QtWidgets.QWidget | None = None,
        region_width: int = DEFAULT_REGION_WIDTH,
    ) -> None:
        super().__init__(parent)

        self.setWindowTitle("Data Classification - Multi-Axis View")
        self.resize(1100, 850)

        self.view_model = view_model

        # Fixed region dimensions and user-selected start/end bounds
        self.region_width: int = region_width
        self.selected_start: int = 0
        self.selected_end: int = self.region_width
        self._current_data_len: int = 0

        # Used as a rentrancy guard for allowing syncing of linear regions across different graphs
        self._syncing_regions: bool = False

        central = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(central)

        # Plot layout widget containing multiple linked graphs
        self.glw: pg.GraphicsLayoutWidget = pg.GraphicsLayoutWidget()
        layout.addWidget(self.glw)

        # Build plots, curves, and synchronized LinearRegionItems for all 6 axes
        self.plots: dict[str, pg.PlotItem] = {}
        self.curves: dict[str, pg.PlotDataItem] = {}
        self.regions: dict[str, pg.LinearRegionItem] = {}

        #----- Setup Graphs -------------------------------------------

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

            # Synchronized fixed-width selection region
            region = pg.LinearRegionItem(
                [self.selected_start, self.selected_end],
                orientation="vertical",
                movable=True,
                brush=pg.mkBrush(46, 204, 113, 50),
            )
            # Disable individual boundary line dragging to enforce fixed width
            region.lines[0].setMovable(False)
            region.lines[1].setMovable(False)
            p.addItem(region)
            self.regions[axis_key] = region

            # Connect region sliding to sync across all graphs
            region.sigRegionChanged.connect(lambda r=region: self._on_region_changed(r))

        #--------------------------------------------------------------

        # Status label displaying current user selection
        self.status_label = QtWidgets.QLabel()
        self.status_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.status_label)
        self._update_status_label()

        self.setCentralWidget(central)

        # Connect session updates
        self.view_model.session_changed.connect(self.on_session_changed)

        # Initial render
        if self.view_model.current_session is not None:
            self.update_graphs(self.view_model.current_session.modified_data)
        else:
            self.clear_graphs()

    #===== Callbacks ==============================================================

    def _on_region_changed(self, source_region: pg.LinearRegionItem) -> None:
        """
        Synchronizes region sliding across all 6 plots with fixed width and boundary clamping.

        The region is treated as a fixed-width window: only its starting position
        is taken from the source region. The start position is clamped so the
        complete window remains within the currently loaded data. The resulting
        integer-aligned bounds are then applied to every plot and reflected in
        the status label.

        Avoiding Rentrancy:
        -------------------
        PyQtGraph emits `sigRegionChanged` whenever the region position changes.
        Because this callback updates every region, those updates can emit the same
        signal again. `_syncing_regions` acts as a re-entrancy guard so that only
        the original user-driven change is processed.
        """

        # if syncing is already being done, return. Avoids infinite loop
        if self._syncing_regions:
            return

        self._syncing_regions = True
        try:
            bounds = cast("tuple[float, float]", source_region.getRegion())
            raw_start = float(bounds[0])

            width = self.region_width
            data_len = self._current_data_len

            # Clamp region within valid data bounds
            if data_len > 0:
                if width > data_len:
                    width = data_len
                max_start = max(0.0, float(data_len - width))
                clamped_start = max(0.0, min(max_start, float(raw_start)))
            else:
                clamped_start = max(0.0, float(raw_start))

            clamped_end = clamped_start + width
            self.selected_start = int(round(clamped_start))
            self.selected_end = int(round(clamped_end))

            # Sync all region items across all axis plots
            for reg in self.regions.values():
                # WARN: Calls sigRegionChanged -> reentry to _on_region_changed
                reg.setRegion((self.selected_start, self.selected_end))

            self._update_status_label()
        finally:
            # guarantee when something goes wrong, LinearRegion is not blocked.
            self._syncing_regions = False

    def _update_status_label(self) -> None:
        """Updates the status label with current selection bounds."""
        length = self.selected_end - self.selected_start
        self.status_label.setText(
            f"Selected Classification Window: [{self.selected_start} : {self.selected_end}]  "
            f"(Length: {length} samples)"
        )

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
        self._current_data_len = 0
        for curve in self.curves.values():
            curve.clear()

    def update_graphs(self, modified_data: pd.DataFrame) -> None:
        """Captures and plots all 6 sensor axes from session.modified_data."""
        df = modified_data
        self._current_data_len = len(df)

        #----- Update Regions -----------------------------------------
        # Center the fixed-width region initially when new data is loaded

        if self._current_data_len > 0:
            width = min(self.region_width, self._current_data_len)
            center = self._current_data_len // 2
            start = max(0, center - width // 2)
            end = start + width

            self.selected_start = start
            self.selected_end = end

            self._syncing_regions = True
            try:
                for reg in self.regions.values():
                    reg.setRegion((self.selected_start, self.selected_end))
            finally:
                # guarantee when something goes wrong, LinearRegion is not blocked.
                self._syncing_regions = False

            self._update_status_label()

        #----- Update Graphs ------------------------------------------
        # Update curves for each axis

        for axis_key, _title, _row, _col, _color in AXIS_CONFIGS:
            if axis_key not in df.columns:
                logger.warning("Axis '%s' missing from session modified_data", axis_key)
                continue

            data = np.asarray(df[axis_key], dtype=np.float64)
            curve = self.curves.get(axis_key)
            if curve is not None:
                curve.setData(data)
