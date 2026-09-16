# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnknownArgumentType=false, reportMissingTypeStubs=false, reportAttributeAccessIssue=false

import logging
from typing import Final

import numpy as np
import pandas as pd
import pyqtgraph as pg
from model.csv_session_loader import CsvSessionLoader
from model.visualizer_session import RepDetectionResult, VisualizerSession
from pyqtgraph.Qt import QtWidgets

from lift_ml.utils.rep_detection import detect_rep_axis
from lift_ml.utils.sampling import calculate_sampling_rate
from ui.panes.filter_selector_pane import FilterSelectionPane
from ui.panes.session_selector_pane import SessionSelectorPane
from ui.view.session_viewmodel import SessionViewModel

logger: Final = logging.getLogger("ui.windows.main_window")


class MainWindow(QtWidgets.QMainWindow):

    def __init__(self) -> None:
        super().__init__()

        self.setWindowTitle("Sensor Data Visualizer")
        self.resize(800, 800)

        central = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(central)

        # Plot display widget
        self.plot_widget: pg.PlotWidget = pg.PlotWidget()
        layout.addWidget(self.plot_widget)

        # Plot items
        self.region: pg.LinearRegionItem | None = None
        self.curve: pg.PlotDataItem | None = None

        # Rep detection toggle button (below graph, above data selection pane)
        self.rep_detection_enabled: bool = True
        self.toggle_detection_btn = QtWidgets.QPushButton("Disable Rep Detection")
        self.toggle_detection_btn.clicked.connect(self.on_toggle_rep_detection)
        layout.addWidget(self.toggle_detection_btn)

        self.setCentralWidget(central)

        # Model Loader & ViewModel
        self.loader = CsvSessionLoader()
        self.view_model = SessionViewModel(self.loader)

        # NOTE: order of callback "connection" matters; make sure the main window
        # callback always called first to avoid

        # Connect session rendering
        self.view_model.session_changed.connect(self.on_session_changed)
        # WARN: maybe unnecessary
        # call the on_session_change to update the UI for new axis
        self.view_model.axis_changed.connect(self.on_session_changed)

        # Bottom controls pane (encapsulates dataset dropdown & navigation buttons)
        self.selector_pane = SessionSelectorPane(
            self.view_model, self.loader.DATA_ROOT
        )
        layout.addWidget(self.selector_pane)

        self.filter_selector_pane = FilterSelectionPane(self.view_model)
        layout.addWidget(self.filter_selector_pane)
        self.filter_selector_pane.filter_applied.connect(self.on_filter_applied)

        # Populate available datasets into selector pane
        available_datasets = self.loader.scan_available_datasets()
        logger.info(
            "Initializing UI with %d available dataset folders",
            len(available_datasets),
        )
        self.selector_pane.set_available_datasets(available_datasets, self.loader.current_data_dir)

        # Initial render
        self.on_session_changed(self.view_model.current_session)

    #===== Callbacks ==============================================================

    def on_filter_applied(self, filtered_df: pd.DataFrame) -> None:
        session = self.view_model.current_session
        if session is None:
            return

        # Updates all the axis
        session.modified_data = filtered_df

        self.update_graph(session.modified_data[axis])
        self.update_detection()

    def on_session_changed(self, session: VisualizerSession | None) -> None:
        if session is None:
            logger.warning("No session to display (session is None)")
            self.clear_graph()
            return

        # Reset modified_data to a fresh copy of original sensor_data for the new session
        session.modified_data = session.sensor_data.copy()

        axis = self.view_model.current_active_axis
        logger.info(
            "Plotting session '%s' with %d data points",
            session.path.name,
            len(session.modified_data[axis]),
        )
        self.update_graph(session.modified_data[axis])
        self.update_detection()

    def on_toggle_rep_detection(self) -> None:
        self.rep_detection_enabled = not self.rep_detection_enabled
        if self.rep_detection_enabled:
            self.toggle_detection_btn.setText("Disable Rep Detection")
            self.update_detection()
        else:
            self.toggle_detection_btn.setText("Enable Rep Detection")
            self.update_region(False, 0, 0)

    #==============================================================================

    def update_graph(self, plot_data: pd.Series | np.ndarray | pd.DataFrame) -> None:
        if self.view_model.current_session is None:
            logger.warning("No session to display (session is None)")
            self.clear_graph()
            return

        # Clear previous curve
        if self.curve is not None:
            self.plot_widget.removeItem(self.curve)

        self.curve = self.plot_widget.plot(
            plot_data,
            clickable=True,
        )
        if self.curve is not None:
            self.curve.curve.setClickable(True)
            self.curve.setPen("w")  ## white pen

    def clear_graph(self) -> None:
        if self.curve is not None:
            self.plot_widget.removeItem(self.curve)
            self.curve = None
        self.update_region(False, 0, 0)

    def update_detection(self) -> None:
        if not self.rep_detection_enabled:
            self.update_region(False, 0, 0)
            return

        session = self.view_model.current_session
        if session is None:
            return

        # Redetect repetitions
        df = session.modified_data
        detection: RepDetectionResult | None = detect_rep_axis(
            df,
            axis=self.view_model.current_active_axis,
            fs=int(round(calculate_sampling_rate(df))),
            baseline_seconds=1.0,
            k_start=24.0,
            k_end=5.0,
            smooth_window=5,
            min_duration=0.12,
        )
        session.detection = detection

        if detection is not None:
            self.update_region(
                True,
                detection.model_start_idx,
                detection.model_end_idx,
            )
        else:
            self.update_region(False, 0, 0)

    def update_region(self, is_on: bool, start: int, end: int) -> None:
        if self.region is not None:
            self.plot_widget.removeItem(self.region)
            self.region = None

        if is_on:
            session = self.view_model.current_session
            if session is not None and session.detection is not None:
                logger.info(
                    "Drawing rep region for '%s': start=%d, end=%d",
                    session.path.name,
                    start,
                    end,
                )
            self.region = pg.LinearRegionItem([start, end], orientation="vertical")
            self.plot_widget.addItem(self.region)

