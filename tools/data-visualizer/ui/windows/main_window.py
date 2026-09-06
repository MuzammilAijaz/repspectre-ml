# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnknownArgumentType=false, reportMissingTypeStubs=false, reportAttributeAccessIssue=false

import logging
from typing import Final

import numpy as np
import pyqtgraph as pg
from model.csv_session_loader import CsvSessionLoader
from model.visualizer_session import VisualizerSession
from pyqtgraph.Qt import QtWidgets

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

        self.setCentralWidget(central)

        # Model Loader & ViewModel
        self.loader = CsvSessionLoader()
        self.view_model = SessionViewModel(self.loader)

        # NOTE: order of callback "connection" matters; make sure the main window
        # callback always called first to avoid

        # Connect session rendering
        self.view_model.session_changed.connect(self.on_session_changed)

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
        self.selector_pane.set_available_datasets(
            available_datasets, self.loader.current_data_dir
        )

        # Initial render
        self.on_session_changed(self.view_model.current_session)

    def on_filter_applied(self, filtered_data: np.ndarray) -> None:
        if self.curve is not None:
            self.curve.setData(filtered_data)

    def on_session_changed(self, session: VisualizerSession | None) -> None:
        if session is None:
            logger.warning("No session to display (session is None)")
            if self.curve is not None:
                self.plot_widget.removeItem(self.curve)
                self.curve = None
            self.update_region(False, 0, 0)
            return

        # Clear previous curve
        if self.curve is not None:
            self.plot_widget.removeItem(self.curve)

        logger.info(
            "Plotting session '%s' with %d data points",
            session.path.name,
            len(session.sensor_data[self.view_model.current_active_axis]),
        )
        self.curve = self.plot_widget.plot(
            session.sensor_data[self.view_model.current_active_axis],
            clickable=True,
        )
        if self.curve is not None:
            self.curve.curve.setClickable(True)
            self.curve.setPen("w")  ## white pen

        if session.detection is not None:
            logger.info(
                "Drawing rep region for '%s': start=%d, end=%d",
                session.path.name,
                session.detection.model_start_idx,
                session.detection.model_end_idx,
            )
            self.update_region(
                True,
                session.detection.model_start_idx,
                session.detection.model_end_idx,
            )
        else:
            self.update_region(False, 0, 0)

    def update_region(self, is_on: bool, start: int, end: int) -> None:
        if self.region is not None:
            self.plot_widget.removeItem(self.region)

        if is_on:
            self.region = pg.LinearRegionItem(
                [start, end], orientation="vertical"
            )
            self.plot_widget.addItem(self.region)
        else:
            self.region = None




