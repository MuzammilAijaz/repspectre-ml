# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnknownArgumentType=false, reportMissingTypeStubs=false, reportAttributeAccessIssue=false

import logging
from pathlib import Path
from typing import Final

import pyqtgraph as pg
from model.csv_session_loader import CsvSessionLoader, VisualizerSession
from pyqtgraph.Qt import QtWidgets

from ui.view.session_viewmodel import SessionViewModel
from ui.widgets.dataset_selection import DatasetSelectionWidget
from ui.widgets.navigation_bar import NavigationBarWidget

logger: Final = logging.getLogger("ui.windows.main_window")


class MainWindow(QtWidgets.QMainWindow):

    def __init__(self) -> None:
        super().__init__()

        self.setWindowTitle("Sensor Data Visualizer")
        self.resize(800, 800)

        central = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(central)

        # Add widget
        self.plot_widget: pg.PlotWidget = pg.PlotWidget()
        layout.addWidget(self.plot_widget)

        # Bottom controls layout
        bottom_layout = QtWidgets.QHBoxLayout()
        self.dataset_widget: DatasetSelectionWidget = DatasetSelectionWidget()
        self.navigation_widget: NavigationBarWidget = NavigationBarWidget()
        bottom_layout.addWidget(self.dataset_widget)
        bottom_layout.addWidget(self.navigation_widget)
        layout.addLayout(bottom_layout)

        # variables
        self.region: pg.LinearRegionItem | None = None
        self.curve: pg.PlotDataItem | None = None

        self.setCentralWidget(central)

        #
        # REFACTOR: view should not be know about model or be responsible for orchestration
        #

        # Model Loader
        self.loader = CsvSessionLoader()

        # ViewModel
        self.view_model = SessionViewModel(self.loader)

        # Populate datasets into dataset selection widget
        available_datasets = self.loader.scan_available_datasets()
        logger.info("Initializing UI with %d available dataset folders", len(available_datasets))
        self.dataset_widget.set_available_datasets(
            available_datasets,
            self.loader.DATA_ROOT,
            self.loader.current_data_dir,
        )

        # Signals
        self.navigation_widget.btn_next.clicked.connect(self.view_model.next_session)
        self.navigation_widget.btn_prev.clicked.connect(self.view_model.previous_session)
        self.dataset_widget.dataset_selected.connect(self.on_dataset_selected)
        self.view_model.session_changed.connect(self.on_session_changed)

        # Initial render
        self.on_session_changed(self.view_model.current_session)

    def on_dataset_selected(self, path: Path) -> None:
        logger.info("Dataset selected from UI: %s", path)
        self.dataset_widget.set_current_dataset_label(path, self.loader.DATA_ROOT)
        self.view_model.select_dataset(path)

    def on_session_changed(self, session: VisualizerSession | None) -> None:
        if session is None:
            logger.warning("No session to display (session is None)")
            self.navigation_widget.label_file.setText("[0/0]  No CSV files found")
            if self.curve is not None:
                self.plot_widget.removeItem(self.curve)
                self.curve = None
            self.update_region(False, 0, 0)
            return

        idx_str = f"[{self.view_model.current_idx + 1}/{self.view_model.session_count}]"
        hz_str = f" ({int(round(session.sampling_rate))} Hz)" if session.sampling_rate else ""
        self.navigation_widget.label_file.setText(
            f"{idx_str}  {session.path.name}{hz_str}"
        )

        # Clear previous curve
        if self.curve is not None:
            self.plot_widget.removeItem(self.curve)

        logger.info("Plotting session '%s' with %d data points",
                    session.path.name, len(session.axis_data))
        self.curve = self.plot_widget.plot(session.axis_data, clickable=True)
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




