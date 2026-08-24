# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnknownArgumentType=false, reportMissingTypeStubs=false, reportAttributeAccessIssue=false

import pyqtgraph as pg
from model.session_repository import Session, SessionRepository
from pyqtgraph.Qt import QtWidgets

from ui.view.session_viewmodel import SessionViewModel
from ui.widgets.navigation_bar import NavigationBarWidget


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

        self.navigation_widget: NavigationBarWidget = NavigationBarWidget()
        layout.addWidget(self.navigation_widget)

        # variables
        self.region: pg.LinearRegionItem | None = None
        self.curve: pg.PlotDataItem | None = None

        self.setCentralWidget(central)

        # Model & ViewModel
        self.repository = SessionRepository()
        self.view_model = SessionViewModel(self.repository)

        # Signals
        self.navigation_widget.btn_next.clicked.connect(self.view_model.next_session)
        self.navigation_widget.btn_prev.clicked.connect(self.view_model.previous_session)
        self.view_model.session_changed.connect(self.on_session_changed)

        # Initial render
        self.on_session_changed(self.view_model.current_session)

    def on_session_changed(self, session: Session) -> None:
        self.navigation_widget.label_file.setText(
            f"[{self.view_model.current_idx + 1}\
/{self.view_model.session_count}]  {session.path.name}"
        )

        # Clear previous curve
        if self.curve is not None:
            self.plot_widget.removeItem(self.curve)

        self.curve = self.plot_widget.plot(session.axis_data, clickable=True)
        if self.curve is not None:
            self.curve.curve.setClickable(True)
            self.curve.setPen("w")  ## white pen

        if session.detection is not None:
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


