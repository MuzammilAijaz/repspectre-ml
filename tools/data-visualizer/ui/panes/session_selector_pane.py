# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnknownArgumentType=false, reportMissingTypeStubs=false

import logging
from enum import Enum
from pathlib import Path
from typing import Final

from model.csv_session_loader import VisualizerSession
from pyqtgraph.Qt import QtWidgets

from ui.view.session_viewmodel import SessionViewModel
from ui.widgets.dataset_selection import DatasetSelectionWidget
from ui.widgets.navigation import NavigationWidget

logger: Final = logging.getLogger("ui.panes.session_selector_pane")



class AxisOption(Enum):
    AX = ("ax", "Accel X")
    AY = ("ay", "Accel Y")
    AZ = ("az", "Accel Z")
    GX = ("gx", "Gyro X")
    GY = ("gy", "Gyro Y")
    GZ = ("gz", "Gyro Z")
    QX = ("qx", "Qauternion X")
    QY = ("qy", "Quaternion Y")
    QZ = ("qz", "Quaternion Z")
    QW = ("qw", "Quaternion W")

    def __init__(self, value: str, label: str) -> None:
        self.axis_name = value
        self.label = label


class SessionSelectorPane(QtWidgets.QWidget):
    """Composite pane combining dataset directory selection and session navigation controls."""

    def __init__(self, view_model: SessionViewModel, data_root: Path) -> None:
        super().__init__()

        self.view_model = view_model
        self.data_root = data_root

        layout = QtWidgets.QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        # Child widgets
        self.dataset_widget: DatasetSelectionWidget = DatasetSelectionWidget()
        self.navigation_widget: NavigationWidget = NavigationWidget()

        layout.addWidget(self.dataset_widget)
        layout.addWidget(self.navigation_widget)

        # Internal signal wiring
        self.navigation_widget.btn_next.clicked.connect(self.view_model.next_session)
        self.navigation_widget.btn_prev.clicked.connect(
            self.view_model.previous_session
        )
        self.dataset_widget.dataset_selected.connect(self._on_dataset_selected)
        self.view_model.session_changed.connect(self._on_session_changed)

        # Select axis
        self.axis_combo = QtWidgets.QComboBox()
        self._populate_axes()

        layout.addWidget(self.dataset_widget)
        layout.addWidget(self.axis_combo)
        layout.addWidget(self.navigation_widget)

        self.axis_combo.currentIndexChanged.connect(self._on_axis_combo_changed)


        # Initial label update
        self._on_session_changed(self.view_model.current_session)

    def _populate_axes(self) -> None:
        for option in AxisOption:
            self.axis_combo.addItem(option.label, option.axis_name)

        index = self.axis_combo.findData(self.view_model.current_active_axis)
        if index >= 0:
            self.axis_combo.setCurrentIndex(index)

    def _on_axis_combo_changed(self, index: int) -> None:
        axis_name = self.axis_combo.itemData(index)
        if axis_name:
            logger.info("Axis changed: %s", axis_name)
            self.view_model.on_axis_changed(axis_name)

    def set_available_datasets(
        self, datasets: list[Path], current_path: Path | None = None
    ) -> None:
        """Populate the hierarchical dataset dropdown menu."""
        self.dataset_widget.set_available_datasets(
            datasets, self.data_root, current_path
        )

    def _on_dataset_selected(self, path: Path) -> None:
        logger.info("Dataset selected: %s", path)
        self.dataset_widget.set_current_dataset_label(path, self.data_root)
        self.view_model.select_dataset(path)

    def _on_session_changed(self, session: VisualizerSession | None) -> None:
        if session is None:
            self.navigation_widget.label_file.setText("[0/0]  No CSV files found")
            return

        idx_str = f"[{self.view_model.current_idx + 1}/{self.view_model.session_count}]"
        hz_str = (
            f" ({int(round(session.sampling_rate))} Hz)"
            if session.sampling_rate
            else ""
        )
        self.navigation_widget.label_file.setText(
            f"{idx_str}  {session.path.name}{hz_str}"
        )
