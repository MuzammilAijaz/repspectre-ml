# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnknownArgumentType=false, reportMissingTypeStubs=false

import logging
from typing import Any, Final

import numpy as np
import pandas as pd
from model.csv_session_loader import VisualizerSession
from pyqtgraph.parametertree import Parameter, ParameterTree
from pyqtgraph.Qt import QtWidgets
from PySide6.QtCore import Signal

from ui.view.session_viewmodel import SessionViewModel

logger: Final = logging.getLogger("ui.panes.filter_selector_pane")


def apply_moving_average(data: np.ndarray, window_size: int) -> np.ndarray:
    """Applies centered rolling average to 1D array."""
    if window_size <= 1 or len(data) == 0:
        return data
    result = (
        pd.Series(data)
        .rolling(window=window_size, min_periods=1, center=True)
        .mean()
    )
    return np.asarray(result)


class FilterSelectionPane(QtWidgets.QWidget):
    """ParameterTree pane for configuring and applying live signal filters."""

    # Emits filtered numpy array whenever filter params or session change
    filter_applied = Signal(object)

    def __init__(self, view_model: SessionViewModel) -> None:
        super().__init__()
        self.view_model = view_model

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        # ParameterTree configuration
        self.param_config = Parameter.create(
                name="Filter Parameters",
                type="group",
                children=[
                    {
                        "name": "Enable Filter",
                        "type": "bool",
                        "value": False
                    },

                    {
                        "name": "Filter Type",
                        "type": "list",
                        "limits": ["Moving Average", "Raw (Bypasses filter)"],
                        "value": "Moving Average",
                    },

                    {
                        "name": "Window Size",
                        "type": "int",
                        "value": 5,
                        "limits": (1, 51),
                        "step": 2,
                    },

                ],
        )

        self.tree = ParameterTree()
        self.tree.setParameters(self.param_config, showTop=False)
        layout.addWidget(self.tree)

        # Internal signal wiring
        self.param_config.sigTreeStateChanged.connect(self._on_param_changed)
        self.view_model.session_changed.connect(self._on_session_changed)

        # Initial execution
        self._run_filter()


    def _on_param_changed(
            self,
            _param: Parameter,
            _changes: list[tuple[Parameter, str, Any]],
    ) -> None:
        logger.info("Parameter changed")
        self._run_filter()

    def _on_session_changed(self, _session: VisualizerSession | None) -> None:
        logger.info("Session updated")
        self._run_filter()

    def _run_filter(self) -> None:
        session = self.view_model.current_session
        if session is None:
            return

        raw_data = np.asarray(session.axis_data)
        enabled: bool = self.param_config["Enable Filter"]
        logger.debug("Filter status: %s", enabled)
        filter_type: str = self.param_config["Filter Type"]
        window_size: int = self.param_config["Window Size"]

        if enabled and filter_type == "Moving Average":
            filtered = apply_moving_average(raw_data, window_size)
        else:
            filtered = raw_data.copy()

        self.filter_applied.emit(filtered)
