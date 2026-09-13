# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnknownArgumentType=false, reportMissingTypeStubs=false, reportUnknownLambdaType=false

import logging
from typing import Any, Final

import numpy as np
from model.csv_session_loader import VisualizerSession
from model.filters import (
    apply_butterworth_lowpass,
    apply_median_filter,
    apply_moving_average,
    apply_quaternion_gravity_removal,
    apply_savgol_filter,
)
from pyqtgraph.parametertree import Parameter, ParameterTree
from pyqtgraph.Qt import QtWidgets
from PySide6.QtCore import Signal

from ui.view.session_viewmodel import SessionViewModel

logger: Final = logging.getLogger("ui.panes.filter_selector_pane")


class FilterSelectionPane(QtWidgets.QWidget):
    """ParameterTree pane for configuring and applying live signal filters sequentially."""

    # Emits filtered numpy array whenever filter params or session change
    filter_applied = Signal(object)

    def __init__(self, view_model: SessionViewModel) -> None:
        super().__init__()
        self.view_model = view_model

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        # 5 stage filtering pipeline
        pipeline_stages = []
        for i in range(1, 6):
            stage = {
                    "name": f"Stage {i}",
                    "type": "group",
                    "children": [
                        {
                            "name": "Filter Type",
                            "type": "list",
                            "limits": [
                                "None",
                                "Moving Average (Uniform)",
                                "Butterworth Lowpass",
                                "Median Filter",
                                "Savitzky-Golay",
                                "Remove Gravity (using Quaternions) (only for acceleration)",
                                ],
                            "value": "None",
                            },
                        {
                            "name": "Window / Kernel Size",
                            "type": "int",
                            "value": 5,
                            "limits": (1, 101),
                            "step": 2,
                            },
                        {
                            "name": "Cutoff Freq (Hz)",
                            "type": "float",
                            "value": 5.0,
                            "limits": (0.1, 100.0),
                            "step": 0.5,
                            },
                        {
                            "name": "Filter Order",
                            "type": "int",
                            "value": 2,
                            "limits": (1, 5),
                            },
                        {
                            "name": "SavGol Polyorder",
                            "type": "int",
                            "value": 2,
                            "limits": (1, 4),
                            },
                        ],
                    }
            pipeline_stages.append(stage)

        self.param_config = Parameter.create(
                name="Filter Pipeline",
                type="group",
                children=[
                    {"name": "Enable Pipeline", "type": "bool", "value": True},
                    *pipeline_stages
                    ],
                )

        self.tree = ParameterTree()
        self.tree.setParameters(self.param_config, showTop=False)
        layout.addWidget(self.tree)

        # visibility toggles for each stage
        for i in range(1, 6):
            stage_param = self.param_config.child(f"Stage {i}")

            # When the Filter Type dropdown changes, update the visible parameters
            type_param = stage_param.child("Filter Type")
            type_param.sigValueChanged.connect(
                    lambda _param, _val, sp=stage_param: self._update_stage_visibility(sp)
            )

            # Initial UI visibility setup
            self._update_stage_visibility(stage_param)

        # Internal signal wiring
        self.param_config.sigTreeStateChanged.connect(self._on_param_changed)
        self.view_model.session_changed.connect(self._on_session_changed)

        # Initial execution
        self._run_filter()

    def _update_stage_visibility(self, stage_param: Parameter) -> None:
        """Dynamically show/hide parameters based on the selected filter type."""
        f_type = stage_param["Filter Type"]

        # Hide all context-specific parameters first
        stage_param.child("Window / Kernel Size").hide()
        stage_param.child("Cutoff Freq (Hz)").hide()
        stage_param.child("Filter Order").hide()
        stage_param.child("SavGol Polyorder").hide()

        # Selectively show only what the current filter needs
        if f_type in ["Moving Average (Uniform)", "Median Filter"]:
            stage_param.child("Window / Kernel Size").show()
        elif f_type == "Butterworth Lowpass":
            stage_param.child("Cutoff Freq (Hz)").show()
            stage_param.child("Filter Order").show()
        elif f_type == "Savitzky-Golay":
            stage_param.child("Window / Kernel Size").show()
            stage_param.child("SavGol Polyorder").show()

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

        raw_axis_data = np.asarray(session.sensor_data[self.view_model.current_active_axis])
        enabled: bool = self.param_config["Enable Pipeline"]
        logger.debug("Filter pipeline status: %s", enabled)

        if not enabled:
            self.filter_applied.emit(raw_axis_data.copy())
            return

        # Initialize data for the first stage and get sampling rate
        data = raw_axis_data.copy()
        fs: float = float(getattr(session, "sampling_rate", 100.0) or 100.0)

        # display 5 different filter options
        for i in range(1, 6):
            stage = self.param_config.child(f"Stage {i}")
            f_type = stage["Filter Type"]

            # Bypass inactive stages to avoid filtering
            if f_type == "None":
                continue

            # Retrieve parameters for the current stage
            window_size: int = stage["Window / Kernel Size"]
            cutoff_hz: float = stage["Cutoff Freq (Hz)"]
            order: int = stage["Filter Order"]
            polyorder: int = stage["SavGol Polyorder"]

            # Process data
            if f_type == "Moving Average (Uniform)":
                data = apply_moving_average(data, window_size)
            elif f_type == "Butterworth Lowpass":
                data = apply_butterworth_lowpass(data, cutoff_hz=cutoff_hz, fs=fs, order=order)
            elif f_type == "Median Filter":
                data = apply_median_filter(data, kernel_size=window_size)
            elif f_type == "Savitzky-Golay":
                data = apply_savgol_filter(data, window_length=window_size, polyorder=polyorder)
            elif f_type == "Remove Gravity (using Quaternions) (only for acceleration)" \
                    and self.view_model.current_active_axis in ['ax', 'ay', 'az']:
                data = apply_quaternion_gravity_removal(
                    data,
                    quaternions=session.sensor_data[['qx', 'qy', 'qz', 'qw']].to_numpy(),
                    axis=self.view_model.current_active_axis
                )
                # TODO: add warning for user if axis is not acceleration, instead of silent ignore.

        # Emit the fully processed signal chain
        self.filter_applied.emit(data)
