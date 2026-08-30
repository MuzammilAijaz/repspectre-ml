# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false

from dataclasses import dataclass, field
from typing import Any

import pandas as pd

# Private sentinel token to ensure Session creation is enforced via to_session() or factories
_SESSION_CREATION_TOKEN = object()


@dataclass
class Session:
    """Domain representation of a recorded sensor session.

    Enforces creation via DbSession.to_session() or controlled factories.
    """

    session_id: int
    sensor_data: pd.DataFrame
    motion_state: str | None = None
    sensor_data_format: str = "FULL_IMU_RAW"
    metadata: dict[str, Any] = field(default_factory=dict)

    def __init__(
        self,
        *,
        _token: object,
        session_id: int,
        sensor_data: pd.DataFrame,
        motion_state: str | None = None,
        sensor_data_format: str = "FULL_IMU_RAW",
        metadata: dict[str, Any] | None = None,
    ) -> None:
        if _token is not _SESSION_CREATION_TOKEN:
            raise TypeError(
                "Session must be created through DbSession.to_session() or create_test_session()"
            )
        self.session_id = session_id
        self.sensor_data = sensor_data
        self.motion_state = motion_state
        self.sensor_data_format = sensor_data_format
        self.metadata = metadata or {}


def create_test_session(
    session_id: int,
    sensor_data: pd.DataFrame,
    motion_state: str | None = None,
    sensor_data_format: str = "FULL_IMU_RAW",
    metadata: dict[str, Any] | None = None,
) -> Session:
    """Helper factory for creating Session instances in tests and synthetic workflows."""
    return Session(
        _token=_SESSION_CREATION_TOKEN,
        session_id=session_id,
        sensor_data=sensor_data,
        motion_state=motion_state,
        sensor_data_format=sensor_data_format,
        metadata=metadata,
    )
