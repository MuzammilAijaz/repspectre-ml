#******************************************************************************
#  Data Ingestion Module
# -----------------------------------------------------------------------------
#  Responsible for fetching raw sensor data and metadata from various sources
#  (primarily SQLite) before any processing or dataset formatting.
#******************************************************************************

# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnknownArgumentType=false, reportArgumentType=false

import sqlite3
from dataclasses import dataclass, field
from typing import Any

import pandas as pd


@dataclass
class Session:
    """Represents a single recorded session with sensor data and metadata."""

    session_id: int
    sensor_data: pd.DataFrame
    motion_state: str | None = None
    sensor_data_format: str = "RAW"
    metadata: dict[str, Any] = field(default_factory=dict)


def get_all_sessions_data(db_path: str, class_type: str, subtype: str) -> list[Session]:
    """Fetch all session raw data for a specific metadata filter."""
    conn = sqlite3.connect(db_path)

    # Fetch session IDs and any relevant metadata
    query = f"SELECT * FROM session WHERE {class_type} = '{subtype}' ORDER BY sessionId ASC;"
    session_meta = pd.read_sql(query, conn)

    sessions: list[Session] = []
    for _, meta_row in session_meta.iterrows():
        sid = int(meta_row["sessionId"])
        df = pd.read_sql(
            f"SELECT ax, ay, az, gx, gy, gz FROM rawdata WHERE sessionId = {sid};",
            conn,
        )

        # Convert row to metadata dict
        metadata: dict[str, Any] = meta_row.to_dict()
        motion_state_val = metadata.get("liftCategory") or metadata.get("noise")
        motion_state = str(motion_state_val) if motion_state_val is not None else None

        sessions.append(
            Session(
                session_id=sid,
                sensor_data=df,
                motion_state=motion_state,
                metadata=metadata,
            )
        )

    conn.close()
    return sessions


def get_column_names(db_path: str, column_name: str, table_name: str) -> list[Any]:
    """Fetch distinct values from a metadata column."""
    conn = sqlite3.connect(db_path)
    df = pd.read_sql(f"SELECT DISTINCT {column_name} FROM {table_name};", conn)
    values_list: list[Any] = df[column_name].tolist()
    conn.close()
    return values_list


def get_all_type_sessions_data(
    db_path: str, class_types: list[str], sub_types: list[str]
) -> list[dict[int, pd.DataFrame]]:
    """Fetch raw data for multiple combinations of metadata types."""
    conn = sqlite3.connect(db_path)
    sessions_data_for_each_type: list[dict[int, pd.DataFrame]] = []

    for class_type in class_types:
        for sub_type in sub_types:
            session_ids = pd.read_sql(
                f"SELECT sessionId FROM session WHERE {class_type} = '{sub_type}' "
                f"ORDER BY sessionId ASC;",
                conn,
            )["sessionId"].tolist()

            sessions_data: dict[int, pd.DataFrame] = {}
            for sid in session_ids:
                df = pd.read_sql(
                    f"SELECT ax, ay, az, gx, gy, gz FROM rawdata WHERE sessionId = {sid};",
                    conn,
                )
                sessions_data[int(sid)] = df
            sessions_data_for_each_type.append(sessions_data)

    conn.close()
    return sessions_data_for_each_type

