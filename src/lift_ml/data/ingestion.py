#******************************************************************************
#  Data Ingestion Module
# -----------------------------------------------------------------------------
#  Responsible for fetching raw sensor data and metadata from various sources
#  (primarily SQLite) before any processing or dataset formatting.
#******************************************************************************

import sqlite3
import pandas as pd
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List

@dataclass
class Session:
    """Represents a single recorded session with sensor data and metadata."""
    session_id: int
    sensor_data: pd.DataFrame
    motion_state: Optional[str] = None
    sensor_data_format: str = "RAW"
    metadata: Dict[str, Any] = field(default_factory=dict)

def get_all_sessions_data(db_path, classType, subtype) -> List[Session]:
    """Fetch all session raw data for a specific metadata filter."""
    conn = sqlite3.connect(db_path)
    
    # Fetch session IDs and any relevant metadata
    query = f"SELECT * FROM session WHERE {classType} = '{subtype}' ORDER BY sessionId ASC;"
    session_meta = pd.read_sql(query, conn)
    
    sessions = []
    for _, meta_row in session_meta.iterrows():
        sid = int(meta_row['sessionId'])
        df = pd.read_sql(
            f"SELECT ax, ay, az, gx, gy, gz FROM rawdata WHERE sessionId = {sid};",
            conn
        )
        
        # Convert row to metadata dict
        metadata = meta_row.to_dict()
        motion_state = metadata.get('liftCategory') or metadata.get('noise')
        
        sessions.append(Session(
            session_id=sid,
            sensor_data=df,
            motion_state=motion_state,
            metadata=metadata
        ))
    
    conn.close()
    return sessions

def get_column_names(db_path, column_name, table_name):
    """Fetch distinct values from a metadata column."""
    conn = sqlite3.connect(db_path)
    df = pd.read_sql(f"SELECT DISTINCT {column_name} FROM {table_name};", conn)
    values_list = df[column_name].tolist()
    conn.close()
    return values_list

def get_all_type_sessions_data(db_path, classTypes, subTypes):
    """Fetch raw data for multiple combinations of metadata types."""
    conn = sqlite3.connect(db_path)
    sessions_data_for_each_type = [] 

    for classType in classTypes:
        for subType in subTypes:
            session_ids = pd.read_sql(
                f"SELECT sessionId FROM session WHERE {classType} = '{subType}' ORDER BY sessionId ASC;",
                conn
            )['sessionId'].tolist()
            
            sessions_data = {}
            for sid in session_ids:
                df = pd.read_sql(
                    f"SELECT ax, ay, az, gx, gy, gz FROM rawdata WHERE sessionId = {sid};",
                    conn
                )
                sessions_data[sid] = df
            sessions_data_for_each_type.append(sessions_data)
    
    conn.close()
    return sessions_data_for_each_type
