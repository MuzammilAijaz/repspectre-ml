#******************************************************************************
#  Data Ingestion Module
# -----------------------------------------------------------------------------
#  Responsible for fetching raw sensor data and metadata from various sources
#  (primarily SQLite) before any processing or dataset formatting.
#******************************************************************************

import sqlite3
import pandas as pd

def get_all_sessions_data(db_path, classType, subtype):
    """Fetch all session raw data for a specific metadata filter."""
    conn = sqlite3.connect(db_path)
    
    # All session IDs for the given type
    session_ids = pd.read_sql(
        f"SELECT sessionId FROM session WHERE {classType} = '{subtype}' ORDER BY sessionId ASC;",
        conn
    )['sessionId'].tolist()
    
    # Fetch rawdata for each session
    sessions_data = {}
    for sid in session_ids:
        df = pd.read_sql(
            f"SELECT ax, ay, az, gx, gy, gz FROM rawdata WHERE sessionId = {sid};",
            conn
        )
        sessions_data[sid] = df
    
    conn.close()
    return sessions_data

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
