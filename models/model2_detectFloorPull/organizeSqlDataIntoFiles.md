---
jupyter:
  jupytext:
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.19.3
  kernelspec:
    display_name: Python (tfmicro)
    language: python
    name: tfmicro
---

```python
SESSION_DURATION = 10 # 10 seconds
DETECTION_DURATION = 0.8 # refer to rep_detection for this.
```

```python
# =====================================================================================
# |                                 PURPOSE
# -------------------------------------------------------------------------------------
# Convert the SQL database into CSV files stored inside "detect" and "none" folders.
# Each session = 1 CSV example.
# =====================================================================================

import os
import sqlite3
import pandas as pd
from rep_detection import *

# -------------------------
# Parameters
# -------------------------
db_path = "sensor_database.db"
barbell_type = "FLOOR_PULL"  # the main barbell sessions
axes = ['ax', 'ay', 'az', 'gx', 'gy', 'gz']

output_root = "./raw_data"
barbell_dir = os.path.join(output_root, "detect")
noise_dir = os.path.join(output_root, "none")

# -------------------------
# Helper function to fetch all sessions for a noise type
# -------------------------
def get_all_sessions_data(db_path, classType, subtype):
    conn = sqlite3.connect(db_path)
    
    # All session IDs for the noise type
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

```

# extract Detection data

```python
# -------------------------
# Load all sessions
# -------------------------
barbell_sessions = get_all_sessions_data(db_path, "liftCategory", barbell_type)
print(len(barbell_sessions))

first_sid, first_df = list(barbell_sessions.items())[0]

# Calculate sampling rate from number of rows (samples) and session duration
fs = len(first_df) / SESSION_DURATION  # if session duration is 10 seconds
print("Estimated sampling rate:", fs, "Hz")

# -------------------------
# extract the reps from the raw data and save them
# -------------------------
def save_detected_reps_to_csv(sessions_dict, target_folder, prefix, axis='ay'):
    """
    Extracts the rep from each session using detect_rep_axis and saves only the rep portion.
    """
    for session_id, df in sessions_dict.items():
        # Detect rep start/end indices
        start_idx, end_idx, ys_s, base_med, start_th, end_th, model_start_idx, model_end_idx = detect_rep_axis(
            df, axis=axis, fs=fs,
            baseline_seconds=1.0,
            k_start=24.0, k_end=5.0,
            smooth_window=5,
            min_duration=0.12
        )

        # Extract only the rep portion
        if model_start_idx is not None and model_end_idx is not None:
            rep_df = df.iloc[model_start_idx:model_end_idx].copy()
        else:
            # fallback: save entire session if detection fails
            continue
            # rep_df = df.copy()

        # Save to CSV
        filename = f"{prefix}_{session_id}.csv"
        os.makedirs(target_folder, exist_ok=True)
        path = os.path.join(target_folder, filename)
        rep_df.to_csv(path, index=False)
        print("Saved detected rep:", path)

# Save barbell reps
save_detected_reps_to_csv(barbell_sessions, barbell_dir, "barbell", axis='ay')

print("Done exporting CSV dataset!")
```

# Save Noise data

```python
def get_column_names(db_path, column_name, table_name):
    conn = sqlite3.connect(db_path)
    
    df = pd.read_sql(f"SELECT DISTINCT {column_name} FROM {table_name};", conn)
    values_list = df[column_name].tolist()
    
    conn.close()
    return values_list
```

```python
noise_types = get_column_names(db_path, "noise", "session")
if None in noise_types:
    noise_types.remove(None)

print(noise_types)
```

```python
def get_all_type_sessions_data(db_path, classTypes, subTypes):
    conn = sqlite3.connect(db_path)

    sessions_data_for_each_type = [] 

    for classType in classTypes:
        for subType in subTypes:
            # All session IDs for the noise type
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
                sessions_data_for_each_type.append(sessions_data)
    
    conn.close()
    return sessions_data_for_each_type # list of dictionary
```

```python
data = get_all_type_sessions_data(db_path, ['noise'], noise_types)
```
