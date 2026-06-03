#******************************************************************************
#  Convert SQL database to csv files 
# -----------------------------------------------------------------------------
#  Convert the SQL database into CSV files stored inside "detect" and
#  "none" folders where each session = 1 CSV example.
#******************************************************************************

import os
import sqlite3
import pandas as pd
from lift_ml.utils.rep_detection import *
from lift_ml.data.ingestion import get_all_sessions_data, get_column_names,\
    get_all_type_sessions_data, Session

SESSION_DURATION = 10 # 10 seconds
DETECTION_DURATION = 0.8 # refer to rep_detection for this.

def save_detected_reps_to_csv(sessions: Session, target_folder, 
                              prefix, fs, axis='ay'):
    """
    Extracts the rep from each session using detect_rep_axis and
    saves only the rep portion.
    """
    for session in sessions:
        df = session.sensor_data
        session_id = session.session_id
        # Detect rep start/end indices
        start_idx, end_idx, ys_s, base_med, start_th, end_th,\
            model_start_idx, model_end_idx = detect_rep_axis(
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

def organize_sql_data(db_path, output_root, barbell_type="FLOOR_PULL", 
                      axes=['ax', 'ay', 'az', 'gx', 'gy', 'gz']):
    barbell_dir = os.path.join(output_root, "detect")
    noise_dir = os.path.join(output_root, "none")

    # Load all sessions
    barbell_sessions: List[Session] = get_all_sessions_data(
            db_path, "liftCategory", barbell_type)
    print(len(barbell_sessions))

    if not barbell_sessions:
        print("No barbell sessions found.")
        return

    first_session = barbell_sessions[0]
    first_df = first_session.sensor_data

    # Calculate sampling rate from number of rows (samples) and session duration
    fs = len(first_df) / SESSION_DURATION  # if session duration is 10 seconds
    print("Estimated sampling rate:", fs, "Hz")

    # Save barbell reps
    save_detected_reps_to_csv(barbell_sessions, barbell_dir, "barbell", 
                              fs=fs, axis='ay')

    # Save Noise data
    noise_types = get_column_names(db_path, "noise", "session")
    if None in noise_types:
        noise_types.remove(None)
    
    print("Noise types:", noise_types)
    print("Done exporting CSV dataset!")

if __name__ == "__main__":
    db_path = "sensor_database.db"
    output_root = "./raw_data"
    organize_sql_data(db_path, output_root)
