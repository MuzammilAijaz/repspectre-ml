#******************************************************************************
#  Convert SQL database to CSV dataset files
# -----------------------------------------------------------------------------
#  Extracts and organizes sensor data from SQLite database into structured
#  class folders:
#
#      rep_sessions/<lift_category>/          — full raw lift sessions per exercise
#      detected_rep_sessions/<lift_category>/ — rep-detected segments per exercise
#      noise_sessions/<motion_state>/         — background / noise sessions per state
#
#******************************************************************************

# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnknownArgumentType=false, reportMissingTypeStubs=false, reportArgumentType=false

import logging
import os
from typing import Final

if __name__ == "__main__":
    import sys
    from pathlib import Path
    PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
    SRC_DIR = PROJECT_ROOT / "src"
    sys.path.insert(0, str(SRC_DIR))

from lift_ml.data.database.repository.session_repository import SessionRepository
from lift_ml.data.domain.session import Session
from lift_ml.utils.rep_detection import RepDetectionResult, detect_rep_axis
from lift_ml.utils.sampling import calculate_sampling_rate

logger: Final = logging.getLogger(__name__)


def get_session_lift_category(session: Session) -> str:
    """Returns the lift category (e.g. 'FLOOR_PULL'), or 'UNKNOWN' if unavailable.

    Used to group rep_sessions and detected_rep_sessions by exercise type.
    """
    lift_cat = session.metadata.get("liftCategory")
    if lift_cat:
        return str(lift_cat)
    return "UNKNOWN"


def get_session_motion_state(session: Session) -> str:
    """Returns the motion state (e.g. 'SENSOR_DRIFT'), or 'UNKNOWN' if unavailable.

    Used to group noise_sessions by their recording state.
    """
    if session.motion_state:
        return str(session.motion_state)
    return "UNKNOWN"


def save_rep_sessions_to_csv(
    sessions: list[Session],
    target_folder: str,
    axes: list[str] | None = None,
) -> list[str]:
    """Exports full raw lift sessions (no segmentation) as CSV files.

    Groups by lift category:
        <target_folder>/<lift_category>/<session_id>.csv

    Returns:
        List of generated CSV file paths.
    """
    saved_paths: list[str] = []

    for session in sessions:
        df = session.sensor_data
        if df.empty:
            continue

        out_df = df.copy()
        if axes is not None:
            selected_cols = [col for col in axes if col in out_df.columns]
            out_df = out_df[selected_cols]

        category = get_session_lift_category(session)
        session_dir = os.path.join(target_folder, category)
        os.makedirs(session_dir, exist_ok=True)

        path = os.path.join(session_dir, f"{session.session_id}.csv")
        out_df.to_csv(path, index=False)
        saved_paths.append(path)
        logger.info("Saved rep session: %s", path)

    return saved_paths


def save_detected_rep_sessions_to_csv(
    sessions: list[Session],
    target_folder: str,
    axis: str = "ay",
    axes: list[str] | None = None,
) -> list[str]:
    """Applies rep detection to lift sessions and saves the extracted rep segments as CSV files.

    Groups by lift category:
        <target_folder>/<lift_category>/<session_id>.csv

    Calculates exact sampling rate per session. Raises ValueError if sampling rate is invalid.

    Returns:
        List of generated CSV file paths.
    """
    saved_paths: list[str] = []

    for session in sessions:
        df = session.sensor_data
        if df.empty:
            continue

        fs = calculate_sampling_rate(df)
        res: RepDetectionResult | None = detect_rep_axis(
            df,
            axis=axis,
            fs=int(round(fs)),
            baseline_seconds=1.0,
            k_start=24.0,
            k_end=5.0,
            smooth_window=5,
            min_duration=0.12,
        )

        if res is None:
            logger.warning("Skipping session %d: rep detection failed", session.session_id)
            continue

        rep_df = df.iloc[res.model_start_idx : res.model_end_idx].copy()
        if axes is not None:
            selected_cols = [col for col in axes if col in rep_df.columns]
            rep_df = rep_df[selected_cols]

        category = get_session_lift_category(session)
        session_dir = os.path.join(target_folder, category)
        os.makedirs(session_dir, exist_ok=True)

        path = os.path.join(session_dir, f"{session.session_id}.csv")
        rep_df.to_csv(path, index=False)
        saved_paths.append(path)
        logger.info("Saved detected rep session (%d Hz): %s", int(round(fs)), path)

    return saved_paths


def save_noise_sessions_to_csv(
    sessions: list[Session],
    target_folder: str,
    axes: list[str] | None = None,
) -> list[str]:
    """Exports noise / background sensor sessions as CSV files.

    Groups by motion state:
        <target_folder>/<motion_state>/<session_id>.csv

    Returns:
        List of generated CSV file paths.
    """
    saved_paths: list[str] = []

    for session in sessions:
        df = session.sensor_data
        if df.empty:
            continue

        out_df = df.copy()
        if axes is not None:
            selected_cols = [col for col in axes if col in out_df.columns]
            out_df = out_df[selected_cols]

        motion_state = get_session_motion_state(session)
        session_dir = os.path.join(target_folder, motion_state)
        os.makedirs(session_dir, exist_ok=True)

        path = os.path.join(session_dir, f"{session.session_id}.csv")
        out_df.to_csv(path, index=False)
        saved_paths.append(path)
        logger.info("Saved noise session: %s", path)

    return saved_paths


def export_sessions_for_analysis(
    db_path: str,
    output_root: str,
    lift_category: str | None = None,
    rep_dir: str = "rep_sessions",
    detected_rep_dir: str = "detected_rep_sessions",
    noise_dir: str = "noise_sessions",
    axes: list[str] | None = None,
    rep_axis: str = "ay",
) -> dict[str, int]:
    """
    Extracts and organizes database records into rep_sessions, detected_rep_sessions,
    and noise_sessions folders.

    Output structure:
        <output_root>/
            rep_sessions/
                <lift_category>/           e.g. FLOOR_PULL/, OVERHEAD_PRESS/
                    <session_id>.csv       full raw lift session
            detected_rep_sessions/
                <lift_category>/           e.g. FLOOR_PULL/, OVERHEAD_PRESS/
                    <session_id>.csv       rep-detected segment only
            noise_sessions/
                <motion_state>/            e.g. SENSOR_DRIFT/, BARBELL_ROLLING/
                    <session_id>.csv

    Returns:
        Dictionary mapping folder names to the count of exported files.
    """
    repo = SessionRepository(db_path)

    # Fetch lift sessions
    if lift_category is not None:
        lift_sessions = repo.get_lift_sessions_by_lift_category(lift_category)
    else:
        distinct_categories = repo.get_distinct_lift_categories()
        if distinct_categories:
            lift_sessions: list[Session] = []
            for cat in distinct_categories:
                lift_sessions.extend(repo.get_lift_sessions_by_lift_category(cat))
        else:
            lift_sessions = repo.get_sessions_by_motion_state("REP_START")

    # Fetch noise / background sessions
    all_motion_states = repo.get_distinct_motion_states()
    noise_motion_states = [ms for ms in all_motion_states if ms != "REP_START"]

    noise_sessions: list[Session] = []
    for ms in noise_motion_states:
        noise_sessions.extend(repo.get_sessions_by_motion_state(ms))

    # Export rep_sessions (full raw lift sessions, grouped by lift category)
    saved_reps = save_rep_sessions_to_csv(
        sessions=lift_sessions,
        target_folder=os.path.join(output_root, rep_dir),
        axes=axes,
    )

    # Export detected_rep_sessions (rep-detected, grouped by lift category)
    saved_detected = save_detected_rep_sessions_to_csv(
        sessions=lift_sessions,
        target_folder=os.path.join(output_root, detected_rep_dir),
        axis=rep_axis,
        axes=axes,
    )

    # Export noise_sessions (grouped by motion state)
    saved_noise = save_noise_sessions_to_csv(
        sessions=noise_sessions,
        target_folder=os.path.join(output_root, noise_dir),
        axes=axes,
    )

    summary = {
        rep_dir: len(saved_reps),
        detected_rep_dir: len(saved_detected),
        noise_dir: len(saved_noise),
    }

    logger.info(
        "Dataset organization complete. %d rep sessions, %d detected"
        "rep sessions, %d noise sessions.",
        len(saved_reps),
        len(saved_detected),
        len(saved_noise),
    )

    return summary


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    db_path_val = "data/sensor-database_06-09-2026_prototype_proper.db"
    output_root_val = "./data"
    export_sessions_for_analysis(db_path_val, output_root_val)
