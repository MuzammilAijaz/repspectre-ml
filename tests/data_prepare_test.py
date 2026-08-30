# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnknownArgumentType=false, reportMissingTypeStubs=false, reportAttributeAccessIssue=false

import os
import shutil
import sqlite3
import unittest

import numpy as np
import pandas as pd

from lift_ml.data.domain.session import create_test_session
from lift_ml.data.prepare_dataset import (
    calculate_sampling_rate,
    export_sessions_for_analysis,
    save_detected_rep_sessions_to_csv,
    save_noise_sessions_to_csv,
    save_rep_sessions_to_csv,
)


class TestPrepareDataset(unittest.TestCase):
    def setUp(self) -> None:
        self.test_dir = "test_prepare_env"
        os.makedirs(self.test_dir, exist_ok=True)

    def tearDown(self) -> None:
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)

    def test_calculate_sampling_rate_from_timestamps_us(self) -> None:
        timestamps = np.linspace(0, 1_000_000, 101, dtype=np.int64)
        df = pd.DataFrame({"ax": np.zeros(101), "timestampUs": timestamps})
        session = create_test_session(session_id=1, sensor_data=df)
        fs = calculate_sampling_rate(session)
        self.assertAlmostEqual(fs, 100.0, places=2)

    def test_calculate_sampling_rate_from_metadata(self) -> None:
        df = pd.DataFrame({"ax": np.zeros(101)})
        session = create_test_session(
            session_id=1,
            sensor_data=df,
            metadata={"startTime": 1000, "endTime": 2000},
        )
        fs = calculate_sampling_rate(session)
        self.assertAlmostEqual(fs, 100.0, places=2)

    def test_calculate_sampling_rate_raises_on_missing_timestamps(self) -> None:
        df = pd.DataFrame({"ax": np.zeros(50)})
        session = create_test_session(session_id=1, sensor_data=df)
        with self.assertRaises(ValueError) as ctx:
            calculate_sampling_rate(session)
        self.assertIn("lacks both 'timestampUs'", str(ctx.exception))

    def test_calculate_sampling_rate_raises_on_non_positive_duration(self) -> None:
        df = pd.DataFrame({
            "ax": np.zeros(5),
            "timestampUs": [1000, 1000, 1000, 1000, 1000],
        })
        session = create_test_session(session_id=1, sensor_data=df)
        with self.assertRaises(ValueError) as ctx:
            calculate_sampling_rate(session)
        self.assertIn("invalid duration", str(ctx.exception))

    def test_save_rep_sessions_to_csv_groups_by_lift_category(self) -> None:
        df = pd.DataFrame({
            "ax": np.ones(50),
            "ay": np.zeros(50),
            "az": np.ones(50),
        })
        session = create_test_session(
            session_id=201,
            sensor_data=df,
            metadata={"liftCategory": "FLOOR_PULL"},
        )
        output_folder = os.path.join(self.test_dir, "rep_out")
        saved = save_rep_sessions_to_csv([session], output_folder, axes=["ax", "ay"])

        self.assertEqual(len(saved), 1)
        expected_path = os.path.join(output_folder, "FLOOR_PULL", "201.csv")
        self.assertTrue(os.path.exists(expected_path))
        saved_df = pd.read_csv(expected_path)
        self.assertEqual(list(saved_df.columns), ["ax", "ay"])
        self.assertEqual(len(saved_df), 50)

    def test_save_detected_rep_sessions_to_csv_groups_by_lift_category(self) -> None:
        fs = 130
        baseline = np.random.normal(0, 1, fs)
        rep_part = np.random.normal(0, 1, fs)
        rep_part[20:60] = 500.0
        ys = np.concatenate([baseline, rep_part])
        df = pd.DataFrame({
            "ax": np.random.normal(0, 1, len(ys)),
            "ay": ys,
            "az": np.random.normal(0, 1, len(ys)),
            "gx": np.random.normal(0, 1, len(ys)),
            "gy": np.random.normal(0, 1, len(ys)),
            "gz": np.random.normal(0, 1, len(ys)),
            "timestampUs": np.linspace(0, 2_000_000, len(ys), dtype=np.int64),
        })
        session = create_test_session(
            session_id=101,
            sensor_data=df,
            metadata={"liftCategory": "FLOOR_PULL"},
        )

        output_folder = os.path.join(self.test_dir, "detected_out")
        saved = save_detected_rep_sessions_to_csv(
            [session],
            output_folder,
            axes=["ax", "ay", "az", "gx", "gy", "gz"],
        )

        self.assertEqual(len(saved), 1)
        expected_path = os.path.join(output_folder, "FLOOR_PULL", "101.csv")
        self.assertTrue(os.path.exists(expected_path))
        saved_df = pd.read_csv(expected_path)
        self.assertLess(len(saved_df), len(df))
        self.assertEqual(len(saved_df.columns), 6)

    def test_save_detected_rep_sessions_to_csv_skips_failed_detection(self) -> None:
        fs = 130
        timestamps = np.linspace(0, 2_000_000, fs * 2, dtype=np.int64)
        df = pd.DataFrame({
            "ax": np.zeros(fs * 2),
            "ay": np.zeros(fs * 2),  # flat — detection must fail
            "timestampUs": timestamps,
        })
        session = create_test_session(
            session_id=202,
            sensor_data=df,
            metadata={"liftCategory": "FLOOR_PULL"},
        )
        output_folder = os.path.join(self.test_dir, "detected_out_empty")
        saved = save_detected_rep_sessions_to_csv([session], output_folder)
        self.assertEqual(len(saved), 0)

    def test_save_noise_sessions_to_csv_groups_by_motion_state(self) -> None:
        df = pd.DataFrame({"ax": np.ones(50), "ay": np.ones(50), "az": np.ones(50)})
        session = create_test_session(
            session_id=303,
            sensor_data=df,
            motion_state="SENSOR_DRIFT",
        )
        output_folder = os.path.join(self.test_dir, "noise_out")
        saved = save_noise_sessions_to_csv([session], output_folder, axes=["ax", "ay", "az"])

        self.assertEqual(len(saved), 1)
        expected_path = os.path.join(output_folder, "SENSOR_DRIFT", "303.csv")
        self.assertTrue(os.path.exists(expected_path))
        saved_df = pd.read_csv(expected_path)
        self.assertEqual(list(saved_df.columns), ["ax", "ay", "az"])
        self.assertEqual(len(saved_df), 50)

    def test_export_sessions_for_analysis_creates_all_three_folders(self) -> None:
        db_path = os.path.join(self.test_dir, "test.db")
        conn = sqlite3.connect(db_path)

        # Create artificial schema matching the sensor database
        # WARN: hardcoded and requires changing test on schema change

        conn.execute(
            """CREATE TABLE session (
                sessionId INTEGER PRIMARY KEY AUTOINCREMENT,
                startTime INTEGER NOT NULL,
                endTime INTEGER,
                motionState TEXT NOT NULL,
                sensorDataFormat TEXT NOT NULL
            );"""
        )
        conn.execute(
            """CREATE TABLE lift_context (
                sessionId INTEGER PRIMARY KEY,
                liftCategory TEXT NOT NULL,
                tempo TEXT NOT NULL,
                rpe INTEGER,
                FOREIGN KEY(sessionId) REFERENCES session(sessionId)
            );"""
        )
        conn.execute(
            """CREATE TABLE full_imu_raw (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                sessionId INTEGER NOT NULL,
                ax REAL NOT NULL,
                ay REAL NOT NULL,
                az REAL NOT NULL,
                gx REAL NOT NULL,
                gy REAL NOT NULL,
                gz REAL NOT NULL,
                qx REAL NOT NULL,
                qy REAL NOT NULL,
                qz REAL NOT NULL,
                qw REAL NOT NULL,
                timestampUs INTEGER NOT NULL,
                FOREIGN KEY(sessionId) REFERENCES session(sessionId)
            );"""
        )

        # Positive: FLOOR_PULL lift session
        conn.execute(
            """INSERT INTO session (sessionId, startTime, endTime, motionState, sensorDataFormat)
               VALUES (1, 1000, 3000, 'REP_START', 'FULL_IMU_RAW');"""
        )
        conn.execute(
            """INSERT INTO lift_context (sessionId, liftCategory, tempo, rpe)
               VALUES (1, 'FLOOR_PULL', 'NORMAL', 1);"""
        )
        fs = 130
        for i in range(fs * 2):
            val = 500.0 if 150 < i < 180 else 0.0
            conn.execute(
                """INSERT INTO full_imu_raw (sessionId, ax, ay, az, gx, gy, gz, qx, qy, qz, qw, timestampUs)
                   VALUES (1, 0, ?, 0, 0, 0, 0, 0, 0, 0, 1, ?);""",
                (val, int(i * (1_000_000 / fs))),
            )

        # Negative: SENSOR_DRIFT noise session
        conn.execute(
            """INSERT INTO session (sessionId, startTime, endTime, motionState, sensorDataFormat)
               VALUES (2, 4000, 5000, 'SENSOR_DRIFT', 'FULL_IMU_RAW');"""
        )
        for i in range(50):
            conn.execute(
                """INSERT INTO full_imu_raw (sessionId, ax, ay, az, gx, gy, gz, qx, qy, qz, qw, timestampUs)
                   VALUES (2, 0.1, 0.1, 0.98, 0, 0, 0, 0, 0, 0, 1, ?);""",
                (int(i * (1_000_000 / fs)),),
            )

        conn.commit()
        conn.close()

        output_root = os.path.join(self.test_dir, "data")
        summary = export_sessions_for_analysis(
            db_path=db_path,
            output_root=output_root,
            axes=["ax", "ay", "az"],
        )

        self.assertEqual(summary["rep_sessions"], 1)
        self.assertEqual(summary["detected_rep_sessions"], 1)
        self.assertEqual(summary["noise_sessions"], 1)

        # rep_sessions/FLOOR_PULL/1.csv — full raw lift session
        rep_path = os.path.join(output_root, "rep_sessions", "FLOOR_PULL", "1.csv")
        self.assertTrue(os.path.exists(rep_path))
        df_rep = pd.read_csv(rep_path)
        self.assertEqual(list(df_rep.columns), ["ax", "ay", "az"])
        self.assertEqual(len(df_rep), fs * 2)  # full session, not trimmed

        # detected_rep_sessions/FLOOR_PULL/1.csv — rep-detected segment only
        detected_path = os.path.join(output_root, "detected_rep_sessions", "FLOOR_PULL", "1.csv")
        self.assertTrue(os.path.exists(detected_path))
        df_detected = pd.read_csv(detected_path)
        self.assertEqual(list(df_detected.columns), ["ax", "ay", "az"])
        self.assertLess(len(df_detected), fs * 2)  # trimmed to rep segment

        # noise_sessions/SENSOR_DRIFT/2.csv
        noise_path = os.path.join(output_root, "noise_sessions", "SENSOR_DRIFT", "2.csv")
        self.assertTrue(os.path.exists(noise_path))
        df_noise = pd.read_csv(noise_path)
        self.assertEqual(list(df_noise.columns), ["ax", "ay", "az"])
        self.assertEqual(len(df_noise), 50)


if __name__ == "__main__":
    unittest.main()
