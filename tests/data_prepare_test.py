import unittest
import os
import shutil
import sqlite3
import pandas as pd
import numpy as np
from lift_ml.data.prepare_dataset import save_detected_reps_to_csv, organize_sql_data
from lift_ml.data.ingestion import Session

class TestPrepareDataset(unittest.TestCase):
    def setUp(self):
        self.test_dir = "test_prepare_env"
        os.makedirs(self.test_dir, exist_ok=True)
        
    def tearDown(self):
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)
            
    def test_save_detected_reps_to_csv_extracts_and_saves_files(self):
        # Create synthetic data: 130 samples (1 sec) baseline + spike
        fs = 130
        # Baseline with very small noise to ensure noise_mad is non-zero but small
        baseline = np.random.normal(0, 1, fs)
        # Rep part: a significant movement
        rep_part = np.random.normal(0, 1, fs)
        rep_part[20:60] = 500.0 # Large movement
        
        ys = np.concatenate([baseline, rep_part])
        df = pd.DataFrame({
            'ax': np.random.normal(0, 1, len(ys)),
            'ay': ys,
            'az': np.random.normal(0, 1, len(ys)),
            'gx': np.random.normal(0, 1, len(ys)),
            'gy': np.random.normal(0, 1, len(ys)),
            'gz': np.random.normal(0, 1, len(ys))
        })
        
        session = Session(session_id=101, sensor_data=df)
        sessions = [session]
        
        output_folder = os.path.join(self.test_dir, "output")
        save_detected_reps_to_csv(sessions, output_folder, "prefix", fs=fs)
        
        # Check if file exists
        expected_path = os.path.join(output_folder, "prefix_101.csv")
        self.assertTrue(os.path.exists(expected_path), f"File {expected_path} was not created")
        
        # Verify content is a subset
        saved_df = pd.read_csv(expected_path)
        self.assertLess(len(saved_df), len(df))
        self.assertEqual(len(saved_df.columns), 6)

    def test_save_detected_reps_to_csv_skips_failed_detections(self):
        # Flat data should fail detection with high thresholds
        fs = 130
        df = pd.DataFrame(np.zeros((300, 6)), columns=['ax', 'ay', 'az', 'gx', 'gy', 'gz'])
        
        session = Session(session_id=202, sensor_data=df)
        sessions = [session]
        
        output_folder = os.path.join(self.test_dir, "output_empty")
        save_detected_reps_to_csv(sessions, output_folder, "prefix", fs=fs)
        
        # File should NOT exist because we "continue" on failed detection
        expected_path = os.path.join(output_folder, "prefix_202.csv")
        self.assertFalse(os.path.exists(expected_path))

    def test_organize_sql_data_integration(self):
        db_path = os.path.join(self.test_dir, "test.db")
        conn = sqlite3.connect(db_path)
        
        # Create tables
        conn.execute("CREATE TABLE session (sessionId INTEGER PRIMARY KEY, liftCategory TEXT, noise TEXT);")
        conn.execute("CREATE TABLE rawdata (sessionId INTEGER, ax REAL, ay REAL, az REAL, gx REAL, gy REAL, gz REAL);")
        
        # Add a barbell session
        conn.execute("INSERT INTO session (sessionId, liftCategory, noise) VALUES (1, 'FLOOR_PULL', NULL);")
        # Add dummy data (baseline + spike)
        fs = 130
        samples = fs * 2 # 2 seconds
        for i in range(samples):
            val = 500.0 if 150 < i < 180 else 0.0
            conn.execute(f"INSERT INTO rawdata (sessionId, ax, ay, az, gx, gy, gz) VALUES (1, 0, {val}, 0, 0, 0, 0);")
            
        # Add a noise session
        conn.execute("INSERT INTO session (sessionId, liftCategory, noise) VALUES (2, NULL, 'TALKING');")
        
        conn.commit()
        conn.close()
        
        output_root = os.path.join(self.test_dir, "raw_data")
        organize_sql_data(db_path, output_root)
        
        # Verify detect folder has barbell rep
        barbell_path = os.path.join(output_root, "detect", "barbell_1.csv")
        self.assertTrue(os.path.exists(barbell_path), f"Expected {barbell_path} to exist")
        
        # Verify CSV content
        df = pd.read_csv(barbell_path)
        self.assertGreater(len(df), 0)
        self.assertEqual(len(df.columns), 6)

if __name__ == "__main__":
    unittest.main()
