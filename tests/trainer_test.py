import os
import shutil
import unittest
import numpy as np
import pandas as pd
import tensorflow as tf
from lift_ml.training.trainer import Trainer
from lift_ml.config import Config, DataConfig, ModelConfig
from lift_ml.models.model_builder import get_model

class TestTrain(unittest.TestCase):

    def setUp(self):
        self.base_dir = "test_train_env"
        self.data_dir = os.path.join(self.base_dir, "data")
        self.output_dir = os.path.join(self.base_dir, "output")
        
        # Create directories
        for subset in ["train", "valid", "test"]:
            for cls in ["barbell", "none"]:
                os.makedirs(os.path.join(self.data_dir, subset, cls), exist_ok=True)
                # Create dummy CSV data (2 columns)
                df = pd.DataFrame(np.random.rand(10, 2), columns=["ax", "ay"])
                df.to_csv(os.path.join(self.data_dir, subset, cls, "data.csv"), index=False)

        self.config = Config(
            data=DataConfig(
                train_path=os.path.join(self.data_dir, "train"),
                valid_path=os.path.join(self.data_dir, "valid"),
                test_path=os.path.join(self.data_dir, "test"),
                seq_length=5,
                data_dimension=2,
                labels=["barbell", "none"]
            ),
            model=ModelConfig(
                type="cnn",
                epochs=1,
                batch_size=2
            ),
            output_dir=self.output_dir
        )

    def tearDown(self):
        if os.path.exists(self.base_dir):
            shutil.rmtree(self.base_dir)

    def test_get_model_factory_returns_valid_cnn_model(self):
        model = get_model(self.config)
        self.assertIsInstance(model, tf.keras.Model)
        # Check input shape for CNN
        self.assertEqual(model.input_shape, (None, 5, 2, 1))

    def test_trainer_train_smoke_runs_full_pipeline_without_crash(self):
        trainer = Trainer(self.config)
        # Just run 1 epoch to ensure no crashes
        trainer.train()
        
        # Check if model files were exported
        self.assertTrue(os.path.exists(os.path.join(self.output_dir, "model.keras")))
        self.assertTrue(os.path.exists(os.path.join(self.output_dir, "model.tflite")))
        self.assertTrue(os.path.exists(os.path.join(self.output_dir, "model_quantized.tflite")))

    def test_reshape_function_applies_cnn_transformation_correctly(self):
        trainer = Trainer(self.config)
        data = tf.random.uniform((5, 2))
        label = tf.constant(1)
        
        reshaped_data, reshaped_label = trainer.reshape_function(data, label)
        self.assertEqual(reshaped_data.shape, (5, 2, 1))
        self.assertEqual(reshaped_label, label)

    def test_load_data_augmentation_toggle_increases_sample_count(self):
        trainer = Trainer(self.config)
        loader_no_aug = trainer.load_data(augment=False)
        loader_aug = trainer.load_data(augment=True)
        # 2 samples * 2 (padded) = 4
        self.assertEqual(loader_no_aug.train_len, 4)
        # Augmentation increases it significantly (each sample -> ~25 samples)
        self.assertGreater(loader_aug.train_len, loader_no_aug.train_len)

    def test_reshape_function_for_lstm_keeps_data_2d(self):
        self.config.model.type = "lstm"
        trainer = Trainer(self.config)
        data = tf.random.uniform((5, 2))
        label = tf.constant(1)
        
        reshaped_data, reshaped_label = trainer.reshape_function(data, label)
        # LSTM expects (seq_length, dim), so no extra dimension
        self.assertEqual(reshaped_data.shape, (5, 2))
        self.assertEqual(reshaped_label, label)

    def test_trainer_train_generates_tensorboard_logs(self):
        # Ensure log directory is clean or we check a specific one
        trainer = Trainer(self.config)
        trainer.train()
        
        # Trainer sets self.logdir = os.path.join("logs/scalars/", ...)
        self.assertTrue(os.path.exists(trainer.logdir))
        self.assertTrue(len(os.listdir(trainer.logdir)) > 0)

    def test_export_model_creates_missing_directory(self):
        custom_output = os.path.join(self.base_dir, "new_dir/deep_path")
        self.config.output_dir = custom_output
        trainer = Trainer(self.config)
        
        # Build a small dummy model to save
        model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(5, 2, 1)),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(2, activation="softmax")
        ])
        
        trainer.export_model(model)
        self.assertTrue(os.path.exists(custom_output))
        self.assertTrue(os.path.exists(os.path.join(custom_output, "model.keras")))

if __name__ == "__main__":
    unittest.main()
