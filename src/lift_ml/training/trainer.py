import os
import datetime
import numpy as np
import tensorflow as tf
from typing import Tuple

from lift_ml.config import Config
from lift_ml.data.loader import DataLoader
from lift_ml.models.model_builder import get_model

class Trainer:
    def __init__(self, config: Config):
        self.config = config
        self.logdir = os.path.join("logs/scalars/", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
        self.tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=self.logdir)

    def reshape_function(self, data, label):
        """For CNN: reshape to (seq_length, dimension, 1)"""
        if self.config.model.type.upper() == "CNN":
            data = tf.reshape(data, [-1, self.config.data.data_dimension, 1])
        return data, label

    def calculate_model_size(self, model):
        model.summary()
        var_sizes = [
            np.prod(list(map(int, v.shape))) * tf.dtypes.as_dtype(v.dtype).size
            for v in model.trainable_variables
        ]
        print("Model size: %.2f KB" % (sum(var_sizes) / 1024.0))

    def load_data(self, augment: bool = False):
        loader = DataLoader(self.config.data, augment_train=augment)
        loader.format()
        
        # Shuffle
        loader.train_data = loader.train_data.shuffle(buffer_size=loader.train_len)
        loader.valid_data = loader.valid_data.shuffle(buffer_size=loader.valid_len)
        loader.test_data = loader.test_data.shuffle(buffer_size=loader.test_len)
        
        return loader

    def train(self):
        print("Loading data...")
        loader = self.load_data(augment=True)
        
        print("Building model...")
        model = get_model(self.config)
        self.calculate_model_size(model)

        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=self.config.model.learning_rate),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"]
        )

        train_ds = loader.train_data.map(self.reshape_function)
        valid_ds = loader.valid_data.map(self.reshape_function)
        test_ds = loader.test_data.map(self.reshape_function)

        batch_size = self.config.model.batch_size
        train_ds = train_ds.batch(batch_size).repeat()
        valid_ds = valid_ds.batch(batch_size)
        test_ds = test_ds.batch(batch_size)

        steps_per_epoch = max(1, int((loader.train_len - 1) / batch_size) + 1)
        validation_steps = max(1, int((loader.valid_len - 1) / batch_size) + 1)

        model.fit(
            train_ds,
            epochs=self.config.model.epochs,
            validation_data=valid_ds,
            steps_per_epoch=steps_per_epoch,
            validation_steps=validation_steps,
            callbacks=[self.tensorboard_callback]
        )

        # Evaluate
        test_steps = max(1, int((loader.test_len - 1) / batch_size) + 1)
        loss, acc = model.evaluate(test_ds, steps=test_steps)
        print(f"Test Loss: {loss:.4f}  Test Accuracy: {acc:.3f}")

        self.export_model(model)

    def export_model(self, model):
        output_dir = self.config.output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        keras_path = os.path.join(output_dir, "model.keras")
        model.save(keras_path)
        print(f"Saved Keras model to {keras_path}")

        # Float TFLite
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        tflite_model = converter.convert()
        tflite_path = os.path.join(output_dir, "model.tflite")
        with open(tflite_path, "wb") as f:
            f.write(tflite_model)
        print(f"Saved TFLite model to {tflite_path}")

        # Quantized TFLite
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        tflite_model_q = converter.convert()
        tflite_q_path = os.path.join(output_dir, "model_quantized.tflite")
        with open(tflite_q_path, "wb") as f:
            f.write(tflite_model_q)
        print(f"Saved Quantized TFLite model to {tflite_q_path}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config file")
    args = parser.parse_args()

    config = Config.from_yaml(args.config)
    trainer = Trainer(config)
    trainer.train()
