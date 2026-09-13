# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnknownArgumentType=false, reportMissingTypeStubs=false, reportAttributeAccessIssue=false, reportArgumentType=false

import datetime
import logging
import os
from collections.abc import Generator
from typing import Final

import numpy as np

if __name__ == "__main__":
    import sys
    from pathlib import Path
    PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
    SRC_DIR = PROJECT_ROOT / "src"
    sys.path.insert(0, str(SRC_DIR))

from lift_ml.config import Config
from lift_ml.data.loader import DataLoader
from lift_ml.data.prepare_dataset import export_sessions_for_analysis
from lift_ml.data.split import split_data
from lift_ml.models.c_array_exporter import export_tflite_to_c_array
from lift_ml.models.model_builder import get_model
from lift_ml.utils.gpu import setup_gpu_environment

setup_gpu_environment()

import tensorflow as tf  # noqa: E402

logger: Final = logging.getLogger("lift_ml.training.trainer")


class Trainer:
    def __init__(self, config: Config) -> None:
        self.config = config
        now_str = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        self.logdir = os.path.join("logs/scalars/", now_str)
        self.tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=self.logdir)

    def reshape_function(self, data: object, label: object) -> tuple[object, object]:
        """For CNN: reshape to (seq_length, dimension, 1)"""
        if self.config.model.type.upper() == "CNN":
            data = tf.reshape(data, [-1, self.config.data.data_dimension, 1])
        return data, label

    def calculate_model_size(self, model: object) -> None:
        model.summary()
        var_sizes = [
            int(np.prod(list(map(int, v.shape)))) * tf.dtypes.as_dtype(v.dtype).size
            for v in model.trainable_variables
        ]
        print("Model size: %.2f KB" % (sum(var_sizes) / 1024.0))

    def load_data(self, augment: bool = False) -> DataLoader:
        loader = DataLoader(self.config.data, augment_train=augment)
        loader.format()

        # Shuffle
        loader.train_data = loader.train_data.shuffle(buffer_size=loader.train_len)
        loader.valid_data = loader.valid_data.shuffle(buffer_size=loader.valid_len)
        loader.test_data = loader.test_data.shuffle(buffer_size=loader.test_len)

        logger.info("Train samples:      %d", loader.train_len)
        logger.info("Validation samples: %d", loader.valid_len)
        logger.info("Test samples:       %d", loader.test_len)

        return loader

    def train(self) -> None:
        print("Loading data...")
        loader = self.load_data(augment=True)

        print("Building model...")
        model = get_model(self.config)
        self.calculate_model_size(model)

        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=self.config.model.learning_rate),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
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
            callbacks=[self.tensorboard_callback],
        )

        # Evaluate
        test_steps = max(1, int((loader.test_len - 1) / batch_size) + 1)
        loss, acc = model.evaluate(test_ds, steps=test_steps)
        print(f"Test Loss: {loss:.4f}  Test Accuracy: {acc:.3f}")

        # Export Model
        self.export_model(model, loader=loader)

    def export_model(self, model: object, loader: DataLoader | None = None) -> None:
        output_dir = self.config.output_dir
        os.makedirs(output_dir, exist_ok=True)

        keras_path = os.path.join(output_dir, "model.keras")
        model.save(keras_path)
        print(f"Saved Keras model to {keras_path}")

        #----- Float export -------------------------------------------

        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        tflite_model = converter.convert()
        tflite_path = os.path.join(output_dir, "model.tflite")
        with open(tflite_path, "wb") as f:
            f.write(tflite_model)
        print(f"Saved TFLite model to {tflite_path} (Size: {len(tflite_model):,} bytes)")

        #----- INT8 export --------------------------------------------
        # Full Integer Quantized TFLite (INT8 weights AND activations)

        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]

        # TensorFlow Lite determine how to quantize the model's activations from float32 to INT8.
        if loader is not None:

            # ASSUMPTION: This is run after train() which does load_data() to shuffle the dataset.
            # Take first 100 "sessions" of the train data to act as the representative dataset,
            # in order to perform quantization.
            def representative_dataset_gen() -> Generator[list[object]]:
                for x, _ in loader.train_data.map(self.reshape_function).take(100):
                    yield [tf.cast(tf.expand_dims(x, axis=0), tf.float32)]

            converter.representative_dataset = representative_dataset_gen
            converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]

        tflite_model_q = converter.convert()
        tflite_q_path = os.path.join(output_dir, "model_quantized.tflite")
        with open(tflite_q_path, "wb") as f:
            f.write(tflite_model_q)
        tflite_q_size = len(tflite_model_q)
        print(
            f"Saved Full INT8 Quantized TFLite model to {tflite_q_path}"
            f" (Size: {tflite_q_size:,} bytes)"
        )

        #----- C array export -----------------------------------------

        header_path, source_path = export_tflite_to_c_array(
            tflite_path=tflite_q_path,
            output_dir=output_dir,
            array_name="g_model_data",
            header_filename="model_data.h",
            source_filename="model_data.c",
        )
        print(f"Saved ESP-IDF C array to {header_path} and {source_path}")


if __name__ == "__main__":
    import argparse
    import shutil
    from pathlib import Path

    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config file")
    parser.add_argument("--db", type=str, default="data/repspectre.db", help="Path to SQLite DB")
    parser.add_argument("--data-root", type=str, default="data", help="Output root for CSV export")
    args = parser.parse_args()

    config = Config.from_yaml(args.config)

    # Remove previous folders
    folders = [Path("data/train"), Path("data/valid"), Path("data/test")]
    for folder in folders:
        if folder.exists():
            shutil.rmtree(folder)
            logger.info("Removed %s", folder)

    export_sessions_for_analysis(db_path=args.db, output_root=args.data_root)

    split_data(
        config=config.data,
        class_a="data/detected_rep_sessions/FLOOR_PULL",
        class_b="data/noise_sessions",
        class_a_collapse=False,
        class_b_collapse=True,
        train_ratio=0.6,
        valid_ratio=0.2,
    )

    trainer = Trainer(config)
    trainer.train()
