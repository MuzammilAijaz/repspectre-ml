# train_barbell_fixed.py
# Adapted for 2-class (barbell vs none), single-user, fixed sequence length, safe TFLite export

from __future__ import absolute_import, division, print_function
import argparse
import datetime
import os
import numpy as np
import tensorflow as tf

from data_load import DataLoader  # Your fixed DataLoader for 6-axis input

EPOCHS = 250
BATCH_SIZE = 64
DIMENSION = 2
# -----------------------
# Logging / TensorBoard
# -----------------------
logdir = "logs/scalars/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir)

# -----------------------
# Helpers
# -----------------------
def reshape_function(data, label):
    """For CNN: reshape to (seq_length, DIMENSION, 1)"""
    data = tf.reshape(data, [-1, DIMENSION, 1])
    return data, label

def calculate_model_size(model):
    model.summary()
    var_sizes = [
        np.product(list(map(int, v.shape))) * v.dtype.size
        for v in model.trainable_variables
    ]
    print("Model size: %.2f KB" % (sum(var_sizes) / 1024.0))

# -----------------------
# Model builders
# -----------------------
def build_cnn(seq_length):
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(8, (5, 3), padding="same", activation="relu",
                               input_shape=(seq_length, DIMENSION, 1)),
        tf.keras.layers.MaxPool2D((4, 1), padding="same"),
        tf.keras.layers.Dropout(0.15),
        tf.keras.layers.Conv2D(16, (4, 1), padding="same", activation="relu"),
        tf.keras.layers.MaxPool2D((4, 1), padding="same"),
        tf.keras.layers.Dropout(0.15),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(32, activation="relu"),
        tf.keras.layers.Dropout(0.15),
        tf.keras.layers.Dense(2, activation="softmax")
    ])
    print("Built small CNN.")
    return model

def build_lstm(seq_length):
    batch_size = BATCH_SIZE 
    model = tf.keras.Sequential([
        tf.keras.layers.Bidirectional(
            tf.keras.layers.LSTM(22, return_sequences=False, batch_input_shape=(batch_size, seq_length, 3))
        ),
        tf.keras.layers.Dense(4, activation="sigmoid")
    ])

    model.build(input_shape=(BATCH_SIZE,seq_length,3))
    print("Built small LSTM.")
    return model

# -----------------------
# Data loading wrapper
# -----------------------
def load_data(train_data_path, valid_data_path, test_data_path, seq_length):
    data_loader = DataLoader(
        train_data_path, valid_data_path, test_data_path, seq_length=seq_length, augment_train=False)
    data_loader.format()
    # Shuffle datasets to avoid order bias
    data_loader.train_data = data_loader.train_data.shuffle(buffer_size=data_loader.train_len)
    data_loader.valid_data = data_loader.valid_data.shuffle(buffer_size=data_loader.valid_len)
    data_loader.test_data = data_loader.test_data.shuffle(buffer_size=data_loader.test_len)
    return (data_loader.train_len, data_loader.train_data,
            data_loader.valid_len, data_loader.valid_data,
            data_loader.test_len, data_loader.test_data)

# -----------------------
# Training loop
# -----------------------
def train_net(model,
              train_len,
              train_data,
              valid_len,
              valid_data,
              test_len,
              test_data,
              kind):
    calculate_model_size(model)

    epochs = EPOCHS
    batch_size = BATCH_SIZE

    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    if kind.upper() == "CNN":
        train_data = train_data.map(reshape_function)
        valid_data = valid_data.map(reshape_function)
        test_data = test_data.map(reshape_function)

    # Prepare test labels
    test_labels = np.zeros(test_len, dtype=np.int32)
    for idx, (data, label) in enumerate(test_data):
        if idx >= test_len:
            break
        test_labels[idx] = int(label.numpy())

    train_data_batched = train_data.batch(batch_size).repeat()
    valid_data_batched = valid_data.batch(batch_size)
    test_data_batched = test_data.batch(batch_size)

    steps_per_epoch = max(1, int((train_len - 1) / batch_size) + 1)
    validation_steps = max(1, int((valid_len - 1) / batch_size) + 1)
    test_steps = max(1, int((test_len - 1) / batch_size) + 1)

    model.fit(
        train_data_batched,
        epochs=epochs,
        validation_data=valid_data_batched,
        steps_per_epoch=steps_per_epoch,
        validation_steps=validation_steps,
        callbacks=[tensorboard_callback]
    )

    # Evaluate
    loss, acc = model.evaluate(test_data_batched, steps=test_steps)
    print("Test Loss: {:.4f}  Test Accuracy: {:.3f}".format(loss, acc))

    # Predictions & confusion matrix
    preds = np.argmax(model.predict(test_data_batched, steps=test_steps), axis=1)
    confusion = tf.math.confusion_matrix(
        labels=tf.constant(test_labels),
        predictions=tf.constant(preds),
        num_classes=2
    )
    print("Confusion matrix:\n", confusion.numpy())

    # Save Keras and TFLite models
    os.makedirs("model", exist_ok=True)
    model.save("model/barbell_model.keras")  # no include_optimizer

    # Float TFLite
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()
    open("model/barbell_model.tflite", "wb").write(tflite_model)

    # Quantized TFLite
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.OPTIMIZE_FOR_SIZE]
    tflite_model_q = converter.convert()
    open("model/barbell_model_quantized.tflite", "wb").write(tflite_model_q)

    print("Saved TFLite models:")
    print(" - barbell_model.tflite (float)")
    print(" - barbell_model_quantized.tflite (quantized)")
    print("Done training and export.")

# -----------------------
# CLI / Main
# -----------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", "-m", default="CNN", help="Model type: CNN or LSTM")
    parser.add_argument("--person", "-p", default="false", help="Use person split (true) or general split (false)")
    args = parser.parse_args()

    seq_length = 128

    print("Loading data...")
    if args.person.lower() == "true":
        train_len, train_data, valid_len, valid_data, test_len, test_data = \
            load_data("./person_split/train", "./person_split/valid", "./person_split/test", seq_length)
    else:
        train_len, train_data, valid_len, valid_data, test_len, test_data = \
            load_data("./../data/train", "./../data/valid", "./../data/test", seq_length)

    print("Building model...")
    if args.model.upper() == "CNN":
        model = build_cnn(seq_length)
    elif args.model.upper() == "LSTM":
        model = build_lstm(seq_length)
    else:
        raise ValueError("Unknown model type: choose CNN or LSTM")

    print("Start training...")
    train_net(model, train_len, train_data, valid_len, valid_data, test_len, test_data, args.model)
    print("Training finished.")
