import tensorflow as tf

def build_cnn(seq_length: int, dimension: int, num_classes: int):
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(8, (5, 3), padding="same", activation="relu",
                               input_shape=(seq_length, dimension, 1)),
        tf.keras.layers.MaxPool2D((4, 1), padding="same"),
        tf.keras.layers.Dropout(0.15),
        tf.keras.layers.Conv2D(16, (4, 1), padding="same", activation="relu"),
        tf.keras.layers.MaxPool2D((4, 1), padding="same"),
        tf.keras.layers.Dropout(0.15),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(32, activation="relu"),
        tf.keras.layers.Dropout(0.15),
        tf.keras.layers.Dense(num_classes, activation="softmax")
    ])
    return model
