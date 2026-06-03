import tensorflow as tf

def build_lstm(seq_length: int, dimension: int, num_classes: int, batch_size: int = 64):
    model = tf.keras.Sequential([
        tf.keras.layers.Bidirectional(
            tf.keras.layers.LSTM(22, return_sequences=False, batch_input_shape=(batch_size, seq_length, dimension))
        ),
        tf.keras.layers.Dense(num_classes, activation="softmax") # Fixed to use num_classes and softmax
    ])

    model.build(input_shape=(batch_size, seq_length, dimension))
    return model
