from lift_ml.config import Config
from lift_ml.models.cnn import build_cnn
from lift_ml.models.lstm import build_lstm

def get_model(config: Config):
    model_type = config.model.type.upper()
    seq_length = config.data.seq_length
    dimension = config.data.data_dimension
    num_classes = len(config.data.labels)
    batch_size = config.model.batch_size

    if model_type == "CNN":
        return build_cnn(seq_length, dimension, num_classes)
    elif model_type == "LSTM":
        return build_lstm(seq_length, dimension, num_classes, batch_size)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
