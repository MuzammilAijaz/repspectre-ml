from pydantic import BaseModel, Field
from typing import List, Optional
import yaml
import os

class DataConfig(BaseModel):
    train_path: str
    valid_path: str
    test_path: str
    seq_length: int = 512
    data_dimension: int = 6
    labels: List[str] = ["notOnBarbell", "onBarbell"]
    label_name: str = "gesture"
    data_name: str = "accel_ms2_xyz"

class ModelConfig(BaseModel):
    type: str = "cnn" # "cnn" or "lstm"
    epochs: int = 250
    batch_size: int = 64
    learning_rate: float = 0.001

class Config(BaseModel):
    data: DataConfig
    model: ModelConfig
    output_dir: str = "models/export"

    @classmethod
    def from_yaml(cls, path: str):
        with open(path, "r") as f:
            data = yaml.safe_load(f)
        return cls(**data)

    def save_yaml(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            yaml.dump(self.model_dump(), f)
