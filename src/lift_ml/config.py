import os
from typing import Any, Self, cast

import yaml
from pydantic import BaseModel


class DataConfig(BaseModel):
    train_path: str
    valid_path: str
    test_path: str
    seq_length: int = 512
    data_dimension: int = 6
    axes: list[str] | None = None  # If None, slice first data_dimension columns
    labels: list[str] = ["noise", "rep_start"]
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
    def from_yaml(cls, path: str) -> Self:
        with open(path) as f:
            data = cast("dict[str, Any]", yaml.safe_load(f))
        return cls(**data)

    def save_yaml(self, path: str) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            yaml.dump(self.model_dump(), f)
