# Refactoring Plan - Lift Detection ML Project

## 1. Objectives
*   **Modularization**: Transition from standalone scripts to a cohesive Python package.
*   **Ease of Use**: Enable training of multiple models (e.g., Device-on-Barbell, Floor-Pull) using a unified interface.
*   **Dependency Injection**: Use a centralized configuration system to inject hyperparameters and file paths into the pipeline.
*   **Fast Experimentation**: Support Jupyter Notebooks that import the core logic without code duplication.
*   **Robustness**: Implement `pytest` suites to ensure data integrity and model functional requirements.

## 2. Proposed Directory Structure

```text
.
├── configs/                # Hyperparameter & path configurations (YAML/JSON)
│   ├── model1_barbell.yaml
│   └── model2_floorpull.yaml
├── data/                   # Centralized data storage
│   ├── raw/                # Unprocessed CSVs and databases
│   └── processed/          # Prepared and split datasets
├── experiments/            # Jupyter notebooks for research & visualization
│   └── analysis.ipynb      # Imports from src/
├── src/                    # Core project package
│   └── lift_ml/
│       ├── __init__.py
│       ├── config.py       # Configuration parser & schema
│       ├── data/           # Data pipeline modules
│       │   ├── loader.py   # Unified DataLoader
│       │   ├── pipeline.py # Orchestrates Prepare -> Split -> Augment
│       │   ├── processor.py# Core processing logic
│       │   └── augment.py  # Data augmentation techniques
│       ├── models/         # Model architecture definitions
│       │   ├── factory.py  # Returns model instance based on config
│       │   ├── cnn.py
│       │   └── lstm.py
│       ├── training/       # Training loop and callbacks
│       │   └── trainer.py
│       └── utils/          # Export, metrics, and size calculations
├── tests/                  # Pytest suite
│   ├── conftest.py         # Shared fixtures
│   ├── test_data.py        # Validation for processing & augmentation
│   └── test_models.py      # Validation for architecture & output shapes
├── Makefile                # Shortcuts for common commands
├── pyproject.toml          # Modern Python packaging configuration (uv managed)
└── README.md
```

## 3. Tooling & Environment
*   **Package Manager**: `uv` (for fast, reproducible builds and dependency resolution).
*   **Python Version**: Managed via `uv python` (targeting 3.10+ for modern TF compatibility).
*   **Task Runner**: `Makefile` to wrap `uv run` commands.

## 4. Data Pipeline Redesign
### Current Sequence
`data_prepare.py` (JSON export) -> `data_split.py` (File split) -> `data_load.py` (Class loader) -> `train.py` (Training).

### Proposed Sequential Pipeline
1.  **Ingestion**: `DataIngestor` reads various formats (CSV, SQLite) and converts them to a standard internal representation (Numpy/Tensors).
2.  **Transformation**: `DataProcessor` applies normalization and cleaning.
3.  **Partitioning**: `DataSplitter` handles person-based or random splitting.
4.  **Loading**: `DataLoader` (extending `tf.data.Dataset`) provides on-the-fly augmentation and batching.

**Refinement**: Use a `Pipeline` class that can be configured to skip steps (e.g., if processed data already exists).

## 4. Implementation Strategy

### Phase 1: Stabilization (Before Refactoring)
*   **Action**: Create a `tests/` directory and implement `pytest` for the current `data_load.py`, `data_prepare.py`, and `data_augmentation.py`.
*   **Goal**: Ensure that the "Gold Standard" output of the current pipeline is preserved.

### Phase 2: Core Package Development
*   **Action**: Initialize `src/lift_ml/` and migrate logic from root scripts.
*   **Feature**: Implement `lift_ml.config.Config` using `Pydantic` or a simple dataclass to handle all parameters.
*   **Feature**: Rewrite `DataLoader` to be generic, accepting paths and augmentation parameters from the `Config`.

### Phase 3: Model Factory & Trainer
*   **Action**: Create a model factory that builds Keras models based on string identifiers in the config.
*   **Action**: Centralize the training loop, TensorBoard callbacks, and TFLite export logic.

### Phase 4: Migration & Cleanup
*   **Action**: Update `models/model1_.../scripts/train.py` to use the new `lift_ml` package.
*   **Action**: Delete redundant script copies in subdirectories.
*   **Action**: Create a `Makefile` to simplify commands like `make train CONFIG=configs/model1_barbell.yaml`.

## 5. Testing Framework
*   **Choice**: `pytest`
*   **Reasoning**: It is the industry standard for Python, supports powerful fixtures (useful for mocking data), and integrates well with CI/CD.
*   **Key Tests**:
    *   `test_augmentation_consistency`: Ensure same seed produces same augmented data.
    *   `test_data_loader_shapes`: Verify output tensor shapes match model input.
    *   `test_model_export`: Verify TFLite conversion doesn't fail.

## 6. Dependency Injection Example
```python
# In src/lift_ml/data/loader.py
class DataLoader:
    def __init__(self, config: Config):
        self.batch_size = config.batch_size
        self.seq_length = config.seq_length
        # ...
```

## 7. Fast Experimentation
Notebooks will start with:
```python
from lift_ml.config import Config
from lift_ml.data.pipeline import run_pipeline
from lift_ml.models.factory import get_model

config = Config.from_yaml("../configs/model1_barbell.yaml")
data = run_pipeline(config)
model = get_model(config)
```
