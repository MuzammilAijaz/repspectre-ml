# RepSpectre ML

RepSpectre ML is the data analysis pipeline and model-training side of the barbell-mounted workout tracker, [RepSpectre](https://github.com/MuzammilAijaz/repspectre).

It exists because a barbell can be carried, unracked, set up, or moved around before a meaningful repetition happens. The aim is to turn those labelled recordings into models that can separate a repetition from the movement around it, then eventually help the firmware reason about a lifter's current phase of a workout.

This repository prepares IMU sessions as datasets that are then used to train TensorFlow models. The resulting models can be quantized and run on an MCU through TensorFlow Lite and TensorFlow Lite for Microcontrollers. It also includes an in-progress data visualizer for inspecting sessions and repetition boundaries. Stacked filtering is available, while interactive correction and live inference remain planned.

## Project context

The pipeline sits between collection and embedded experiments:

```text
   repspectre: Firmware captures barbell motion
        |
        v
   repspectre-mobile: Stores labelled workout sessions
        |
        v
-> repspectre-ml: This repository prepares, inspects, and trains on those sessions
```

The pipeline works with IMU sessions collected through [RepSpectre Mobile](https://github.com/MuzammilAijaz/repspectre-mobile), originally alongside [repspectre-lite](https://github.com/MuzammilAijaz/repspectre-lite) prototype. Its exported artifacts are intended for [RepSpectre](https://github.com/MuzammilAijaz/repspectre), the portable event-driven firmware project.

> **Project status:** Database export, dataset splitting, TensorFlow training, TensorFlow Lite conversion, and C-array export are implemented. The visualizer supports session inspection, repetition-boundary estimation, and stacked filtering. Interactive correction, augmented-data views, in-UI training, live inference, model quality, and firmware deployment are still being iterated on.

## Model goals, design and purpose

Goal: Determine when a repetition has started and ended over the course of a whole workout session by combining ML predictions with the broader movement context maintained by the firmware.

The intended architecture combines heuristics and a state machine to keep track of the broader context, allowing ML models to solve smaller and more specific classification problems instead of asking one model to make every decision.

Problem: A barbell can be carried, rolled, unracked, set up, nudged, dropped, or exposed to vibration before a meaningful repetition happens. A typical sequence may include starting the device, placing it on the bar, setting up, taking slack from the bar, performing one or more repetitions, and setting the bar down again.

Each of these phases can produce motion that looks like part of a lift on its own. The system therefore needs to keep track of what happened before and after the current movement, rather than judging each motion in isolation.

### Maintaining History

The firmware will maintain a broader "Motion State" describing where the device is in the current sequence. A future state flow may include states such as:

```text
  IDLE → SETUP → UNRACK → LIFTING → REP → LIFTING → RACKED
```

The state machine provides context for interpreting the current sensor data. A movement that looks similar to a lift may have a different meaning depending on the state the device is currently in.

This history is separate from the temporal information maintained internally by a model such as an LSTM. The state machine keeps track of the broader movement context, while the model learns patterns within the sensor data itself.

This is why the dataset allows context-dependent "noise" sessions, including cases where the start of a lift should not be predicted. Examples include setup noise, walking around the gym, a barbell being hit, and plates being placed on a barbell.

These movements are useful training examples because they can look similar to lifting movements while having a different meaning depending on the current Motion State. Keeping the more specific motion-state labels allows the system to further fine-tune these distinctions.

A simple binary classifier that only answers "lift" or "not lift" loses this information. Instead, the ML models can provide evidence for particular states or transitions, while the state machine uses that evidence together with the history of the session to make the broader decision.

This allows the ML problem to be broken into smaller classifications rather than requiring one model to understand the entire workout.

### Use of smaller ML models for "simpler" classifications

> **Note**: Still in works. Requires experimentation and actual numbers before concluding.

Question: Can simpler ML models reliably distinguish certain motions when the state machine provides additional context, making a more complex model unnecessary for those classifications?

Assumption: Smaller/traditional ML models, such as nearest centroid and HMM, can be sufficiently effective for inferring the motion state when the difference between two motions is more apparent, such as walking vs. idle.

When the state machine already knows that the device is in a particular part of the workout, the model may only need to distinguish between a smaller number of possible movements. This could make simpler models sufficiently effective for some classifications.

However, distinguishing portions of a similar dynamic motion is much harder and may lead to more misfires. For these cases, a more complex model such as a CNN, LSTM, or newer deep-learning approach may provide a practical benefit.

The reason for using simpler models where possible is that they can require less processing power than a neural network. If a simpler model is sufficient for a particular classification, using it could reduce the computational and battery cost of the system.

The goal is therefore to combine state-machine logic, heuristics, and ML models, rather than relying on a single model to make every decision. The state machine provides history and context, while ML models provide evidence for the individual movement classifications where they are useful.

## Data model

### Data format

The canonical recording format is a full IMU stream:

```text
ax, ay, az, gx, gy, gz, qx, qy, qz, qw, timestampUs
```

The SQLite schema associates those samples with a session. A session carries a motion-state label and data format, and may also have lift context such as lift category, tempo, and RPE. The same collection database supports inspection, export, and later changes to the labeling policy.

The initial mobile collection workflow uses a 10-second timer mode. A bounded recording gives room for a single repetition and its surrounding setup while keeping the label and rep-boundary problem manageable during early data collection.

Continuous collection is a later option, but it would introduce longer stretches of ambiguous movement and make segmentation more demanding.

## Database

The database files are development data. The collection strategy and labels will continue to change as more lift types and difficult non-lift cases are captured.

The schema diagram can be regenerated with:

```sh
make db-to-diagram
```

It is written to `docs/schema.png`.

## Pipeline

```text
Android collection database
        |
        v
SQLite sessions and metadata
        |
        v
CSV export by lift category or motion state
        |
        v
Rep segmentation and dataset splitting
        |
        v
TensorFlow training
        |
        v
TFLite and C-array export for embedded experiments
```

1. `prepare_dataset.py` exports complete lift sessions, detected rep segments, and noise sessions into separate class-oriented directories. 
2. `split.py` then produces train, validation, and test directories from selected classes.
3. The `trainer.py` supports CNN and LSTM model configurations, exports Keras and TensorFlow Lite artifacts, and can generate C source/header files for an embedded build.

The exported artifacts are the handoff point for firmware integration.

## Getting started

This project uses Python 3.13 or newer and `uv` for environment management.

```sh
uv sync
make test
make lint
```

Set `DATABASE_FILE` when working with a database other than the current default:

```sh
make db-to-csv DATABASE_FILE=./data/your-recordings.db
make db-to-diagram DATABASE_FILE=./data/your-recordings.db
```

Training is currently driven directly through the trainer configuration:

```sh
uv run python src/lift_ml/training/trainer.py \
     --config src/lift_ml/sample_config.yaml
```

The trainer’s executable entry point currently selects its database and class folders in code. Review those values before running it: the flow regenerates prepared data directories as part of a training run.

## Visualizer

The PySide6/PyQtGraph visualizer is the start of a desktop environment for working through the data-preparation choices that affect a model. It loads recorded sessions, plots multi-axis IMU data, estimates a repetition boundary, and helps assess collection and segmentation before training.

```sh
make run-visualizer
```

The visualizer will let data transformations, filtering choices, and later lightweight inference be combined in one place and inspected before another training run. It avoids repeating the same experiments across disconnected scripts and notebooks. Manual rep-boundary correction, augmented-data views, in-UI training, and live inference remain incomplete.

### Purpose

When working with sensor data, there are many different transformations, cleaning mechanisms, and filtering techniques that can be applied before the data is given to a model.

For example:

- Data modifications: using different sensor axes, removing gravity, changing the data representation, etc.
- Filtering modifications: applying different noise-removal techniques, smoothing, mean filters, and other filtering methods.
- Algorithmic modifications: applying FFTs, spectrograms, and other transformations that may provide different representations of the sensor data.

The purpose of the visualizer is to allow these different modifications to be applied and combined quickly, while being able to immediately see how they affect the recorded sensor data.

This allows different combinations of data preparation to be experimented with before training a model. The resulting data can then be inspected to determine whether a particular transformation or filtering approach makes the repetition or other motion states easier to distinguish.

The visualizer is also intended to allow ML/NN inference to be run on the currently selected data preparation, so that the effect of these modifications can be observed directly. For algorithms that are fast enough to run interactively, such as basic ML models, training can also be performed using the current combination of data preparation.

The overall goal is to make experimenting with data preparation, filtering, and models a faster and more visual process, rather than requiring each combination to be tested through separate scripts or notebooks.

### Features

#### Data Modifications

- [x] Allow viewing of raw "sessions".
- [x] Allow application of different stacked filtering algorithms and visualize the results.
- [x] Allow detection of the start of the repetition and display it per session. Since only a single rep is performed per 10-second period, a simple algorithm can be used to detect the start.
- [ ] Allow users to manually override the detected repetition boundary, since these simple algorithms can get things wrong.

#### Filtering & Preparation

- [ ] Allow viewing of augmented data per session.
- [ ] Allow combining different data modifications and filtering approaches.
- [ ] Allow inspection of the resulting data before it is used for training or inference.

#### Machine Learning

- [ ] Allow training algorithms using the current data modifications and preparation.
- [ ] Allow users to run inference on data using the current data modifications and preparation.
- [ ] Allow live inspection of inference results when the selected algorithm is fast enough to run interactively.

## Repository layout

```text
data/                   Local SQLite recordings and generated datasets

src/lift_ml/
  data/                 Database access, export, augmentation, loading, and splitting
  models/               Model definitions and TFLite/C-array export helpers
  training/             Training orchestration
  utils/                Sampling and rep-detection utilities

tools/data-visualizer/  Desktop inspection tool

tests/                  Data, split, training, and utility tests
```
