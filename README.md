> Note: current developmental branch is [here](https://github.com/MuzammilAijaz/repspectre-ml/tree/data-analysis-pipeline)

# RepSpectre ML

RepSpectre ML is the data analysis pipeline and model-training side of the barbell-mounted workout tracker, [RepSpectre](https://github.com/MuzammilAijaz/repspectre).

It exists because a barbell can be carried, unracked, set up, or moved around before a meaningful repetition happens. The aim is to turn those labelled recordings into models that can separate a repetition from the movement around it, then eventually help the firmware reason about a lifter's current phase of a workout.

This repository prepares IMU sessions as datasets that are then used to train TensorFlow models. The resulting models can be quantized and run on an MCU through TensorFlow Lite and TensorFlow Lite for Microcontrollers. It also includes an in-progress data visualizer for inspecting sessions and repetition boundaries. Stacked filtering is available, while interactive correction and live inference remain planned.
