from pathlib import Path

from roboflow.core.dataset import Dataset
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Conv2DTranspose, Concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.applications import ResNet50

from models.utils.tensorflow_utils import run_model


def segnet(input_size=(256, 256, 3)):
    inputs = Input(input_size)

    # Encoder
    # Block 1
    x = Conv2D(64, (3, 3), activation="relu", padding="same")(inputs)
    x = Conv2D(64, (3, 3), activation="relu", padding="same")(x)
    pool1 = MaxPooling2D((2, 2), padding="same")(x)

    # Block 2
    x = Conv2D(128, (3, 3), activation="relu", padding="same")(pool1)
    x = Conv2D(128, (3, 3), activation="relu", padding="same")(x)
    pool2 = MaxPooling2D((2, 2), padding="same")(x)

    # Block 3
    x = Conv2D(256, (3, 3), activation="relu", padding="same")(pool2)
    x = Conv2D(256, (3, 3), activation="relu", padding="same")(x)
    pool3 = MaxPooling2D((2, 2), padding="same")(x)

    # Decoder
    # Block 3
    x = Conv2DTranspose(256, (3, 3), strides=(2, 2), padding="same")(pool3)
    x = Conv2D(256, (3, 3), activation="relu", padding="same")(x)

    # Block 2
    x = Conv2DTranspose(128, (3, 3), strides=(2, 2), padding="same")(x)
    x = Conv2D(128, (3, 3), activation="relu", padding="same")(x)

    # Block 1
    x = Conv2DTranspose(64, (3, 3), strides=(2, 2), padding="same")(x)
    x = Conv2D(64, (3, 3), activation="relu", padding="same")(x)

    # Output layer
    outputs = Conv2D(1, (1, 1), activation="sigmoid")(x)

    model = Model(inputs, outputs)
    return model


def run_segnet(results_dir: Path, dataset: Dataset, epochs: int, learning_rate: float, train_imgsize: str):
    run_model(results_dir, dataset, epochs, learning_rate, train_imgsize, segnet, "segnet_model.h5")
