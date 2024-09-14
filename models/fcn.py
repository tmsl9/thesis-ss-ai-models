from pathlib import Path

from roboflow.core.dataset import Dataset
from tensorflow.keras.layers import Input, Conv2D, Conv2DTranspose, MaxPooling2D
from tensorflow.keras.models import Model

from models.utils.tensorflow_utils import run_model


def fcn(input_size=(256, 256, 3)):
    inputs = Input(input_size)

    # Encoder
    x = Conv2D(64, (3, 3), activation="relu", padding="same")(inputs)
    x = Conv2D(64, (3, 3), activation="relu", padding="same")(x)
    x = MaxPooling2D((2, 2))(x)

    x = Conv2D(128, (3, 3), activation="relu", padding="same")(x)
    x = Conv2D(128, (3, 3), activation="relu", padding="same")(x)
    x = MaxPooling2D((2, 2))(x)

    x = Conv2D(256, (3, 3), activation="relu", padding="same")(x)
    x = Conv2D(256, (3, 3), activation="relu", padding="same")(x)
    x = MaxPooling2D((2, 2))(x)

    # Decoder
    x = Conv2DTranspose(256, (3, 3), strides=(2, 2), padding="same")(x)
    x = Conv2D(128, (3, 3), activation="relu", padding="same")(x)

    x = Conv2DTranspose(128, (3, 3), strides=(2, 2), padding="same")(x)
    x = Conv2D(64, (3, 3), activation="relu", padding="same")(x)

    x = Conv2DTranspose(64, (3, 3), strides=(2, 2), padding="same")(x)

    # Output layer
    outputs = Conv2D(1, (1, 1), activation="sigmoid")(x)

    model = Model(inputs, outputs)
    return model


def run_fcn(results_dir: Path, dataset: Dataset, epochs: int, learning_rate: float, train_imgsize: str):
    run_model(results_dir, dataset, epochs, learning_rate, train_imgsize, fcn, "fcn_model.h5")
