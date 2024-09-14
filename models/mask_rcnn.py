from pathlib import Path

from roboflow.core.dataset import Dataset
from tensorflow.keras.layers import Conv2D, Input, MaxPooling2D, UpSampling2D
from tensorflow.keras.models import Model

from models.utils.tensorflow_utils import run_model


def mask_rcnn(input_size=(256, 256, 3)):
    inputs = Input(input_size)

    # Encoder: Downsampling the input
    conv1 = Conv2D(64, (3, 3), activation="relu", padding="same")(inputs)
    pool1 = MaxPooling2D((2, 2))(conv1)

    conv2 = Conv2D(128, (3, 3), activation="relu", padding="same")(pool1)
    pool2 = MaxPooling2D((2, 2))(conv2)

    conv3 = Conv2D(256, (3, 3), activation="relu", padding="same")(pool2)
    pool3 = MaxPooling2D((2, 2))(conv3)

    # Decoder: Upsampling to match the original input size
    up4 = UpSampling2D((2, 2))(pool3)
    conv4 = Conv2D(128, (3, 3), activation="relu", padding="same")(up4)

    up5 = UpSampling2D((2, 2))(conv4)
    conv5 = Conv2D(64, (3, 3), activation="relu", padding="same")(up5)

    up6 = UpSampling2D((2, 2))(conv5)
    # Output layer: Sigmoid for binary segmentation
    outputs = Conv2D(1, (1, 1), activation="sigmoid", padding="same")(up6)

    model = Model(inputs=[inputs], outputs=[outputs])
    return model


def run_mask_rcnn(results_dir: Path, dataset: Dataset, epochs: int, learning_rate: float, train_imgsize: str):
    run_model(results_dir, dataset, epochs, learning_rate, train_imgsize, mask_rcnn, "mask_rcnn_model.h5")
