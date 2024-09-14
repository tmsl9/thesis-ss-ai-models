from pathlib import Path

import tensorflow as tf
from tensorflow.keras.layers import (
    Input,
    Conv2D,
    BatchNormalization,
    Activation,
    AveragePooling2D,
    UpSampling2D,
    Concatenate,
    Dropout,
)
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Model
from tensorflow.keras.losses import BinaryCrossentropy
from roboflow.core.dataset import Dataset

from models.utils.tensorflow_utils import run_model


def ASPP(x, filters):
    """Atrous Spatial Pyramid Pooling (ASPP)"""
    shape = x.shape

    # 1x1 convolution
    y1 = Conv2D(filters, 1, padding="same", use_bias=False)(x)
    y1 = BatchNormalization()(y1)
    y1 = Activation("relu")(y1)

    # Atrous convolutions with different dilation rates
    y2 = Conv2D(filters, 3, padding="same", dilation_rate=6, use_bias=False)(x)
    y2 = BatchNormalization()(y2)
    y2 = Activation("relu")(y2)

    y3 = Conv2D(filters, 3, padding="same", dilation_rate=12, use_bias=False)(x)
    y3 = BatchNormalization()(y3)
    y3 = Activation("relu")(y3)

    y4 = Conv2D(filters, 3, padding="same", dilation_rate=18, use_bias=False)(x)
    y4 = BatchNormalization()(y4)
    y4 = Activation("relu")(y4)

    # Image-level features
    y5 = AveragePooling2D(pool_size=(shape[1], shape[2]))(x)
    y5 = Conv2D(filters, 1, padding="same", use_bias=False)(y5)
    y5 = BatchNormalization()(y5)
    y5 = Activation("relu")(y5)
    y5 = UpSampling2D(size=(shape[1], shape[2]), interpolation="bilinear")(y5)

    # Concatenate all the paths
    y = Concatenate()([y1, y2, y3, y4, y5])

    y = Conv2D(filters, 1, padding="same", use_bias=False)(y)
    y = BatchNormalization()(y)
    y = Activation("relu")(y)

    return y


def deeplabv3plus(input_size=(256, 256, 3)):
    inputs = Input(shape=input_size)

    # Use a ResNet50 backbone
    base_model = ResNet50(weights="imagenet", include_top=False, input_tensor=inputs)

    # Extract features
    image_features = base_model.get_layer("conv4_block6_out").output

    # Apply ASPP with 256 filters
    x_a = ASPP(image_features, filters=256)

    # Low-level feature map from earlier in the network
    low_level_features = base_model.get_layer("conv2_block3_out").output
    low_level_features = Conv2D(64, (1, 1), padding="same", use_bias=False)(
        low_level_features
    )  # Start with 64 filters
    low_level_features = BatchNormalization()(low_level_features)
    low_level_features = Activation("relu")(low_level_features)

    # Upsample x_a to match the low-level feature map dimensions
    x_a = UpSampling2D(size=(4, 4), interpolation="bilinear")(x_a)

    # Concatenate low-level features with upsampled ASPP output
    x = Concatenate()([x_a, low_level_features])

    # Apply convolution layers with progressively increasing filters: 64, then 128, and finally 256
    x = Conv2D(64, (3, 3), padding="same", use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    x = Dropout(0.3)(x)  # Adding Dropout to prevent overfitting

    x = Conv2D(128, (3, 3), padding="same", use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    x = Dropout(0.3)(x)  # Adding Dropout to prevent overfitting

    x = Conv2D(256, (3, 3), padding="same", use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    # Upsample to original input size
    x = UpSampling2D(size=(4, 4), interpolation="bilinear")(x)

    # Output layer for binary segmentation
    outputs = Conv2D(1, (1, 1), activation="sigmoid")(x)

    model = Model(inputs, outputs)

    return model


def run_deeplabv3plus(results_dir: Path, dataset: Dataset, epochs: int, learning_rate: float, train_imgsize: str):
    run_model(results_dir, dataset, epochs, learning_rate, train_imgsize, deeplabv3plus, "deeplabv3plus_model.h5")
