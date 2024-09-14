from pathlib import Path

from roboflow.core.dataset import Dataset
from tensorflow.keras.layers import Input, Conv2D, UpSampling2D, AveragePooling2D, Concatenate, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.applications import ResNet50

from models.utils.tensorflow_utils import run_model


def pyramid_pooling_module(x, pool_sizes):
    """Pyramid Pooling Module"""
    concat_list = [x]

    for pool_size in pool_sizes:
        # Apply average pooling
        pooled = AveragePooling2D(pool_size=(pool_size, pool_size))(x)
        pooled = Conv2D(256, (1, 1), padding="same", activation="relu")(pooled)  # 256 filters
        pooled = UpSampling2D(size=(pool_size, pool_size), interpolation="bilinear")(pooled)
        concat_list.append(pooled)

    # Concatenate the results
    return Concatenate()(concat_list)


def pspnet(input_size=(256, 256, 3), target_size=(256, 256)):
    inputs = Input(shape=input_size)

    # Use a backbone, e.g., ResNet50
    base_model = ResNet50(weights="imagenet", include_top=False, input_tensor=inputs)

    # Extract features from different layers
    x = base_model.get_layer("conv4_block6_out").output  # Feature from ResNet block 4

    # First convolution with 64 filters
    x = Conv2D(64, (3, 3), padding="same", activation="relu")(x)
    x = BatchNormalization()(x)  # Adding Batch Normalization

    # Pyramid pooling module with 256 filters for pooled layers
    x = pyramid_pooling_module(x, pool_sizes=[1, 2, 3, 6])

    # Second convolution with 128 filters
    x = Conv2D(128, (3, 3), padding="same", activation="relu")(x)
    x = BatchNormalization()(x)  # Adding Batch Normalization

    # Final Convolution Layer with 256 filters
    x = Conv2D(256, (3, 3), padding="same", activation="relu")(x)
    x = BatchNormalization()(x)  # Adding Batch Normalization

    # Calculate the total downsampling factor of the backbone up to "conv4_block6_out"
    # For ResNet50, it's typically 16 for input_size=(256,256)
    # total_downsampling = 16
    # Calculate the upsampling factor to reach the target size
    # Ensure target_size is a multiple of (input_size / total_downsampling)
    # upsampling_factor = target_size[0] // (input_size[0] // total_downsampling)

    # Upsample to target size
    # x = UpSampling2D(size=(upsampling_factor, upsampling_factor), interpolation='bilinear')(x)
    x = UpSampling2D(size=(input_size[0] // x.shape[1], input_size[1] // x.shape[2]), interpolation="bilinear")(
        x
    )  # Upsample to input size

    # Output Layer
    outputs = Conv2D(1, (1, 1), activation="sigmoid")(x)

    model = Model(inputs=[inputs], outputs=[outputs])
    return model


def run_pspnet(results_dir: Path, dataset: Dataset, epochs: int, learning_rate: float, train_imgsize: str):
    run_model(results_dir, dataset, epochs, learning_rate, train_imgsize, pspnet, "pspnet_model.h5")
