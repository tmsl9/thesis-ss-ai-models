import tensorflow as tf
from tensorflow.keras.layers import (
    Input,
    Conv2D,
    BatchNormalization,
    ReLU,
    MaxPooling2D,
    Add,
    UpSampling2D,
    GlobalAveragePooling2D,
    Multiply,
)
from tensorflow.keras.models import Model
from pathlib import Path
from roboflow.core.dataset import Dataset
from models.utils.tensorflow_utils import run_model


def conv_block(inputs, filters, kernel_size=3, strides=1, padding="same"):
    x = Conv2D(filters, kernel_size=kernel_size, strides=strides, padding=padding)(inputs)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    return x


def spatial_path(inputs):
    # Spatial Path adjusted to be comparable to FCN
    x = conv_block(inputs, 64, kernel_size=3, strides=1, padding="same")  # 1st conv layer
    x = conv_block(x, 64, kernel_size=3, strides=1, padding="same")  # 2nd conv layer
    x = MaxPooling2D((2, 2))(x)  # Downsample

    x = conv_block(x, 128, kernel_size=3, strides=1, padding="same")  # 3rd conv layer
    x = conv_block(x, 128, kernel_size=3, strides=1, padding="same")  # 4th conv layer
    x = MaxPooling2D((2, 2))(x)  # Downsample

    x = conv_block(x, 256, kernel_size=3, strides=1, padding="same")  # 5th conv layer
    return x


def feature_fusion_module(spatial_output, context_output1, context_output2):
    # Resize context_output2 to match the spatial dimensions of context_output1
    context_output2_resized = UpSampling2D(
        size=(
            context_output1.shape[1] // context_output2.shape[1],
            context_output1.shape[2] // context_output2.shape[2],
        )
    )(context_output2)

    # Use Conv2D to match the number of channels
    context_output1_adjusted = Conv2D(256, (1, 1), padding="same")(
        context_output1
    )  # Adjust context_output1 channels to 256
    context_combined = Add()([context_output1_adjusted, context_output2_resized])

    # Combine with spatial features after ensuring channel compatibility
    spatial_resized = Conv2D(256, (1, 1), padding="same")(spatial_output)  # Adjust spatial_output channels to 256
    spatial_resized = UpSampling2D(
        size=(
            context_combined.shape[1] // spatial_resized.shape[1],
            context_combined.shape[2] // spatial_resized.shape[2],
        )
    )(spatial_resized)

    combined = Add()([spatial_resized, context_combined])

    return combined


def context_path(inputs):
    # Context Path adjusted to match FCN's complexity
    x = inputs

    # Block 1
    x1 = conv_block(x, 64, kernel_size=3, strides=2, padding="same")  # 1st block
    x1 = conv_block(x1, 128, kernel_size=3, strides=2, padding="same")  # 2nd block

    # Block 2
    x2 = conv_block(x1, 256, kernel_size=3, strides=2, padding="same")  # 3rd block
    x2 = conv_block(x2, 256, kernel_size=3, strides=2, padding="same")  # 4th block

    # Global Average Pooling Feature for Feature Fusion
    global_context = GlobalAveragePooling2D()(x2)
    global_context = tf.keras.layers.Reshape((1, 1, 256))(global_context)
    global_context = Conv2D(256, (1, 1), padding="same", activation="sigmoid")(global_context)
    x2 = Multiply()([x2, global_context])

    # Adjust the upsampling sizes
    x2 = UpSampling2D(size=(4, 4))(x2)  # Ensure x2 has spatial dimensions matching x1
    x1 = Conv2D(256, (1, 1), padding="same")(x1)  # Adjust channels to 256 for compatibility
    x1 = UpSampling2D(size=(2, 2))(x1)

    return x1, x2


def bisenet(input_size=(256, 256, 3)):
    inputs = Input(shape=input_size)

    # Spatial and Context Paths
    spatial_output = spatial_path(inputs)
    context_output1, context_output2 = context_path(inputs)

    # Feature Fusion Module
    combined_features = feature_fusion_module(spatial_output, context_output1, context_output2)

    # Final output layers
    x = conv_block(combined_features, 256, kernel_size=3, strides=1, padding="same")
    x = UpSampling2D(size=(input_size[0] // x.shape[1], input_size[1] // x.shape[2]))(x)  # Upsample to input size

    # Output layer
    outputs = Conv2D(1, (1, 1), activation="sigmoid")(x)

    model = Model(inputs, outputs)
    return model


def run_bisenet(results_dir: Path, dataset: Dataset, epochs: int, learning_rate: float, train_imgsize: str):
    run_model(results_dir, dataset, epochs, learning_rate, train_imgsize, bisenet, "bisenet_model.h5")
