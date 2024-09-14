import tensorflow as tf
from tensorflow.keras.layers import add, Layer, Input, Conv2D, BatchNormalization, ReLU, MaxPooling2D, UpSampling2D
from tensorflow.keras.models import Model
from roboflow.core.dataset import Dataset
from models.utils.tensorflow_utils import run_model


class ResizeLayer(Layer):
    def __init__(self, target_size):
        super(ResizeLayer, self).__init__()
        self.target_size = target_size

    def call(self, inputs):
        return tf.image.resize(inputs, self.target_size)

    def get_config(self):
        config = super().get_config()
        config.update({"target_size": self.target_size})
        return config


def conv_block(x, filters, kernel_size=3, stride=1, padding="same"):
    x = Conv2D(filters, kernel_size, strides=stride, padding=padding)(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    return x


def high_resolution_module(x, num_filters):
    x1 = conv_block(x, num_filters)  # First block
    x2 = conv_block(x1, num_filters)  # Second block

    # Upsample x2 to match the spatial dimensions of x1
    x3 = UpSampling2D(size=(2, 2))(x2)  # First upsampling
    x3 = conv_block(x3, num_filters)

    # Resize x3 down to match x1 if needed
    if x1.shape[1] < x3.shape[1] or x1.shape[2] < x3.shape[2]:
        resize_layer = ResizeLayer((x1.shape[1], x1.shape[2]))
        x3 = resize_layer(x3)

    # Merge x1 and x3
    x4 = add([x1, x3])
    return x4


def hrnet(input_size=(256, 256, 3)):
    inputs = Input(shape=input_size)

    # Initial Convolution Block
    x = conv_block(inputs, 64, kernel_size=7, stride=2)

    # Downsampling
    x = MaxPooling2D(pool_size=(3, 3), strides=2, padding="same")(x)

    # High-Resolution Blocks
    x = high_resolution_module(x, 64)  # First block with 64 filters
    x = high_resolution_module(x, 128)  # Second block with 128 filters
    x = high_resolution_module(x, 256)  # Third block with 256 filters

    # Final Layers
    x = UpSampling2D(size=(input_size[0] // x.shape[1], input_size[1] // x.shape[2]))(
        x
    )  # UpSample to desired output size
    x = conv_block(x, 256)
    x = conv_block(x, 128)
    x = conv_block(x, 64)

    # Output Layer
    outputs = Conv2D(1, (1, 1), activation="sigmoid")(x)

    model = Model(inputs, outputs)
    return model


def run_hrnet(results_dir: str, dataset: Dataset, epochs: int, learning_rate: float, train_imgsize: str):
    run_model(results_dir, dataset, epochs, learning_rate, train_imgsize, hrnet, "hrnet_model.h5")
