import gc
import logging
import os

import tensorflow as tf
import torch

logger = logging.getLogger(__name__)
logging.basicConfig()
logger.setLevel(logging.INFO)


def _is_tensorflow_supported(model: str):
    return model != "YOLOv8n"


def free_gpu(model_name: str):
    if _is_tensorflow_supported(model_name):
        # Clear the session and reset the GPU state
        tf.keras.backend.clear_session()
        # Clean up memory
        gc.collect()
    else:
        # Clear any cached memory
        torch.cuda.empty_cache()
        # Clean up CPU memory
        gc.collect()


def configure_gpu_memory(model_name: str, buffer_percent=1):
    if _is_tensorflow_supported(model_name):
        tensorflow_configure_gpu_memory(model_name, buffer_percent)
    else:
        pytorch_configure_gpu_memory(model_name, buffer_percent)


# Configure TensorFlow to handle GPU memory efficiently by enabling memory growth
def tensorflow_configure_gpu_memory(model_name: str, buffer_percent=1):
    free_gpu(model_name)

    # Ignore non-critical warnings
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
    os.environ["TF_GPU_ALLOCATOR"] = "cuda_malloc_async"

    # List physical devices
    gpus = tf.config.list_physical_devices("GPU")
    if not gpus:
        logger.warning("No GPUs found")
        return

    for gpu in gpus:
        logger.info("Configuring GPU: %s", gpu)
        # Correctly format the device name
        device_name = gpu.name.split(":")[-1]  # Extract '0' from '/physical_device:GPU:0'
        device_name = f"GPU:{device_name}"  # Format to 'GPU:0'

        try:
            # Retrieve memory info using correct device name
            memory_info = tf.config.experimental.get_memory_info(device_name)
            total_memory = memory_info["peak"]
            free_memory = total_memory * (1 - buffer_percent / 100)

            # Set virtual device configuration
            tf.config.experimental.set_virtual_device_configuration(
                gpu, [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=free_memory)]
            )
            logger.info("Configured GPU: %s with %s bytes", device_name, free_memory)
        except RuntimeError as e:
            logger.exception("Error configuring GPU memory: %s", e)


# Configure PyTorch to handle GPU memory efficiently
def pytorch_configure_gpu_memory(model_name: str, buffer_percent=1):
    free_gpu(model_name)

    # Optionally set environment variables to suppress warnings
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # No longer relevant for PyTorch
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "garbage_collection_threshold:0.9,max_split_size_mb:256"

    if not torch.cuda.is_available():
        logger.info("No GPUs found")
        return

    try:
        # Retrieve the total GPU memory and calculate the memory limit
        total_memory = torch.cuda.get_device_properties(0).total_memory
        free_memory = total_memory * (1 - buffer_percent / 100)

        # Set per-process memory fraction (PyTorch)
        torch.cuda.set_per_process_memory_fraction(free_memory / total_memory)

        logger.info("Configured GPU with %.2f MB free memory", free_memory / (1024**2))
    except RuntimeError as e:
        logger.exception("Error configuring GPU memory: %s", e)
