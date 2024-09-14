from datetime import datetime, timedelta
import logging
from pathlib import Path
import sys
import time
from typing import Any, Callable

import cv2
import matplotlib.pyplot as plt
import numpy as np
from roboflow.core.dataset import Dataset
from sklearn.metrics import precision_score, recall_score, f1_score
import tensorflow as tf
from tensorflow.keras.metrics import MeanIoU
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

from models.utils.configure_gpu import free_gpu

logger = logging.getLogger(__name__)
logging.basicConfig()
logger.setLevel(logging.INFO)


"""
Using data generators and pre-processing data on the GPU using tf.data.Dataset will help reduce the risk of GPU
out-of-memory (OOM) errors for the following reasons:

1. Avoid Loading All Data at Once
    - Problem: Loading the entire dataset into memory at once (especially large image datasets) consumes a
    significant amount of GPU memory. This can easily lead to an OOM error if the memory is insufficient.
    - Solution: Data generators load data in small batches during training. Instead of keeping the entire dataset
    in memory, only the data for the current batch is loaded and processed. This approach significantly reduces
    memory usage, leaving more memory available for the model and other tasks.
2. Efficient Data Pipeline with tf.data.Dataset
    - Problem: When performing data loading and pre-processing on the CPU and then transferring to the GPU, the
    memory overhead can accumulate, especially if you're not managing the data efficiently.
    - Solution: The tf.data.Dataset API enables efficient pre-processing directly on the GPU, reducing the memory
    transfer between CPU and GPU. Additionally, it pipelines data loading, augmentation, and pre-processing in an
    optimized, asynchronous manner. This avoids bottlenecks and ensures that only the data needed for the current
    step is pre-processed and loaded onto the GPU.
        - By prefetching batches asynchronously, TensorFlow can efficiently feed the GPU with data without
        blocking the GPU's operations.
3. Dynamic Memory Management
    - Problem: TensorFlow's default behavior allocates all available GPU memory at the start of training, which
    can lead to OOM errors if your data or model size exceeds the available memory.
    - Solution: The combination of memory growth settings (tf.config.experimental.set_memory_growth) and virtual
    device configuration (allocating a portion of GPU memory) with a data generator ensures that only the
    necessary amount of memory is used at any given time. The rest of the GPU memory remains available for model
    parameters and computations.
4. Improved GPU Utilization
    - Problem: When performing data pre-processing on the CPU, the GPU can remain idle while waiting for data to
    be transferred and processed, leading to inefficient usage.
    - Solution: By moving pre-processing to the GPU (via tf.data.Dataset), both pre-processing and model
    computations can happen in parallel on the GPU. This maximizes GPU utilization and reduces idle time, lowering
    the chance of memory-related delays.
5. Batch Size Control
    - Problem: Larger batch sizes require more memory, leading to higher risks of OOM errors.
    - Solution: Using a data generator allows you to dynamically adjust the batch size based on your GPU's
    available memory. With small batches loaded in each step, you have more flexibility to balance memory usage
    between the data and model, minimizing memory overload.
"""


class DataGenerator(tf.keras.utils.Sequence):
    def __init__(self, images, masks, batch_size, shuffle=True):
        self.images = images
        self.masks = masks
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.indexes = np.arange(len(self.images))
        self.on_epoch_end()

    def __len__(self):
        return int(np.floor(len(self.images) / self.batch_size))

    def __getitem__(self, index):
        batch_indexes = self.indexes[index * self.batch_size : (index + 1) * self.batch_size]
        batch_images = self.images[batch_indexes]
        batch_masks = self.masks[batch_indexes]
        return batch_images, batch_masks

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indexes)


# Load and preprocess dataset
def _load_images_and_masks(dataset_path: Path, img_size):
    images = []
    masks = []
    original_dimensions = []  # Store original dimensions
    images_names = []

    for image_path in dataset_path.rglob(str(Path("images", "*"))):
        images_names.append(image_path.name)

        img = cv2.imread(image_path)
        original_dimensions.append(img.shape[:2])  # Store the original dimensions (height, width)

        img = cv2.resize(img, img_size)
        images.append(img)

        mask = cv2.imread(image_path.parent.parent / "masks" / image_path.name, cv2.IMREAD_GRAYSCALE)
        mask = cv2.resize(mask, img_size)
        mask = np.expand_dims(mask, axis=-1)
        masks.append(mask)

    return np.array(images), np.array(masks), original_dimensions, images_names


def _load_dataset(dataset: Dataset, train_imgsize: tuple):
    # Load dataset
    dataset_path = Path(dataset.location)
    # Load train data
    train_images, train_masks, _, _ = _load_images_and_masks(dataset_path / "train", train_imgsize)
    # Load validation data
    val_images, val_masks, _, _ = _load_images_and_masks(dataset_path / "valid", train_imgsize)
    # Load test data
    test_images, test_masks, test_original_dims, test_images_names = _load_images_and_masks(
        dataset_path / "test", train_imgsize
    )

    # Normalize images and masks
    train_images = train_images / 255.0
    train_masks = train_masks / 255.0
    val_images = val_images / 255.0
    val_masks = val_masks / 255.0
    test_images = test_images / 255.0

    # Create data generators
    train_generator = DataGenerator(train_images, train_masks, batch_size=4)
    val_generator = DataGenerator(val_images, val_masks, batch_size=4)

    return train_generator, val_generator, test_images, test_masks, test_original_dims, test_images_names


def _exec_model_operation_w_retry(
    results_dir: Path, model_name: str, operation_name: str, model_operation: Callable, **kwargs
) -> Any:
    # GPU can be out-of-memory so we retry a few times to operate the model
    retry_count = 0
    max_retries = 10
    timer_start = datetime.now()
    while retry_count < max_retries:
        try:
            free_gpu(model_name)
            # Execute model operation
            logger.info("%sing the model.", operation_name)
            operation_result = model_operation(**kwargs)
            break
        except Exception as e:
            free_gpu(model_name)
            logger.exception("An unexpected error occurred: %s", e)
            retry_count += 1
            if retry_count == max_retries:
                # Stop execution because the process keeps raising exceptions
                logger.exception("Stop execution because the process keeps raising exceptions")
                sys.exit()
            logger.info("Retry model %s.", operation_name)
            time.sleep(2)

    # Store time spent in model operation
    with (results_dir / f"process_time_{operation_name}.txt").open(mode="w") as file:
        # Remove 2 seconds of each retry sleep
        timer_end = datetime.now() - timedelta(seconds=2 * retry_count)
        file.write(f"Start: {timer_start}\n")
        file.write(f"End: {timer_end}\n")
        file.write(f"Duration: {timer_end - timer_start}\n")

    return operation_result


def _train_model(
    results_dir: Path,
    model_name: str,
    model: Model,
    train_generator: DataGenerator,
    val_generator: DataGenerator,
    epochs: int,
) -> None:
    _exec_model_operation_w_retry(
        results_dir, model_name, "train", model.fit, x=train_generator, validation_data=val_generator, epochs=epochs
    )


def _predict_model(results_dir: Path, model_name: str, model: Model, test_images: np.ndarray) -> np.ndarray:
    return _exec_model_operation_w_retry(results_dir, model_name, "predict", model.predict, x=test_images)


def _save_result_images(
    predicted_masks, original_images, original_dimensions, original_images_names, results_dir: Path, threshold=0.5
):
    for i, _ in enumerate(predicted_masks):
        # Get predicted mask and original image
        pred_mask = (predicted_masks[i] > threshold).astype(np.uint8)  # Threshold the prediction
        orig_image = original_images[i]

        # Resize predicted mask back to the original size
        original_height, original_width = original_dimensions[i]
        pred_mask_resized = cv2.resize(pred_mask, (original_width, original_height), interpolation=cv2.INTER_NEAREST)

        # Create a plot with two subplots
        fig, ax = plt.subplots(1, 2, figsize=(15, 5))

        # Plot Original Image
        # Convert the original image to uint8 if it isn't already
        if orig_image.dtype != np.uint8:
            orig_image = (orig_image * 255).astype(np.uint8)

        # Now apply the color conversion
        ax[0].imshow(cv2.cvtColor(orig_image, cv2.COLOR_BGR2RGB))  # Convert BGR to RGB for correct color display
        ax[0].set_title("Original Image")
        ax[0].axis("off")

        # Plot Predicted Mask
        ax[1].imshow(pred_mask_resized.squeeze(), cmap="gray")  # Ensure grayscale mask display
        ax[1].set_title("Predicted Mask")
        ax[1].axis("off")

        # Save the combined plot to file
        (results_dir / "combined_predictions").mkdir(parents=True, exist_ok=True)
        plt.savefig(results_dir / "combined_predictions" / original_images_names[i])
        plt.close()

        # Save the resized predicted mask as a standalone image
        (results_dir / "predictions").mkdir(parents=True, exist_ok=True)
        pred_mask_path = results_dir / "predictions" / original_images_names[i]
        cv2.imwrite(
            pred_mask_path, (pred_mask_resized * 255).astype(np.uint8)
        )  # Convert mask to 0-255 scale before saving
        logger.info("Saved predicted mask to %s\n", pred_mask_path)


def _calculate_n_store_metrics(predictions, masks, results_dir: Path, threshold=0.5):
    """Calculate precision, recall, and F1 score for binary masks."""
    # Threshold the predicted probabilities to get binary predictions
    predictions_flat = (predictions.flatten() > threshold).astype(int)

    # Flatten the arrays for binary classification metrics
    masks_flat = masks.flatten()
    # Ensure masks is also binary (if it isn't already)
    masks_flat = (masks_flat > threshold).astype(int)

    # Precision, Recall, F1
    precision = round(precision_score(masks_flat, predictions_flat), 3)
    recall = round(recall_score(masks_flat, predictions_flat), 3)
    f1 = round(f1_score(masks_flat, predictions_flat), 3)
    logger.info("Precision: %.3f, Recall: %.3f, F1 Score: %.3f\n", precision, recall, f1)

    # Store results
    with (results_dir / "results.txt").open(mode="w") as file:
        file.write(f"all {len(predictions)} - - - - - {precision} {recall} - -\n")
        logger.info("Results stored in: %s\n", results_dir / "results.txt")


def run_model(
    results_dir: Path,
    dataset: Dataset,
    epochs: int,
    learning_rate: float,
    train_imgsize: tuple,
    init_model: Callable,
    model_name: str,
):
    train_generator, val_generator, test_images, test_masks, test_original_dims, test_images_names = _load_dataset(
        dataset, train_imgsize
    )

    # Initialize and compile model
    model = init_model(input_size=(train_imgsize[0], train_imgsize[1], 3))
    # 2 classes: background and lesion (binary segmentation)
    model.compile(
        optimizer=Adam(learning_rate=learning_rate), loss="binary_crossentropy", metrics=[MeanIoU(num_classes=2)]
    )

    _train_model(results_dir, model_name, model, train_generator, val_generator, epochs)

    # Save the model
    model.save(model_name)

    predicted_masks = _predict_model(results_dir, model_name, model, test_images)

    # Save validation results with correct dimensions
    _save_result_images(
        predicted_masks, test_images, test_original_dims, test_images_names, results_dir, threshold=0.5
    )

    # Evaluate model performance on test data
    _calculate_n_store_metrics(test_masks, predicted_masks, results_dir)

    free_gpu(model_name)
