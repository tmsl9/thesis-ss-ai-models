import logging
from pathlib import Path

import cv2
import numpy as np

from dataset_utils import copy_dataset_folder, get_algorithm_image_result_dir_path

logger = logging.getLogger(__name__)
logging.basicConfig()
logger.setLevel(logging.INFO)


def apply_gabor_filter(image_path: Path, ksize=15, sigma=2.0, theta=0, lambd=8.0, gamma=0.5, psi=0):
    logger.info("Applying gabor filter in %s", image_path)

    alg_name = "gabor_filter"
    res_dir_image_path = get_algorithm_image_result_dir_path(image_path, alg_name)
    res_image_path = res_dir_image_path / image_path.name
    if res_image_path.exists():
        logger.info("Gabor filter is already applied, ignoring...")
        return
    copy_dataset_folder(image_path.parent.parent.parent, alg_name=alg_name, ignore_images=True)

    image = cv2.imread(str(image_path))

    # Split the image into R, G, and B channels
    channels = cv2.split(image)

    # Create the Gabor kernel
    gabor_kernel = cv2.getGaborKernel((ksize, ksize), sigma, theta, lambd, gamma, psi, ktype=cv2.CV_32F)

    # Apply the Gabor filter to each channel
    filtered_channels = []
    for channel in channels:
        filtered = cv2.filter2D(channel, cv2.CV_32F, gabor_kernel)  # Apply filter
        filtered = cv2.normalize(filtered, None, 0, 255, cv2.NORM_MINMAX)  # Normalize to 0-255
        filtered = np.uint8(filtered)  # Convert to 8-bit image
        filtered_channels.append(filtered)

    # Merge the filtered channels back into a color image
    filtered_image = cv2.merge(filtered_channels)
    cv2.imwrite(str(res_image_path), filtered_image)
