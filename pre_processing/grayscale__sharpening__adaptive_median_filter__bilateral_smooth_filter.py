import logging
from pathlib import Path

import cv2
import numpy as np

from dataset_utils import copy_dataset_folder, get_algorithm_image_result_dir_path

logger = logging.getLogger(__name__)
logging.basicConfig()
logger.setLevel(logging.INFO)


def apply_grayscale__sharpening__adaptive_median_filter__bilateral_smooth_filter(
    image_path: Path, ksize: int = 5, diameter: int = 9, sigma_color: int = 75, sigma_space: int = 75
):
    logger.info(
        "Apply grayscale, sharpening, adaptive median filter level '%d' and bilateral smooth filter diameter '%d' sigma_color '%d' sigma_space '%d' in %s",
        ksize,
        diameter,
        sigma_color,
        sigma_space,
        image_path,
    )

    alg_name = f"grayscale__sharpening__adaptive_median_filter_{ksize}__bilateral_smooth_filter_{diameter}_{sigma_color}_{sigma_space}"
    res_dir_image_path = get_algorithm_image_result_dir_path(image_path, alg_name)
    res_image_path = res_dir_image_path / image_path.name
    if res_image_path.exists():
        logger.info(
            "Grayscale, sharpening, adaptive median filter and bilateral smooth filter are already applied, ignoring..."
        )
        return
    copy_dataset_folder(image_path.parent.parent.parent, alg_name=alg_name, ignore_images=True)

    # Grayscale
    image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)

    # Sharpening
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    sharpened_image = cv2.filter2D(image, -1, kernel)

    # Adaptive median filter
    filtered_image = cv2.medianBlur(sharpened_image, ksize)
    cv2.imwrite(str(res_image_path), filtered_image)

    # Bilateral smooth filter
    filtered_image = cv2.bilateralFilter(filtered_image, diameter, sigma_color, sigma_space)
    cv2.imwrite(str(res_image_path), filtered_image)
