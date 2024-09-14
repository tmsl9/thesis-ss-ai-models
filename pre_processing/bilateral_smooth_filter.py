import logging
from pathlib import Path

import cv2

from dataset_utils import copy_dataset_folder, get_algorithm_image_result_dir_path

logger = logging.getLogger(__name__)
logging.basicConfig()
logger.setLevel(logging.INFO)


def apply_bilateral_smooth_filter(image_path: Path, diameter: int = 9, sigma_color: int = 75, sigma_space: int = 75):
    """
    Apply a bilateral smooth filter to the image.

    Commonly Used Combinations
        For moderate smoothing:
            diameter = 9
            sigma_color = 75
            sigma_space = 75
        For stronger smoothing:
            diameter = 15
            sigma_color = 100
            sigma_space = 100
        For minimal smoothing (preserving more details):
            diameter = 5
            sigma_color = 50
            sigma_space = 50

    Parameters:
        image_path (str): The path to the input image.
        diameter (int): Diameter of each pixel neighborhood used during filtering.
        sigma_color (float): Filter sigma in the color space. Larger values mean that distant colors will mix together.
        sigma_space (float): Filter sigma in the coordinate space. Larger values mean that farther pixels will influence each other.
    """
    logger.info(
        "Applying bilateral smooth filter diameter '%d' sigma_color '%d' sigma_space '%d' in %s",
        diameter,
        sigma_color,
        sigma_space,
        image_path,
    )

    alg_name = f"bilateral_smooth_filter_{diameter}_{sigma_color}_{sigma_space}"
    res_dir_image_path = get_algorithm_image_result_dir_path(image_path, alg_name)
    res_image_path = res_dir_image_path / image_path.name
    if res_image_path.exists():
        logger.info("Bilateral smooth filter is already applied, ignoring...")
        return
    copy_dataset_folder(image_path.parent.parent.parent, alg_name=alg_name, ignore_images=True)

    image = cv2.imread(str(image_path))
    filtered_image = cv2.bilateralFilter(image, diameter, sigma_color, sigma_space)
    cv2.imwrite(str(res_image_path), filtered_image)
