import logging
from pathlib import Path

import cv2

from dataset_utils import copy_dataset_folder, get_algorithm_image_result_dir_path

logger = logging.getLogger(__name__)
logging.basicConfig()
logger.setLevel(logging.INFO)


def apply_adaptive_median_filter(image_path: Path, ksize: int = 5):
    """
    Common values are odd integers like 3, 5, or 7:
    3: Provides minimal smoothing, suitable for reducing light noise while retaining more details.
    5: Offers a moderate level of smoothing, useful for reducing medium-level noise.
    7 or higher: Applies stronger smoothing but may cause some loss of finer details.

    Args:
        image_path (Path): original image path
        ksize (int, optional): smoothing level. Defaults to 5.
    """
    logger.info("Applying adaptive median filter level '%d' in %s", ksize, image_path)

    alg_name = f"adaptive_median_filter_{ksize}"
    res_dir_image_path = get_algorithm_image_result_dir_path(image_path, alg_name)
    res_image_path = res_dir_image_path / image_path.name
    if res_image_path.exists():
        logger.info("Adaptive median filter is already applied, ignoring...")
        return
    copy_dataset_folder(image_path.parent.parent.parent, alg_name=alg_name, ignore_images=True)

    image = cv2.imread(str(image_path))
    filtered_image = cv2.medianBlur(image, ksize)
    cv2.imwrite(str(res_image_path), filtered_image)
