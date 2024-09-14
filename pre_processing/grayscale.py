import logging
from pathlib import Path

import cv2

from dataset_utils import copy_dataset_folder, get_algorithm_image_result_dir_path

logger = logging.getLogger(__name__)
logging.basicConfig()
logger.setLevel(logging.INFO)


def apply_grayscale(image_path: Path):
    logger.info("Applying grayscale in %s", image_path)

    alg_name = "grayscale"
    res_dir_image_path = get_algorithm_image_result_dir_path(image_path, alg_name)
    res_image_path = res_dir_image_path / image_path.name
    if res_image_path.exists():
        logger.info("Grayscale is already applied, ignoring...")
        return
    copy_dataset_folder(image_path.parent.parent.parent, alg_name=alg_name, ignore_images=True)

    gray_image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    cv2.imwrite(str(res_image_path), gray_image)
