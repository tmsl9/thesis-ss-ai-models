import logging
from pathlib import Path

import cv2
from PIL import Image

from dataset_utils import copy_dataset_folder, get_algorithm_image_result_dir_path

logger = logging.getLogger(__name__)
logging.basicConfig()
logger.setLevel(logging.INFO)


def apply_resize__adaptive_median_filter__grayscale(image_path: Path, resize_value: tuple, ksize: int = 5):
    logger.info(
        "Rezising to '%s' and apply adaptive median filter level '%d' and grayscale in %s",
        resize_value,
        ksize,
        image_path,
    )

    alg_name = f"resize_{resize_value[0]}x{resize_value[1]}__adaptive_median_filter_{ksize}__grayscale"
    res_dir_image_path = get_algorithm_image_result_dir_path(image_path, alg_name)
    res_image_path = res_dir_image_path / image_path.name
    if res_image_path.exists():
        logger.info(
            "Image is already resized and adaptive median filter and grayscale are already applied, ignoring..."
        )
        return
    copy_dataset_folder(image_path.parent.parent.parent, alg_name=alg_name, ignore_images=True)

    # Resize image
    with Image.open(image_path) as img:
        resized_img = img.resize(resize_value, Image.Resampling.LANCZOS)
        resized_img.save(res_image_path)

    # Adaptive median filter
    resized_img = cv2.imread(str(res_image_path))
    filtered_image = cv2.medianBlur(resized_img, ksize)

    # Grayscale
    gray_image = cv2.cvtColor(filtered_image, cv2.COLOR_BGR2GRAY)
    cv2.imwrite(str(res_image_path), gray_image)
