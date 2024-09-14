import logging
from pathlib import Path

import cv2
from PIL import Image

from dataset_utils import copy_dataset_folder, get_algorithm_image_result_dir_path

logger = logging.getLogger(__name__)
logging.basicConfig()
logger.setLevel(logging.INFO)


def apply_resize__hair_removal__gaussian_filter(image_path: Path, resize_value: tuple):
    logger.info("Rezising to '%s' and applying hair removal and gaussian filter in %s", resize_value, image_path)

    alg_name = f"resize_{resize_value[0]}x{resize_value[1]}__hair_removal__gaussian_filter"
    res_dir_image_path = get_algorithm_image_result_dir_path(image_path, alg_name)
    res_image_path = res_dir_image_path / image_path.name
    if res_image_path.exists():
        logger.info("Image is already resized and hair removal and gaussian filter are already applied, ignoring...")
        return
    copy_dataset_folder(image_path.parent.parent.parent, alg_name=alg_name, ignore_images=True)

    # Resize image
    with Image.open(image_path) as img:
        resized_img = img.resize(resize_value, Image.Resampling.LANCZOS)
        resized_img.save(res_image_path)

    """
    Created on Tue Feb 18 11:42:26 2020

    @author: Javier Velasquez P.

    DULL RAZOR (REMOVE HAIR)
    """
    resized_img = cv2.imread(str(res_image_path))
    # Gray scale
    grayScale = cv2.cvtColor(resized_img, cv2.COLOR_RGB2GRAY)
    # Black hat filter
    kernel = cv2.getStructuringElement(1, (9, 9))
    blackhat = cv2.morphologyEx(grayScale, cv2.MORPH_BLACKHAT, kernel)
    # Gaussian filter
    bhg = cv2.GaussianBlur(blackhat, (3, 3), cv2.BORDER_DEFAULT)
    # Binary thresholding (MASK)
    _, mask = cv2.threshold(bhg, 10, 255, cv2.THRESH_BINARY)
    # Replace pixels of the mask
    dst = cv2.inpaint(resized_img, mask, 6, cv2.INPAINT_TELEA)
    cv2.imwrite(str(res_image_path), dst)
