import logging
from pathlib import Path

import cv2
from PIL import Image, ImageFilter

from dataset_utils import copy_dataset_folder, get_algorithm_image_result_dir_path

logger = logging.getLogger(__name__)
logging.basicConfig()
logger.setLevel(logging.INFO)


def apply_adaptive_median_filter__gaussian_filter__contrast_enhancement(image_path: Path, ksize: int = 5):
    logger.info(
        "Apply adaptive median filter level '%d', gaussian filter and contrast enhancement in %s", ksize, image_path
    )

    alg_name = f"adaptive_median_filter_{ksize}__gaussian_filter__contrast_enhancement"
    res_dir_image_path = get_algorithm_image_result_dir_path(image_path, alg_name)
    res_image_path = res_dir_image_path / image_path.name
    if res_image_path.exists():
        logger.info(
            "Image is already resized and adaptive median filter and grayscale are already applied, ignoring..."
        )
        return
    copy_dataset_folder(image_path.parent.parent.parent, alg_name=alg_name, ignore_images=True)

    # Adaptive median filter
    image = cv2.imread(str(image_path))
    filtered_image = cv2.medianBlur(image, ksize)
    cv2.imwrite(str(res_image_path), filtered_image)

    # Gaussian filter
    image = Image.open(res_image_path)
    # Crop the image
    smol_image = image.crop((0, 0, image.width, image.height))
    # Blur on the cropped image
    blurred_image = smol_image.filter(ImageFilter.GaussianBlur)
    # Paste the blurred image on the original image
    image.paste(blurred_image, (0, 0))
    # Store the result image
    image.save(res_image_path)

    # Contrast enhancement
    image = cv2.imread(str(res_image_path))
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l = cv2.equalizeHist(l)
    enhanced_image = cv2.merge((l, a, b))
    enhanced_image = cv2.cvtColor(enhanced_image, cv2.COLOR_LAB2BGR)
    cv2.imwrite(str(res_image_path), enhanced_image)
