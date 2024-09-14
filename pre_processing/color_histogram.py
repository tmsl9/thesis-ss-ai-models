import logging
from pathlib import Path

import cv2

from dataset_utils import copy_dataset_folder, get_algorithm_image_result_dir_path

logger = logging.getLogger(__name__)
logging.basicConfig()
logger.setLevel(logging.INFO)


def apply_color_histogram(image_path: Path):
    logger.info("Applying color histogram in %s", image_path)

    alg_name = "color_histogram"
    res_dir_image_path = get_algorithm_image_result_dir_path(image_path, alg_name)
    res_image_path = res_dir_image_path / image_path.name
    if res_image_path.exists():
        logger.info("Color histogram is already applied, ignoring...")
        return
    copy_dataset_folder(image_path.parent.parent.parent, alg_name=alg_name, ignore_images=True)

    image = cv2.imread(str(image_path))

    # Convert the image from BGR to YCrCb color space
    ycrcb_img = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)

    # Split into channels
    y_channel, cr_channel, cb_channel = cv2.split(ycrcb_img)

    # Apply histogram equalization only on the Y channel (luminance)
    y_channel_eq = cv2.equalizeHist(y_channel)

    # Merge the channels back
    ycrcb_img_eq = cv2.merge((y_channel_eq, cr_channel, cb_channel))

    # Convert back to BGR color space
    result_image = cv2.cvtColor(ycrcb_img_eq, cv2.COLOR_YCrCb2BGR)
    cv2.imwrite(str(res_image_path), result_image)
