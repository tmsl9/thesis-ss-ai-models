import logging
from pathlib import Path

from PIL import Image, ImageFilter

from dataset_utils import copy_dataset_folder, get_algorithm_image_result_dir_path

logger = logging.getLogger(__name__)
logging.basicConfig()
logger.setLevel(logging.INFO)


def apply_gaussian_filter(image_path: Path):
    logger.info("Applying gaussian filter in %s", image_path)

    alg_name = "gaussian_filter"
    res_dir_image_path = get_algorithm_image_result_dir_path(image_path, alg_name)
    res_image_path = res_dir_image_path / image_path.name
    if res_image_path.exists():
        logger.info("Gaussian filter is already applied, ignoring...")
        return
    copy_dataset_folder(image_path.parent.parent.parent, alg_name=alg_name, ignore_images=True)

    image = Image.open(image_path)

    # Crop the image
    smol_image = image.crop((0, 0, image.width, image.height))

    # Blur on the cropped image
    blurred_image = smol_image.filter(ImageFilter.GaussianBlur)

    # Paste the blurred image on the original image
    image.paste(blurred_image, (0, 0))

    # Store the result image
    image.save(res_image_path)
