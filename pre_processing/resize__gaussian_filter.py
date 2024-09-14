import logging
from pathlib import Path

from PIL import Image, ImageFilter

from dataset_utils import copy_dataset_folder, get_algorithm_image_result_dir_path

logger = logging.getLogger(__name__)
logging.basicConfig()
logger.setLevel(logging.INFO)


def apply_resize__gaussian_filter(image_path: Path, resize_value: tuple):
    logger.info("Rezising to '%s' and applying gaussian filter in %s", resize_value, image_path)

    alg_name = f"resize_{resize_value[0]}x{resize_value[1]}__gaussian_filter"
    res_dir_image_path = get_algorithm_image_result_dir_path(image_path, alg_name)
    res_image_path = res_dir_image_path / image_path.name
    if res_image_path.exists():
        logger.info("Image is already resized and gaussian filter is already applied, ignoring...")
        return
    copy_dataset_folder(image_path.parent.parent.parent, alg_name=alg_name, ignore_images=True)

    with Image.open(image_path) as img:
        # Resize image
        resized_img = img.resize(resize_value, Image.Resampling.LANCZOS)

        # Gaussian filter
        # Crop the image
        smol_image = resized_img.crop((0, 0, resized_img.width, resized_img.height))
        # Blur on the cropped image
        blurred_image = smol_image.filter(ImageFilter.GaussianBlur)
        # Paste the blurred image on the original image
        resized_img.paste(blurred_image, (0, 0))
        # Store the result image
        resized_img.save(res_image_path)
