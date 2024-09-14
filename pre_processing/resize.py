import logging
from pathlib import Path

from PIL import Image

from dataset_utils import copy_dataset_folder, get_algorithm_image_result_dir_path

logger = logging.getLogger(__name__)
logging.basicConfig()
logger.setLevel(logging.INFO)


def resize_image(image_path: Path, resize_value: tuple) -> None:
    """
    Resize an image to the specified dimensions while maintaining its aspect ratio.

    Parameters:
        image_path (str or Path): The path to the image to be resized.
        resize_value (tuple): A tuple (width, height) specifying the new size.

    Returns:
        None: The resized image will be saved in place.
    """
    logger.info("Rezising to '%s' %s", resize_value, image_path)

    alg_name = f"resize_{resize_value[0]}x{resize_value[1]}"
    res_dir_image_path = get_algorithm_image_result_dir_path(image_path, alg_name)
    res_image_path = res_dir_image_path / image_path.name
    if res_image_path.exists():
        logger.info("Resized image already exists, ignoring...")
        return
    copy_dataset_folder(image_path.parent.parent.parent, alg_name=alg_name, ignore_images=True)

    with Image.open(image_path) as img:
        resized_img = img.resize(resize_value, Image.Resampling.LANCZOS)
        resized_img.save(res_image_path)
