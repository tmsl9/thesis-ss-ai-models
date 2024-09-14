import logging
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageFilter

from dataset_utils import copy_dataset_folder, get_algorithm_image_result_dir_path
from pre_processing.utils.masking_utils import get_contours_coordinates

logger = logging.getLogger(__name__)
logging.basicConfig()
logger.setLevel(logging.INFO)

BLACK_COLOR = (0, 0, 0)


def apply_resize__gaussian_filter__background_masking(image_path: Path, resize_value: tuple):
    logger.info("Rezising to '%s' and applying gaussian filter and background masking in %s", resize_value, image_path)

    alg_name = f"resize_{resize_value[0]}x{resize_value[1]}__gaussian_filter__background_masking"
    res_dir_image_path = get_algorithm_image_result_dir_path(image_path, alg_name)
    res_image_path = res_dir_image_path / image_path.name
    if res_image_path.exists():
        logger.info(
            "Image is already resized and gaussian filter and background masking are already applied, ignoring..."
        )
        return
    copy_dataset_folder(image_path.parent.parent.parent, alg_name=alg_name, ignore_images=True)

    # In a realistic scenario of a mobile application for example, the user captures an image and sends it to the
    # backend which runs the model with the image. The backend will run some pre-processing techniques in the input
    # image, but how will it run background masking or region of interest if it doesn't know where the lesion is in
    # the image? That's what the AI model will identify.
    if "/test/" not in str(image_path):
        # Background masking
        img_label_name = image_path.with_suffix(".txt").name
        labels_path = image_path.parent.parent / "labels" / img_label_name

        image = cv2.imread(str(image_path))
        height, width = image.shape[:2]

        # Create an empty mask
        mask = np.zeros((height, width), dtype=np.uint8)

        # Fill the mask with the polygons
        all_labels_coordinates = get_contours_coordinates(labels_path, width, height)
        for coordinates in all_labels_coordinates:
            cv2.fillPoly(mask, [np.array(coordinates, np.int32)], 255)  # Fill the polygon with white (255)

        # Apply the mask to the image
        image[mask == 0] = BLACK_COLOR  # Set background pixels to black

        cv2.imwrite(str(res_image_path), image)
        image_path = res_image_path

    with Image.open(image_path) as img:
        # Resize image
        resized_img = img.resize(resize_value, Image.Resampling.LANCZOS)
        resized_img.save(res_image_path)

        # Gaussian filter
        # Crop the image
        smol_image = resized_img.crop((0, 0, resized_img.width, resized_img.height))
        # Blur on the cropped image
        blurred_image = smol_image.filter(ImageFilter.GaussianBlur)
        # Paste the blurred image on the original image
        resized_img.paste(blurred_image, (0, 0))
        # Store the result image
        resized_img.save(res_image_path)
