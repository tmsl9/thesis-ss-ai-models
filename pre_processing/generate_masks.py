"""
Output:
- It will generate the following:
    - <Path>/**/contours/*.jpg
    - <Path>/**/masks/*.jpg
"""

import logging
from pathlib import Path

import cv2
import numpy as np

from pre_processing.utils.masking_utils import get_contours_coordinates, is_point_in_polygon

# %matplotlib inline  # if you are running this code in Jupyter notebook

logger = logging.getLogger(__name__)
logging.basicConfig()
logger.setLevel(logging.INFO)

BLACK_COLOR = (0, 0, 0)
WHITE_COLOR = (255, 255, 255)
THICKNESS = 1


def create_contours_n_masks(image_path: Path):
    """Based on lists of points which define contours on the images, it creates the contours and masks the images,
    these generated images are then stored on a proper folder"""
    logger.info("Generating contours and masks in %s", image_path)

    image_name = image_path.name
    img_label_name = image_path.with_suffix(".txt").name
    labels_path = image_path.parent.parent / "labels" / img_label_name
    contoured_image_path = image_path.parent.parent / "contours" / image_name
    masked_image_path = image_path.parent.parent / "masks" / image_name

    if contoured_image_path.exists() and masked_image_path.exists():
        logger.info("Contours and masks already exist, ignoring...")
        return

    image = cv2.imread(image_path)
    image_contour = image.copy()
    image_mask = image.copy()

    height = image.shape[0]
    width = image.shape[1]

    all_labels_coordinates = get_contours_coordinates(labels_path, width, height)
    for coordinates in all_labels_coordinates:
        for i in range(1, len(coordinates)):
            cv2.line(image_contour, coordinates[i - 1], coordinates[i], WHITE_COLOR, THICKNESS)

    # CREATE MASKS
    # Create an empty mask
    mask = np.zeros((height, width), dtype=np.uint8)

    # Fill the mask with the polygons
    for coordinates in all_labels_coordinates:
        cv2.fillPoly(mask, [np.array(coordinates, np.int32)], 255)  # Fill the polygon with white (255)

    # Apply the mask to the image
    image_mask[mask == 0] = BLACK_COLOR  # Set background pixels to black
    image_mask[mask != 0] = WHITE_COLOR  # Set lesion pixels to white

    # Create dirs
    contoured_image_path.parent.mkdir(parents=True, exist_ok=True)
    masked_image_path.parent.mkdir(parents=True, exist_ok=True)
    # Save images
    cv2.imwrite(contoured_image_path, image_contour)
    cv2.imwrite(masked_image_path, image_mask)
