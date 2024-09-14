import cv2
import logging
import numpy as np
from pathlib import Path
import sys

project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

from config import DATASETS_CONFIGURATION_DEFAULT
from dataset_utils import DATASET_PATTERN, DATASETS_DIR, load_dataset


logger = logging.getLogger(__name__)
logging.basicConfig()
logger.setLevel(logging.INFO)

BLACK_COLOR = (0, 0, 0)
WHITE_COLOR = (255, 255, 255)
THICKNESS = 1

LABELS_CLASS = "0"

DATA_AUGMENTATIONS_NAME = "scale__2x"
DATA_AUGMENTATIONS_RES_FILENAME_LABEL = "_scaled"


def scale(image_path: Path, scale_factor: float = 3):
    logger.info("Scaling in '%.1f'x in %s", scale_factor, image_path)

    image_name = image_path.name
    res_image_name = image_path.stem + DATA_AUGMENTATIONS_RES_FILENAME_LABEL + image_path.suffix
    res_label_name = image_path.stem + f"{DATA_AUGMENTATIONS_RES_FILENAME_LABEL}.txt"
    contoured_image_path = image_path.parent.parent / "contours" / image_name
    masked_image_path = image_path.parent.parent / "masks" / image_name
    res_labels_path = image_path.parent.parent / "labels" / res_label_name

    image = cv2.imread(image_path)
    contoured_image = cv2.imread(contoured_image_path)
    masked_image = cv2.imread(masked_image_path)

    height, width = image.shape[:2]

    # Calculate the new dimensions
    new_width = int(width * scale_factor)
    new_height = int(height * scale_factor)

    # Resize the original, contoured and masked image back to original size
    scaled_image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
    scaled_contoured_image = cv2.resize(contoured_image, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
    scaled_masked_image = cv2.resize(masked_image, (new_width, new_height), interpolation=cv2.INTER_LINEAR)

    cv2.imwrite(str(image_path.parent / res_image_name), scaled_image)
    cv2.imwrite(str(contoured_image_path.parent / res_image_name), scaled_contoured_image)
    cv2.imwrite(str(masked_image_path.parent / res_image_name), scaled_masked_image)

    ####################################
    # Update the labels for Yolov8n
    ####################################
    # Threshold the image to ensure it's binary (black and white)
    scaled_gray_image = cv2.imread(str(masked_image_path.parent / res_image_name), cv2.IMREAD_GRAYSCALE)
    _, binary = cv2.threshold(scaled_gray_image, 127, 255, cv2.THRESH_BINARY)

    # Find contours in the binary image
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Iterate over all contours found
    all_vertices = []
    with res_labels_path.open("w") as file:
        for contour in contours:
            # Approximate the contour to get vertices
            epsilon = 0.001 * cv2.arcLength(contour, True)  # Adjust the epsilon value for simplification
            approx = cv2.approxPolyDP(contour, epsilon, True)
            # Extract the vertices as (x, y) tuples
            vertices = [tuple(point[0]) for point in approx]
            all_vertices.append(vertices)
            label_format = [LABELS_CLASS] + [
                str(coord)
                for vertex in vertices
                for coord in [round(vertex[0] / new_width, 16), round(vertex[1] / new_height, 16)]
            ]
            file.write(" ".join(label_format) + "\n")


if __name__ == "__main__":
    # Download the roboflow datasets
    versions = []
    for dataset_configuration in DATASETS_CONFIGURATION_DEFAULT:
        if dataset_configuration["augmentations"] == DATA_AUGMENTATIONS_NAME:
            versions.append(dataset_configuration.get("dest_version") or dataset_configuration["version"])
            load_dataset(dataset_configuration)

    images_paths = list(DATASETS_DIR.rglob(str(Path("*", "train", "images", "*"))))
    images_paths = [
        image_path
        for image_path in images_paths
        if any(f"IS-{version}" in str(image_path) for version in versions)
        and DATASET_PATTERN.search(str(image_path))
        and DATA_AUGMENTATIONS_RES_FILENAME_LABEL not in str(image_path.name)
    ]

    for i, image_path in enumerate(images_paths, 1):
        logger.info("(%d/%d) Processing image: '%s' ...", i, len(images_paths), image_path)

        # Scaling of original, contoured and masked image
        scale(image_path)

        logger.info("done!\n\n")
