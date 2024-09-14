import cv2
import logging
import numpy as np
from pathlib import Path
import re
import sys

project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

from config import DATASETS_CONFIGURATION_DEFAULT, dataset_45_imgs_no_augmentations, dataset_63_imgs_no_augmentations
from dataset_utils import DATASET_PATTERN, DATASETS_DIR, load_dataset


logger = logging.getLogger(__name__)
logging.basicConfig()
logger.setLevel(logging.INFO)

BLACK_COLOR = (0, 0, 0)
WHITE_COLOR = (255, 255, 255)
THICKNESS = 1

LABELS_CLASS = "0"

DATA_AUGMENTATIONS_NAME = "translate__zoom_in"


# Use shift values from 0 to 1 otherwise the labels values shifting will need further robustness
def translate__zoom_in(
    image_path: Path, shift_x_perc: float = 0.2, shift_y_perc: float = 0.2, zoom_factor: float = 2.0
):
    logger.info(
        "Translating to '%.1f%%' of width and '%.1f%%' of height and zoom in '%.1f'x in %s",
        shift_x_perc * 100,
        shift_y_perc * 100,
        zoom_factor,
        image_path,
    )

    image_name = image_path.name
    contoured_image_path = image_path.parent.parent / "contours" / image_name
    masked_image_path = image_path.parent.parent / "masks" / image_name
    labels_path = image_path.parent.parent / "labels" / image_name

    image = cv2.imread(image_path)
    contoured_image = cv2.imread(contoured_image_path)
    masked_image = cv2.imread(masked_image_path)

    height, width = image.shape[:2]

    ##########################
    # Translate
    ##########################
    shift_x = int(shift_x_perc * width)
    shift_y = int(shift_y_perc * height)

    # Define the translation matrix
    translation_matrix = np.float32([[1, 0, shift_x], [0, 1, shift_y]])

    # Translate the original, contoured and masked image
    translated_image = cv2.warpAffine(image, translation_matrix, (width, height), borderMode=cv2.BORDER_REFLECT)
    translated_contoured_image = cv2.warpAffine(
        contoured_image, translation_matrix, (width, height), borderMode=cv2.BORDER_REFLECT
    )
    translated_masked_image = cv2.warpAffine(
        masked_image, translation_matrix, (width, height), borderMode=cv2.BORDER_REFLECT
    )

    ##########################
    # Zoom in
    ##########################
    # Calculate the cropping box to zoom into the center
    new_width = int(width / zoom_factor)
    new_height = int(height / zoom_factor)

    left = (width - new_width) // 2
    top = (height - new_height) // 2
    right = left + new_width
    bottom = top + new_height

    # Crop the image
    cropped_image = translated_image[top:bottom, left:right]
    cropped_contoured_image = translated_contoured_image[top:bottom, left:right]
    cropped_masked_image = translated_masked_image[top:bottom, left:right]

    # Resize the original, contoured and masked image back to original size
    zoomed_image = cv2.resize(cropped_image, (width, height), interpolation=cv2.INTER_LINEAR)
    zoomed_contoured_image = cv2.resize(cropped_contoured_image, (width, height), interpolation=cv2.INTER_LINEAR)
    zoomed_masked_image = cv2.resize(cropped_masked_image, (width, height), interpolation=cv2.INTER_LINEAR)

    cv2.imwrite(str(image_path), zoomed_image)
    cv2.imwrite(str(contoured_image_path), zoomed_contoured_image)
    cv2.imwrite(str(masked_image_path), zoomed_masked_image)

    ####################################
    # Update the labels for Yolov8n
    ####################################
    # Threshold the image to ensure it's binary (black and white)
    zoomed_gray_image = cv2.imread(str(masked_image_path), cv2.IMREAD_GRAYSCALE)
    _, binary = cv2.threshold(zoomed_gray_image, 127, 255, cv2.THRESH_BINARY)

    # Find contours in the binary image
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Iterate over all contours found
    all_vertices = []
    with labels_path.open("w") as file:
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
                for coord in [round(vertex[0] / width, 16), round(vertex[1] / height, 16)]
            ]
            file.write(" ".join(label_format) + "\n")


if __name__ == "__main__":
    # Download the roboflow datasets
    versions = []
    for dataset_configuration in DATASETS_CONFIGURATION_DEFAULT:
        if dataset_configuration.get("manual_augmentations") == DATA_AUGMENTATIONS_NAME:
            versions.append(dataset_configuration.get("dest_version") or dataset_configuration["version"])
            load_dataset(dataset_configuration)

    images_paths = []
    for image_path in list(DATASETS_DIR.rglob(str(Path("*", "train", "images", "*")))):
        original_img = False
        if any(f"IS-{version}" in str(image_path) for version in versions) and DATASET_PATTERN.search(str(image_path)):
            for default_dataset_version in [
                dataset_45_imgs_no_augmentations["version"],
                dataset_63_imgs_no_augmentations["version"],
            ]:
                default_dataset_path = re.sub(
                    r"(\d+)(?!.*\d)", str(default_dataset_version), str(image_path.parent.parent.parent)
                )
                if Path(
                    default_dataset_path, image_path.parent.parent.name, image_path.parent.name, image_path.name
                ).exists():
                    original_img = True
                    break
            if not original_img:
                images_paths.append(image_path)

    for i, image_path in enumerate(images_paths, 1):
        logger.info("(%d/%d) Processing image: '%s' ...", i, len(images_paths), image_path)

        # Translation and zoom in of original, contoured and masked image
        translate__zoom_in(image_path)

        logger.info("done!\n\n")
