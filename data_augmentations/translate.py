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

DATA_AUGMENTATIONS_NAME = "translate__2x"
DATA_AUGMENTATIONS_RES_FILENAME_LABEL = "_translated"


# Use shift values from 0 to 1 otherwise the labels values shifting will need further robustness
def translate(image_path: Path, shift_x_perc: float = 0.2, shift_y_perc: float = 0.2):
    logger.info(
        "Translating to '%.1f%%' of width and '%.1f%%' of height in %s",
        shift_x_perc * 100,
        shift_y_perc * 100,
        image_path,
    )

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

    cv2.imwrite(str(image_path.parent / res_image_name), translated_image)
    cv2.imwrite(str(contoured_image_path.parent / res_image_name), translated_contoured_image)
    cv2.imwrite(str(masked_image_path.parent / res_image_name), translated_masked_image)

    ####################################
    # Update the labels for Yolov8n
    ####################################
    """ Something like this could also be done:

    labels_path = Path(f"datasets/Psoriasis-Detection---IS-20/train/labels/{image_name}.txt")
    labels = []
    with labels_path.open(mode="r", encoding="utf-8") as file:
        for line in file:
            # First item is the label name
            values = line.split()[1:]
            coordinates = tuple((int(float(values[i]) * width), int(float(values[i+1]) * height)) for i in range(0, len(values), 2))
            labels.append(coordinates)

    shifted_labels = []
    for label in labels:
        orig_labels = []
        left_labels = []
        top_labels = []
        top_left_labels = []
        for coordinate in label:
            x, y = coordinate
            orig_labels.append((x + shift_x, y + shift_y))
            # Left
            left_labels.append((-x + shift_x, y + shift_y))
            # Top
            top_labels.append((x + shift_x, -y + shift_y))
            # Top left
            top_left_labels.append((-x + shift_x, -y + shift_y))
        shifted_labels.append(orig_labels)
        shifted_labels.append(left_labels)
        shifted_labels.append(top_labels)
        shifted_labels.append(top_left_labels)
    """

    # Threshold the image to ensure it's binary (black and white)
    translated_gray_image = cv2.imread(str(masked_image_path.parent / res_image_name), cv2.IMREAD_GRAYSCALE)
    _, binary = cv2.threshold(translated_gray_image, 127, 255, cv2.THRESH_BINARY)

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
                for coord in [round(vertex[0] / width, 16), round(vertex[1] / height, 16)]
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

        # Translation of original, contoured and masked image
        translate(image_path)

        logger.info("done!\n\n")
