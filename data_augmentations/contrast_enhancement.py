import cv2
import logging
import numpy as np
from pathlib import Path
import shutil
import sys

project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

from config import DATASETS_CONFIGURATION_DEFAULT
from dataset_utils import DATASET_PATTERN, DATASETS_DIR, load_dataset


logger = logging.getLogger(__name__)
logging.basicConfig()
logger.setLevel(logging.INFO)

DATA_AUGMENTATIONS_NAME = "contrast_enhancement__2x"
DATA_AUGMENTATIONS_RES_FILENAME_LABEL = "_contrast_enhanced"


def contrast_enhance(image_path: Path):
    logger.info("Contrast enhance %s", image_path)

    image_name = image_path.name
    res_image_name = image_path.stem + DATA_AUGMENTATIONS_RES_FILENAME_LABEL + image_path.suffix
    res_label_name = image_path.stem + f"{DATA_AUGMENTATIONS_RES_FILENAME_LABEL}.txt"
    contoured_image_path = image_path.parent.parent / "contours" / image_name
    masked_image_path = image_path.parent.parent / "masks" / image_name
    labels_path = image_path.parent.parent / "labels" / (image_path.stem + ".txt")

    image = cv2.imread(image_path)

    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l = cv2.equalizeHist(l)
    enhanced_image = cv2.merge((l, a, b))
    enhanced_image = cv2.cvtColor(enhanced_image, cv2.COLOR_LAB2BGR)

    cv2.imwrite(str(image_path.parent / res_image_name), enhanced_image)
    # Duplicate contoured image
    shutil.copy(contoured_image_path, contoured_image_path.parent / res_image_name)
    # Duplicate masked image
    shutil.copy(masked_image_path, masked_image_path.parent / res_image_name)
    shutil.copy(labels_path, labels_path.parent / res_label_name)


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

        # Contrast enhancement of original and duplication of contoured and masked image
        contrast_enhance(image_path)

        logger.info("done!\n\n")
