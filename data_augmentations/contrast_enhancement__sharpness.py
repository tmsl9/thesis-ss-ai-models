import cv2
import logging
import numpy as np
from pathlib import Path
import re
import shutil
import sys

project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

from config import DATASETS_CONFIGURATION_DEFAULT, dataset_45_imgs_no_augmentations, dataset_63_imgs_no_augmentations
from dataset_utils import DATASET_PATTERN, DATASETS_DIR, load_dataset


logger = logging.getLogger(__name__)
logging.basicConfig()
logger.setLevel(logging.INFO)

DATA_AUGMENTATIONS_NAME = "contrast_enhancement__sharp"


def contrast_enhance__sharp(image_path: Path):
    logger.info("Contrast enhance and sharp %s", image_path)

    image = cv2.imread(image_path)

    ##########################
    # Sharp
    ##########################
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    sharpened_image = cv2.filter2D(image, -1, kernel)

    ##########################
    # Contrast enhancement
    ##########################
    lab = cv2.cvtColor(sharpened_image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l = cv2.equalizeHist(l)
    enhanced_image = cv2.merge((l, a, b))
    enhanced_image = cv2.cvtColor(enhanced_image, cv2.COLOR_LAB2BGR)

    cv2.imwrite(str(image_path), sharpened_image)


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

        # Contrast enhancement and sharping of original and duplication of contoured and masked image
        contrast_enhance__sharp(image_path)

        logger.info("done!\n\n")
