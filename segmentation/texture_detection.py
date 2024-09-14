import copy
import logging
from pathlib import Path
import re
import shutil
from typing import Any

import cv2 as cv

logger = logging.getLogger(__name__)
logging.basicConfig()
logger.setLevel(logging.INFO)

ROOT_DIR = Path().cwd()
DATASETS_DIR = ROOT_DIR / "datasets"

DATASET_PATTERN = re.compile(r".*/Psoriasis-Detection---IS-\d+/.*")


def texture_detection_algorithm(td_algorithm: Any, alg_name: str, image_path: Path):
    img_name = image_path.name
    img_name_without_suffix = image_path.stem
    img_ext = image_path.suffix

    # Create new dir for the algorithm results by copying the original dataset folder, because it contains other
    # important files required for the image segmentation model
    alg_results_path = DATASETS_DIR / f"{image_path.parent.parent.parent.name}_{alg_name}_texture_detection_algorithm"
    if not alg_results_path.exists():
        shutil.copytree(image_path.parent.parent.parent, alg_results_path, dirs_exist_ok=True)
    alg_results_path = alg_results_path / image_path.parent.parent.name / image_path.parent.name

    rgb_kp_img_name = f"{img_name_without_suffix}-rgb_key_points_image{img_ext}"
    gray_kp_img_name = f"{img_name_without_suffix}-gray_key_points_image{img_ext}"

    img = cv.imread(image_path)
    gray_img = cv.imread(image_path, cv.IMREAD_GRAYSCALE)

    img_rgb_kp = copy.copy(img)
    img_gray_kp = copy.copy(gray_img)

    keypoints = td_algorithm.detect(gray_img, None)
    # keypoints, descriptors = org.detectAndCompute(gray_img, None)

    # key_points_image = cv.drawKeypoints(gray_img, keypoints, None)
    rich_kp_image = cv.drawKeypoints(gray_img, keypoints, None, flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    cv.drawKeypoints(gray_img, keypoints, img_rgb_kp, flags=cv.DRAW_MATCHES_FLAGS_DRAW_OVER_OUTIMG)
    cv.drawKeypoints(gray_img, keypoints, img_gray_kp, flags=cv.DRAW_MATCHES_FLAGS_DRAW_OVER_OUTIMG)

    # Store images
    cv.imwrite(alg_results_path / img_name, img)
    cv.imwrite(alg_results_path / rgb_kp_img_name, img_rgb_kp)
    cv.imwrite(alg_results_path / gray_kp_img_name, img_gray_kp)


if __name__ == "__main__":
    images_path = DATASETS_DIR.rglob(str(Path("**", "images", "*")))
    for image_path in images_path:
        if not DATASET_PATTERN.search(str(image_path)):
            continue
        logger.info("Processing %s image...", image_path)
        # deprecated or something
        # texture_detection_algorithm(cv.xfeatures2d.SIFT_create(), "sift", image_path)
        texture_detection_algorithm(cv.ORB_create(nfeatures=99999999), "orb", image_path)
        logger.info("done!")
