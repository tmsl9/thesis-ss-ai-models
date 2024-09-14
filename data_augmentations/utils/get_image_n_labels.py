from pathlib import Path

import cv2

from pre_processing.utils.masking_utils import get_contours_coordinates


def get_image_n_labels(image_path: Path):
    img_label_name = image_path.with_suffix(".txt").name
    labels_path = image_path.parent.parent / "labels" / img_label_name

    image = cv2.imread(image_path)

    height = image.shape[0]
    width = image.shape[1]

    all_labels_coordinates = get_contours_coordinates(labels_path, width, height)
    return all_labels_coordinates
