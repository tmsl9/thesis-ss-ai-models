import logging
from pathlib import Path
import sys

project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

from config import DATASETS_CONFIGURATION
from dataset_utils import DATASET_PATTERN, DATASETS_DIR, load_dataset

from pre_processing.adaptive_median_filter import apply_adaptive_median_filter
from pre_processing.background_masking import apply_background_masking
from pre_processing.bilateral_smooth_filter import apply_bilateral_smooth_filter
from pre_processing.color_histogram import apply_color_histogram
from pre_processing.contrast_enhancement import apply_contrast_enhancement
from pre_processing.dilation import apply_dilation
from pre_processing.erosion import apply_erosion
from pre_processing.gabor_filter import apply_gabor_filter
from pre_processing.gaussian_filter import apply_gaussian_filter
from pre_processing.generate_masks import create_contours_n_masks
from pre_processing.grayscale import apply_grayscale
from pre_processing.hair_removal import apply_hair_removal
from pre_processing.resize import resize_image
from pre_processing.sharpening import apply_sharpening

from pre_processing.adaptive_median_filter__gaussian_filter__contrast_enhancement import (
    apply_adaptive_median_filter__gaussian_filter__contrast_enhancement,
)
from pre_processing.erosion__dilation__gabor_filter__gaussian_filter import (
    apply_erosion__dilation__gabor_filter__gaussian_filter,
)
from pre_processing.grayscale__sharpening__adaptive_median_filter__bilateral_smooth_filter import (
    apply_grayscale__sharpening__adaptive_median_filter__bilateral_smooth_filter,
)
from pre_processing.resize__adaptive_median_filter__grayscale import apply_resize__adaptive_median_filter__grayscale
from pre_processing.resize__gaussian_filter__background_masking import (
    apply_resize__gaussian_filter__background_masking,
)
from pre_processing.resize__gaussian_filter import apply_resize__gaussian_filter
from pre_processing.resize__hair_removal__gaussian_filter import apply_resize__hair_removal__gaussian_filter

logger = logging.getLogger(__name__)
logging.basicConfig()
logger.setLevel(logging.INFO)

KSIZE_VALUES = (3, 5, 7)
RESIZE_VALUES = ((480, 480), (640, 640), (720, 720))
BILATERAL_SMOOTH_FILTER_VALUES = ((9, 75, 75), (15, 100, 100), (5, 50, 50))

if __name__ == "__main__":
    # Download the roboflow datasets
    for dataset_configuration in DATASETS_CONFIGURATION:
        load_dataset(dataset_configuration)

    versions = []
    for dataset_configuration in DATASETS_CONFIGURATION:
        versions.append(dataset_configuration.get("dest_version") or dataset_configuration["version"])

    images_paths = list(DATASETS_DIR.rglob(str(Path("**", "images", "*"))))
    images_paths = [
        image_path
        for image_path in images_paths
        if any(f"IS-{version}" in str(image_path) for version in versions) and DATASET_PATTERN.search(str(image_path))
    ]

    # Firstly create masks and contours in the original datasets. When the other techniques are executed,
    # a new dataset will be created based on the original ones.
    for i, image_path in enumerate(images_paths, 1):
        logger.info("(%d/%d) Processing image: '%s' ...", i, len(images_paths), image_path)
        create_contours_n_masks(image_path)
        logger.info("done!\n\n")

    for i, image_path in enumerate(images_paths, 1):
        logger.info("(%d/%d) Processing image: '%s' ...", i, len(images_paths), image_path)
        # Execute pre-processing techniques
        apply_background_masking(image_path)
        apply_color_histogram(image_path)
        apply_contrast_enhancement(image_path)
        apply_dilation(image_path)
        apply_erosion(image_path)
        apply_gaussian_filter(image_path)
        apply_grayscale(image_path)
        apply_hair_removal(image_path)
        apply_sharpening(image_path)
        for ksize in KSIZE_VALUES:
            apply_adaptive_median_filter(image_path, ksize)
        for filter in BILATERAL_SMOOTH_FILTER_VALUES:
            apply_bilateral_smooth_filter(image_path, *filter)
        for resize_value in RESIZE_VALUES:
            resize_image(image_path, resize_value)

        # Mixed techniques
        for resize_value in RESIZE_VALUES:
            for ksize in KSIZE_VALUES:
                apply_resize__adaptive_median_filter__grayscale(image_path, resize_value, ksize)
            apply_resize__hair_removal__gaussian_filter(image_path, resize_value)
            apply_resize__gaussian_filter(image_path, resize_value)
            apply_resize__gaussian_filter__background_masking(image_path, resize_value)
        for ksize in KSIZE_VALUES:
            apply_adaptive_median_filter__gaussian_filter__contrast_enhancement(image_path, ksize)
            for filter in BILATERAL_SMOOTH_FILTER_VALUES:
                apply_grayscale__sharpening__adaptive_median_filter__bilateral_smooth_filter(
                    image_path, ksize, *filter
                )
        apply_erosion__dilation__gabor_filter__gaussian_filter(image_path)
        apply_gabor_filter(image_path)

        logger.info("done!\n\n")
