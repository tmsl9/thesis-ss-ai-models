import logging
import os
from pathlib import Path
import re
import shutil
import sys

# External libraries
from roboflow import Roboflow

from pre_processing.generate_masks import create_contours_n_masks

project_root = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(project_root))

logger = logging.getLogger(__name__)
logging.basicConfig()
logger.setLevel(logging.INFO)

ROOT_DIR = Path().cwd()
DATASETS_DIR = ROOT_DIR / "datasets"

DATASET_PATTERN = re.compile(r".*/Psoriasis-Detection---IS-\d+/.*")

def load_dataset(dataset_configuration: dict, pre_processing_technique: str = None):
    ########################################################################################
    # Get annotated dataset
    ########################################################################################
    # Create dataset dir
    DATASETS_DIR.mkdir(parents=True, exist_ok=True)
    # Enter in dataset dir
    current_location = os.getcwd()
    os.chdir(DATASETS_DIR)
    logger.info("Entered in: %s", DATASETS_DIR)

    logger.info("Downloading dataset:")
    rf = Roboflow(api_key="KmTAioKJexTi6Wkq9NQk")
    project = rf.workspace("thesis-zv0qn").project("psoriasis-detection-is")
    version = project.version(dataset_configuration["version"])
    dataset = version.download("yolov8")
    # Go to previous location
    os.chdir(current_location)
    logger.info("Entered in: %s", current_location)

    # Duplicate roboflow dataset. This is needed because some data augmentations need to be executed locally
    if new_version := dataset_configuration.get("dest_version"):
        # Search for the last occurrence of a number and replace it
        dest_dataset_name = re.sub(r"(\d+)(?!.*\d)", str(new_version), Path(dataset.location).name)
        logger.info("Duplicating dataset '%s' into '%s'", Path(dataset.location).name, dest_dataset_name)
        # Duplicate dataset folder
        copy_dataset_folder(dataset.location, dest_dataset_name=dest_dataset_name)
        dataset.location = Path(dataset.location).parent / dest_dataset_name

    # Generate contours and masks to the donwloaded dataset
    images_paths = list(Path(dataset.location).rglob(str(Path("**", "images", "*"))))
    for i, image_path in enumerate(images_paths, 1):
        logger.info("(%d/%d) Processing image: '%s' ...", i, len(images_paths), image_path)
        create_contours_n_masks(image_path)
        logger.info("done!\n\n")

    if pre_processing_technique:
        dataset.location = f"{dataset.location}_{pre_processing_technique}"

    return project, dataset

def get_algorithm_image_result_dir_path(image_path: Path, alg_name: str):
    alg_results_path = DATASETS_DIR / f"{image_path.parent.parent.parent.name}_{alg_name}"
    alg_results_path = alg_results_path / image_path.parent.parent.name / image_path.parent.name
    return alg_results_path


def ignore_images_files(directory, files):
    return [file for file in files if file.endswith(".jpg") and "images" in directory]

def ignore_test_images_files(directory, files):
    return [file for file in files if file.endswith(".jpg") and "images" in directory and "/test/" not in directory]

def ignore_contours_n_masks_images_files(directory, files):
    return [file for file in files if file.endswith(".jpg") and "images" in directory and ("/contours/" not in directory and "/masks/" not in directory)]


def copy_dataset_folder(dataset_path: Path, dest_dataset_name: str = None, alg_name: str = None, ignore_images: bool = False, ignore_test_images: bool = False, ignore_contours_n_masks: bool = False):
    # Create new dir for the algorithm results
    dest_dataset_path = DATASETS_DIR
    if alg_name:
        dest_dataset_path /= f"{dataset_path.name}_{alg_name}"
    elif dest_dataset_name:
        dest_dataset_path /= dest_dataset_name
    else:
        dest_dataset_path = dataset_path

    if dest_dataset_path.exists():
        logger.info("'%s' already exists, ignoring...", dest_dataset_path)
        return

    logger.info("Copying '%s' into '%s'", dataset_path, dest_dataset_path)
    ignore_images_files_func = None
    if ignore_test_images:
        ignore_images_files_func = ignore_test_images_files
    elif ignore_contours_n_masks:
        ignore_images_files_func = ignore_contours_n_masks_images_files
    elif ignore_images:
        ignore_images_files_func = ignore_images_files
    shutil.copytree(dataset_path, dest_dataset_path, dirs_exist_ok=True, ignore=ignore_images_files_func)
