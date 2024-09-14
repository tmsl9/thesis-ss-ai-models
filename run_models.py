from datetime import datetime
import logging
import os
from pathlib import Path
import shutil

# Local packages
from config import (
    DATASETS_CONFIGURATION,
    EPOCHS,
    LEARNING_RATE,
    MODELS,
    MODELS_TRAIN_IMGSIZE,
    PRE_PROCESSING_TECHNIQUES,
)
from dataset_utils import load_dataset
from models.bisenet import run_bisenet
from models.deeplabv3plus import run_deeplabv3plus
from models.fcn import run_fcn
from models.hrnet import run_hrnet
from models.mask_rcnn import run_mask_rcnn
from models.pspnet import run_pspnet
from models.segnet import run_segnet
from models.unet import run_unet
from models.yolov8n import run_yolov8n
from models.utils.configure_gpu import configure_gpu_memory

logger = logging.getLogger(__name__)
logging.basicConfig()
logger.setLevel(logging.INFO)

ROOT_DIR = Path().cwd()
DATASETS_DIR = ROOT_DIR / "datasets"
RESULTS_DIR = ROOT_DIR / "results"


def _parse_train_imgsize(train_imgsize: str):
    if "," in str(train_imgsize):
        train_imgsize = train_imgsize.split(",")
    elif "x" in str(train_imgsize):
        train_imgsize = train_imgsize.split("x")
    else:
        train_imgsize = (train_imgsize, train_imgsize)
    train_imgsize = (int(train_imgsize[0]), int(train_imgsize[1]))
    return train_imgsize


if __name__ == "__main__":
    count = 0
    total_runs = (
        len(MODELS)
        * len(EPOCHS)
        * len(LEARNING_RATE)
        * len(MODELS_TRAIN_IMGSIZE)
        * len(PRE_PROCESSING_TECHNIQUES)
        * len(DATASETS_CONFIGURATION)
    )

    runs_configurations = {
        "-".join(
            (
                f"{epochs}_epochs",
                f"{learning_rate}_learningRate",
                f"{train_imgsize}_trainImgsize",
                f"{dataset_configuration['images']}_images",
                f"{pre_processing_technique or 'no'}_preprocessing",
                f"{dataset_configuration['augmentations'] or 'no'}_augmentations",
            )
        ): [
            epochs,
            learning_rate,
            _parse_train_imgsize(train_imgsize),
            dataset_configuration,
            pre_processing_technique,
        ]
        for epochs in EPOCHS
        for learning_rate in LEARNING_RATE
        for train_imgsize in MODELS_TRAIN_IMGSIZE
        for pre_processing_technique in PRE_PROCESSING_TECHNIQUES
        for dataset_configuration in DATASETS_CONFIGURATION
    }
    total_runs_to_exec = total_runs
    for model in MODELS:
        for run_configuration_name in runs_configurations:
            if (RESULTS_DIR / f"{model}_model-{run_configuration_name}" / "results.txt").exists():
                total_runs_to_exec -= 1
    logger.info("Will execute %d runs.", total_runs_to_exec)

    ########################################################################################
    # Run model(s)
    ########################################################################################
    total_timer_start = datetime.now()
    for model in MODELS:
        model_timer_start = datetime.now()

        # Configure GPU
        configure_gpu_memory(model, buffer_percent=1)
        for run_configuration_name, run_configuration in runs_configurations.items():
            count += 1
            run_configuration_name = f"{model}_model-{run_configuration_name}"

            logger.info("\n\n\n\n%d/%d: %s\n", count, total_runs, run_configuration_name)

            epochs, learning_rate, train_imgsize, dataset_configuration, pre_processing_technique = run_configuration

            ###########################
            # CHECK IF USE-CASE WAS ALREADY TRAINED AND VALIDATED
            ###########################
            results_dir = RESULTS_DIR / run_configuration_name
            # Ignore use-case because it was already trained and validated
            if (results_dir / "results.txt").exists():
                logger.info("Use-case already trained and validated. Ignoring...\n")
                continue

            total_runs_to_exec -= 1
            logger.info("\n\nStill missing %d runs\n", total_runs_to_exec)

            ###########################
            # LOAD DATASET
            ###########################
            project, dataset = load_dataset(dataset_configuration, pre_processing_technique)

            ###########################
            # ENTER IN RESULTS DIR
            ###########################
            # Clean dir
            if results_dir.exists():
                shutil.rmtree(results_dir)
            # Create results dir
            results_dir.mkdir(parents=True, exist_ok=True)
            # Enter in results dir
            os.chdir(results_dir)
            logger.info("Entered in: %s", results_dir)

            ###########################
            # RUN THE MODEL
            ###########################
            timer_start = datetime.now()
            if model == "BiseNet":
                run_bisenet(results_dir, dataset, epochs, learning_rate, train_imgsize)
            elif model == "DeepLabv3plus":
                run_deeplabv3plus(results_dir, dataset, epochs, learning_rate, train_imgsize)
            elif model == "FCN":
                run_fcn(results_dir, dataset, epochs, learning_rate, train_imgsize)
            elif model == "HRNet":
                run_hrnet(results_dir, dataset, epochs, learning_rate, train_imgsize)
            elif model == "Mask_RCNN":
                run_mask_rcnn(results_dir, dataset, epochs, learning_rate, train_imgsize)
            elif model == "PSPNet":
                run_pspnet(results_dir, dataset, epochs, learning_rate, train_imgsize)
            elif model == "SegNet":
                run_segnet(results_dir, dataset, epochs, learning_rate, train_imgsize)
            elif model == "UNet":
                run_unet(results_dir, dataset, epochs, learning_rate, train_imgsize)
            elif model == "YOLOv8n":
                run_yolov8n(results_dir, project, dataset, epochs, learning_rate, train_imgsize)

            with (RESULTS_DIR / run_configuration_name / "process_time.txt").open(mode="w") as file:
                timer_end = datetime.now()
                file.write(f"Total Start: {total_timer_start}\n")
                file.write(f"Model Start: {model_timer_start}\n")
                file.write(f"Start: {timer_start}\n")
                file.write(f"End: {timer_end}\n")
                file.write(f"Duration: {timer_end - timer_start}\n")

        with (ROOT_DIR / f"process_time_{model}.txt").open(mode="w") as file:
            model_timer_end = datetime.now()
            file.write(f"Total Start: {total_timer_start}\n")
            file.write(f"Start: {model_timer_start}\n")
            file.write(f"End: {model_timer_end}\n")
            file.write(f"Duration: {model_timer_end - model_timer_start}\n")

    with (ROOT_DIR / "process_time.txt").open(mode="w") as file:
        total_timer_end = datetime.now()
        file.write(f"Start: {total_timer_start}\n")
        file.write(f"End: {total_timer_end}\n")
        file.write(f"Duration: {total_timer_end - total_timer_start}\n")
