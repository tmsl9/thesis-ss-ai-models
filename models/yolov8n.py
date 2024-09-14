from datetime import datetime, timedelta
import logging
import subprocess
from pathlib import Path
import sys
import time

# External libraries
from IPython.display import display, Image
from roboflow.core.dataset import Dataset
from roboflow.core.workspace import Workspace

logger = logging.getLogger(__name__)
logging.basicConfig()
logger.setLevel(logging.INFO)

MAX_RETRIES = 10


def _get_last_train_model(results_dir: Path):
    train_model_paths = list(results_dir.glob(str(Path("runs", "segment", "train*", "weights", "last.pt"))))
    return train_model_paths[-1] if train_model_paths else None


def run_yolov8n(
    results_dir: Path, project: Workspace, dataset: Dataset, epochs: int, learning_rate: float, train_imgsize: tuple
):
    ########################################################################################
    # Custom Training
    ########################################################################################

    retry_count = 0
    resume = True if _get_last_train_model(results_dir) else False
    train_model_rel_path = _get_last_train_model(results_dir) or "yolov8s-seg.pt"
    timer_start = datetime.now()
    while retry_count < MAX_RETRIES:
        try:
            logger.info("Training model:")
            cmd = [
                "yolo",
                "task=segment",
                "mode=train",
                f"model={train_model_rel_path}",
                f"data={dataset.location}/data.yaml",
                f"epochs={epochs}",
                f"imgsz={train_imgsize}",
                f"lr0={learning_rate}",
            ]
            if resume:
                cmd.append("resume")
            subprocess.run(cmd, check=True)
            # Success
            break
        except subprocess.CalledProcessError as e:
            # In case the model is to be resumed but it's already fully trained, the process will give an exception as
            # well. However we cannot compare the exception message.
            logger.exception("An unexpected error occurred: %s", e)
            # Add option to resume from last train checkpoint
            resume = True
            # Check last train model path
            train_model_rel_path = _get_last_train_model(results_dir)
            retry_count += 1
            if retry_count == MAX_RETRIES:
                # Stop execution because the process keeps raising exceptions
                logger.exception("Stop execution because the process keeps raising exceptions")
                sys.exit()
            logger.info("Retry model training.")
            time.sleep(2)

    # Store time spent in model operation
    with (results_dir / f"process_time_train.txt").open(mode="w") as file:
        # Remove 2 seconds of each retry sleep
        timer_end = datetime.now() - timedelta(seconds=2 * retry_count)
        file.write(f"Start: {timer_start}\n")
        file.write(f"End: {timer_end}\n")
        file.write(f"Duration: {timer_end - timer_start}\n")

    if (results_dir.parent.parent / "runs").exists() and not (results_dir / "runs").exists():
        cmd = ["mv", results_dir.parent.parent / "runs", results_dir / "runs"]
        subprocess.run(cmd, check=True)

    cmd = ["ls", results_dir / "runs" / "segment" / "train"]
    subprocess.run(cmd, check=True)

    # Image(filename=workdir / "runs" / "segment" / "train" / "confusion_matrix.png", width=600)
    # Image(filename=workdir / "runs" / "segment" / "train" / "results.png", width=600)
    # Image(filename=workdir / "runs" / "segment" / "train" / "val_batch0_pred.jpg", width=600)

    ########################################################################################
    # Validate Custom Model
    ########################################################################################
    # This is the usual operation in order to validate the model performance,
    # instead of using prediction operation. So we will change the validation to use test images
    # to be fair to compare with other models, which predict with the test test images.
    file_content = []
    with Path(dataset.location, "data.yaml").open(mode="r") as file:
        for line in file:
            if line.startswith("test:"):
                line = "test: valid/images\n"
            elif line.startswith("val:"):
                line = "val: test/images\n"
            file_content.append(line)

    with Path(dataset.location, "data.yaml").open(mode="w") as file:
        file.writelines(file_content)

    retry_count = 0
    timer_start = datetime.now()
    while retry_count < MAX_RETRIES:
        try:
            logger.info("Validating model:")
            cmd = [
                "yolo",
                "task=segment",
                "mode=val",
                f"model={results_dir}/runs/segment/train/weights/best.pt",
                f"data={dataset.location}/data.yaml",
            ]
            result = subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            # Success
            break
        except subprocess.CalledProcessError as e:
            logger.exception("An unexpected error occurred: %s", e)
            retry_count += 1
            if retry_count == MAX_RETRIES:
                # Stop execution because the process keeps raising exceptions
                logger.exception("Stop execution because the process keeps raising exceptions")
                sys.exit()
            logger.info("Retry model validation.")
            time.sleep(2)

    # Store time spent in model operation
    with (results_dir / f"process_time_predict.txt").open(mode="w") as file:
        # Remove 2 seconds of each retry sleep
        timer_end = datetime.now() - timedelta(seconds=2 * retry_count)
        file.write(f"Start: {timer_start}\n")
        file.write(f"End: {timer_end}\n")
        file.write(f"Duration: {timer_end - timer_start}\n")

    with (results_dir / "results.txt").open(mode="w") as file:
        results = result.stdout.decode() or result.stderr.decode()
        logger.info("Prediction results: %s", results)
        file.write(f"{results}\n")
        logger.info("Results stored in: %s", results_dir / "results.txt")

    ########################################################################################
    # Inference with Custom Model
    ########################################################################################
    """
    retry_count = 0
    while retry_count < MAX_RETRIES:
        try:
            logger.info("Predicting model:")
            cmd = [
                "yolo",
                "task=segment",
                "mode=predict",
                f"model={results_dir}/runs/segment/train/weights/best.pt",
                "conf=0.25",
                f"source={dataset.location}/test/images",
            ]
            result = subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            # Success
            break
        except subprocess.CalledProcessError as e:
            logger.exception("An unexpected error occurred: %s", e)
            retry_count += 1
            if retry_count == MAX_RETRIES:
                # Stop execution because the process keeps raising exceptions
                logger.exception("Stop execution because the process keeps raising exceptions")
                sys.exit()
            logger.info("Retry model prediction.")
            time.sleep(2)
    """

    # Show last 3 images
    # for image_path in workdir.glob(str(Path("runs" / "segment" / "predict*" / "*.jpg")))[:3]:
    #     display(Image(filename=image_path, height=600))

    # Save & Deploy model
    # Check in https://universe.roboflow.com/thesis-zv0qn/psoriasis-detection-is/dataset/
    # project.version(dataset.version).deploy(model_type="yolov8-seg", model_path=workdir / "runs" / "segment" / "train")
