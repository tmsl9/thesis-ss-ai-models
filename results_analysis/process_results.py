from datetime import datetime
import logging
from pathlib import Path

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

ROOT_DIR = Path().cwd()
RESULTS_DIR = ROOT_DIR / "results"

FULL_REPORT_CSV_PATH = ROOT_DIR / "full_report.csv"
# Box metrics are not important because the goal is to identify pixels, not detect objects
REPORT_HEADERS = (
    "MODEL",
    "EPOCHS",
    "LEARNING_RATE",
    "TRAINING_IMGSIZE",
    "TOTAL_IMAGES",
    "PREPROCESSING",
    "AUGMENTATIONS",
    "PREDICT_IMAGES",
    # "INSTANCES",
    # "BOX(PRECISION)",
    # "BOX(RECALL)",
    # "BOX(mAP50)",
    # "BOX(mAP50-95)",
    "PRECISION",
    "RECALL",
    "F1_SCORE",
    # "mAP50",
    # "mAP50-95",
    "PREDICT_TIME_PER_IMAGE",
)

predictions_results = []
for path in RESULTS_DIR.glob(str(Path("**", "results.txt"))):
    model_eval_details = None
    model_eval_results = None
    with path.open(mode="r") as results_file:
        for line in results_file.readlines():
            if "all" not in line.split():
                continue
            model_eval_results = line.split()[1:]
            val_images = model_eval_results[0]
            mask_precision = round(float(model_eval_results[6]), 3)
            mask_recall = round(float(model_eval_results[7]), 3)
            mask_f1_score = 0.0
            if mask_precision or mask_recall:
                mask_f1_score = round(2 * (mask_precision * mask_recall) / (mask_precision + mask_recall), 3)
            model_eval_results = [val_images, str(mask_precision), str(mask_recall), str(mask_f1_score)]

            use_case_name = path.parent.name
            model_eval_details = use_case_name.split("-")
            for i, model_eval_detail in enumerate(model_eval_details):
                model_eval_details[i] = (
                    " ".join(model_eval_detail.replace("__", " & ").split("_")[:-1]) or model_eval_detail
                )
            break

    seconds_per_predict_image = None
    with (path.parent / "process_time_predict.txt").open(mode="r") as proc_time_file:
        for line in proc_time_file.readlines():
            if not line.startswith("Duration:"):
                continue
            _, _, time = line.strip().partition(": ")
            time_obj = datetime.strptime(time, "%H:%M:%S.%f")
            total_seconds = (
                time_obj.hour * 3600 + time_obj.minute * 60 + time_obj.second + time_obj.microsecond / 1000000
            )
            seconds_per_predict_image = str(round(total_seconds / int(model_eval_results[0]), 3))

    # Write results
    predictions_results.append(model_eval_details + model_eval_results + [seconds_per_predict_image])

with FULL_REPORT_CSV_PATH.open(mode="w") as report_file:
    # Write headers
    report_file.write(",".join(REPORT_HEADERS) + "\n")
    # Sort by F1 score metric in descending order
    predictions_results_sorted = sorted(predictions_results, key=lambda x: x[-2], reverse=True)
    for result in predictions_results_sorted:
        report_file.write(",".join(result) + "\n")
