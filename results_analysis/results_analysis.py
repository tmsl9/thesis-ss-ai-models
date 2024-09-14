"""
final_analysis = {
    "BOX(PRECISION)": [
        {
            "value": 123,
            "MODEL": "...",
            "...": "...",
        }
    ]
}
"""

import csv
import json
import logging
from pathlib import Path
from typing import DefaultDict

logger = logging.getLogger(__name__)
logging.basicConfig()
logger.setLevel(logging.INFO)

ROOT_DIR = Path().cwd()
FULL_REPORT_CSV_PATH = ROOT_DIR / "full_report.csv"
REPORT_ANALYSIS_JSON_PATH = ROOT_DIR / "report_analysis.json"
REPORT_ANALYSIS_CSV_PATH = ROOT_DIR / "report_analysis.csv"

REPORT_RESULTS_HEADERS = (
    "PRECISION",
    "RECALL",
    "F1_SCORE",
    "PREDICT_TIME_PER_IMAGE",
)
REPORT_TRAIN_MODEL_DETAILS_HEADERS = (
    "MODEL",
    "EPOCHS",
    "LEARNING_RATE",
    "TRAINING_IMGSIZE",
    "TOTAL_IMAGES",
    "PREPROCESSING",
    "AUGMENTATIONS",
    "PREDICT_IMAGES",
)

# Show ranking of results per metric
MAX_RANK = 5

final_analysis = {header: [] for header in REPORT_RESULTS_HEADERS}

with FULL_REPORT_CSV_PATH.open(mode="r") as report_file:
    report_data_list = list(csv.DictReader(report_file))
    for csv_line_nr, report_data in enumerate(report_data_list):
        for results_header in REPORT_RESULTS_HEADERS:
            final_analysis[results_header].append(
                {
                    "value": report_data[results_header],
                    "csv_line_nr": csv_line_nr,
                    **{key_detail: report_data[key_detail] for key_detail in REPORT_TRAIN_MODEL_DETAILS_HEADERS},
                }
            )

    # Sort by metric value into a ranking per metric
    model_use_case_overall_dict = DefaultDict()
    for metric_key, metric_ranking in final_analysis.items():
        metric_ranking_sorted = sorted(metric_ranking, key=lambda x: x["value"], reverse=True)
        if metric_key == "PREDICT_TIME_PER_IMAGE":
            # Reverse order, lesser time the better
            final_analysis[metric_key] = metric_ranking_sorted[: -(MAX_RANK + 1) : -1]
        else:
            final_analysis[metric_key] = metric_ranking_sorted[:MAX_RANK]
        for position, use_case_details in enumerate(final_analysis[metric_key]):
            # Increment counter for the model use-case
            csv_line_nr = use_case_details["csv_line_nr"]
            if csv_line_nr not in model_use_case_overall_dict:
                model_use_case_overall_dict[csv_line_nr] = {"count": 0}
            model_use_case_overall_dict[csv_line_nr]["count"] += 1
            model_use_case_overall_dict[csv_line_nr][metric_key] = position
            # Remove key-value, it was just used to check the overall metrics ranking
            del use_case_details["csv_line_nr"]
            final_analysis[metric_key][position] = use_case_details

    # Store report analysis in json file
    with REPORT_ANALYSIS_JSON_PATH.open("w", encoding="utf-8") as report_analysis_file:
        report_analysis_file.write(json.dumps(final_analysis, indent=4))

    with REPORT_ANALYSIS_CSV_PATH.open(mode="w") as report_post_analysis_file:
        # Write headers
        report_post_analysis_file.write(",".join(["COUNT", *report_data_list[0].keys()]) + "\n")
        model_use_case_overall_ranking = dict(
            sorted(model_use_case_overall_dict.items(), key=lambda item: item[1]["count"], reverse=True)
        )
        for csv_line_nr, run_details in model_use_case_overall_ranking.items():
            results_report_data = report_data_list[csv_line_nr]
            for metric in run_details:
                if metric != "count":
                    results_report_data[metric] = f"{results_report_data[metric]} ({run_details[metric] + 1})"
            report_data = {"COUNT": str(run_details["count"]), **report_data_list[csv_line_nr]}
            report_post_analysis_file.write(",".join(report_data.values()) + "\n")
