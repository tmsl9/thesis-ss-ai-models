import logging
import math
from pathlib import Path
import sys
from typing import Any

import cv2
import matplotlib.pyplot as plt

project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

from config import (
    ARTICLE_1_PRE_PROCESSING_TECHNIQUES,
    ARTICLE_2_PRE_PROCESSING_TECHNIQUES,
    ARTICLE_5_PRE_PROCESSING_TECHNIQUES,
    ARTICLE_6_PRE_PROCESSING_TECHNIQUES,
    ARTICLE_8_PRE_PROCESSING_TECHNIQUES,
    ARTICLE_12_PRE_PROCESSING_TECHNIQUES,
    ARTICLE_15_PRE_PROCESSING_TECHNIQUES,
    ARTICLE_19_PRE_PROCESSING_TECHNIQUES,
    ARTICLE_20_PRE_PROCESSING_TECHNIQUES,
    ARTICLE_21_PRE_PROCESSING_TECHNIQUES,
    ARTICLE_22_PRE_PROCESSING_TECHNIQUES,
    ARTICLE_23_PRE_PROCESSING_TECHNIQUES,
    dataset_45_imgs_no_augmentations,
)
from dataset_utils import load_dataset


logger = logging.getLogger(__name__)
logging.basicConfig()
logger.setLevel(logging.INFO)

IMAGE_NAME = "Psoriasis-Chronic-Plaque-167_jpg.rf.48b3abd3136229399fdfcbdce5ab46ae.jpg"
MAX_COLS = 3
MAX_ROWS = 4
TITLE_FONTSIZE = 10
FIG_WIDTH = 15
FIG_HEIGHT = 15
OUTPUT_PATH = project_root / "data" / "articles_used_pre_processing_techniques.jpg"


def get_var_name(value: Any) -> str:
    found_names = []
    global_vars = globals().copy()
    for var_name, var_value in global_vars.items():
        if isinstance(var_value, list) and value in var_value:
            found_names.append(var_name)
    return found_names


if __name__ == "__main__":
    _, dataset = load_dataset(dataset_45_imgs_no_augmentations)

    # Use Path.glob to find the image
    default_dataset_dirname = Path(dataset.location).name
    base_path = Path(dataset.location).parent
    image_paths = list(base_path.glob(f"{default_dataset_dirname}*/train/images/{IMAGE_NAME}"))

    image_paths_found = {}
    for image_path in image_paths:
        pre_processing_names = image_path.parent.parent.parent.name.replace(f"{default_dataset_dirname}_", "").replace(
            default_dataset_dirname, ""
        )
        if not pre_processing_names:
            image_paths_found[image_path] = ("0", "original")
        else:
            var_names = get_var_name(pre_processing_names)
            art_nrs = [var_name for var_name in var_names if "ARTICLE_" in var_name]
            if not art_nrs:
                continue
            # Fetch articles numbers
            art_nrs = [art_nr.replace("ARTICLE_", "").replace("_PRE_PROCESSING_TECHNIQUES", "") for art_nr in art_nrs]
            art_nrs.sort()
            techniques = pre_processing_names.replace("__", " & ").replace("_", " ")
            image_paths_found[image_path] = ("/".join(art_nrs), techniques)

    sorted_image_paths = dict(
        sorted(
            image_paths_found.items(),
            key=lambda item: int(item[1][0].split("/")[0]) if "/" in item[1][0] else int(item[1][0]),
        )
    )

    image_paths = list(sorted_image_paths.keys())
    total_images = len(image_paths)

    # Split images into chunks of 9 (MAX_COLS * MAX_ROWS)
    num_images_split = math.ceil(total_images / (MAX_COLS * MAX_ROWS))

    for grid_index in range(num_images_split):
        start_idx = grid_index * (MAX_COLS * MAX_ROWS)
        end_idx = min(start_idx + (MAX_COLS * MAX_ROWS), total_images)
        current_images = image_paths[start_idx:end_idx]

        num_current_images = len(current_images)

        # Adjust rows based on the number of images in the current grid
        cols = MAX_COLS
        rows = math.ceil(num_current_images / cols)

        # Adjust the figure size (height) dynamically based on the number of rows
        fig_height = rows * 5  # Adjust scaling factor 5 if you want more/less height per row
        fig, axes = plt.subplots(rows, cols, figsize=(15, fig_height))

        # Flatten the axes array for easier iteration, even if there's only one row/column
        if rows > 1:
            axes = axes.flatten()
        else:
            axes = [axes] if cols == 1 else axes.flatten()

        # Loop through each image and plot it in the grid
        title_inverse = True
        for img_nr, image_path in enumerate(current_images):
            img = cv2.imread(str(image_path))
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB for matplotlib

            axes[img_nr].imshow(img_rgb)
            axes[img_nr].axis("off")  # Turn off axis labels

            # Add the title below the image
            art_nrs = image_paths_found[image_path][0]
            title = art_nrs if art_nrs else "original"
            techniques = image_paths_found[image_path][1]

            title_inverse = not title_inverse
            if title_inverse:
                title = f"Art. {title}\n{techniques}"
            else:
                title = f"{techniques}\nArt. {title}"
            # Add the title below the image
            axes[img_nr].text(0.5, -0.1, title, fontsize=TITLE_FONTSIZE, ha="center", transform=axes[img_nr].transAxes)

        # Hide any unused subplots
        for j in range(num_current_images, len(axes)):
            axes[j].axis("off")

        # Adjust layout and save the final image grid
        plt.subplots_adjust(wspace=0.02, hspace=0.2)  # Adjust spacing as needed
        OUTPUT_PATH.parent.mkdir(exist_ok=True, parents=True)
        plt.savefig(OUTPUT_PATH.parent / (OUTPUT_PATH.stem + f"_{grid_index + 1}" + OUTPUT_PATH.suffix))
        plt.close(fig)  # Close the figure to save memory
