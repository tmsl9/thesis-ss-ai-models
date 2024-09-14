import logging
import math
from pathlib import Path
import sys

import cv2
import matplotlib.pyplot as plt

project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

from config import dataset_45_imgs_no_augmentations
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
OUTPUT_PATH = project_root / "data" / "pre_processing_techniques.jpg"

if __name__ == "__main__":
    _, dataset = load_dataset(dataset_45_imgs_no_augmentations)

    # Use Path.glob to find the image
    default_dataset_dirname = Path(dataset.location).name
    base_path = Path(dataset.location).parent
    image_paths = list(base_path.glob(f"{default_dataset_dirname}*/train/images/{IMAGE_NAME}"))

    image_paths_found = {}
    for image_path in image_paths:
        pre_processing_name = image_path.parent.parent.parent.name.replace(f"{default_dataset_dirname}_", "").replace(
            default_dataset_dirname, ""
        )
        if not pre_processing_name:
            pre_processing_name = "original"
        if "__" in pre_processing_name:
            continue
        pre_processing_name = pre_processing_name.replace("_", " ")

        image_paths_found[image_path] = pre_processing_name

    image_paths = list(image_paths_found.keys())
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
        for img_nr, image_path in enumerate(current_images):
            img = cv2.imread(str(image_path))
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB for matplotlib

            axes[img_nr].imshow(img_rgb)
            axes[img_nr].axis("off")  # Turn off axis labels

            # Add the title below the image
            axes[img_nr].text(
                0.5,
                -0.1,
                f"{img_nr + start_idx + 1}. {image_paths_found[image_path]}",
                fontsize=TITLE_FONTSIZE,
                ha="center",
                transform=axes[img_nr].transAxes,
            )

        # Hide any unused subplots
        for j in range(num_current_images, len(axes)):
            axes[j].axis("off")

        # Adjust layout and save the final image grid
        plt.subplots_adjust(wspace=0.02, hspace=0.2)  # Adjust spacing as needed
        OUTPUT_PATH.parent.mkdir(exist_ok=True, parents=True)
        plt.savefig(OUTPUT_PATH.parent / (OUTPUT_PATH.stem + f"_{grid_index + 1}" + OUTPUT_PATH.suffix))
        plt.close(fig)  # Close the figure to save memory
