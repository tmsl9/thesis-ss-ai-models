# thesis-ss-ai-models
 1. Psoriasis detection using semantic segmentation AI models.
 2. Comparison between the different models.

# Setup for WSL2 Ubuntu 22.04

## Install cuDNN 9.4.0 for Linux Ubuntu 22.04 x86_64
https://forums.developer.nvidia.com/t/windows-11-wsl2-cuda-windows-11-home-22000-708-nvidia-studio-driver-512-96/217721/3
```bash
wget https://developer.download.nvidia.com/compute/cudnn/9.4.0/local_installers/cudnn-local-repo-ubuntu2204-9.4.0_1.0-1_amd64.deb
sudo dpkg -i cudnn-local-repo-ubuntu2204-9.4.0_1.0-1_amd64.deb
sudo cp /var/cudnn-local-repo-ubuntu2204-9.4.0/cudnn-*-keyring.gpg /usr/share/keyrings/
sudo apt-get update
sudo apt-get -y install cudnn
sudo apt-get -y install libcudnn8 libcudnn8-dev
echo "\
# CUDA
export PATH=/usr/local/cuda/bin:\$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:\$LD_LIBRARY_PATH
export CUDA_HOME=/usr/local/cuda
" >> ~/.bashrc
```

## Install CUDA Toolkit 12.1 for Linux WSL-Ubuntu 2.0 x86_64
```bash
wget https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/cuda-wsl-ubuntu.pin
sudo mv cuda-wsl-ubuntu.pin /etc/apt/preferences.d/cuda-repository-pin-600
wget https://developer.download.nvidia.com/compute/cuda/12.1.1/local_installers/cuda-repo-wsl-ubuntu-12-1-local_12.1.1-1_amd64.deb
sudo dpkg -i cuda-repo-wsl-ubuntu-12-1-local_12.1.1-1_amd64.deb
sudo cp /var/cuda-repo-wsl-ubuntu-12-1-local/cuda-*-keyring.gpg /usr/share/keyrings/
sudo apt-get update
sudo apt-get -y install cuda
```

## Install PyTorch with a Supported CUDA Version
```bash
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

## Verify Installation
```bash
python3 -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
```

## Install Ultralytics YOLOv8:
```bash
pip3 install ultralytics==8.0.196
```

## Install OpenCV (for image processing):
```bash
pip3 install opencv-python
```

## Install Roboflow (if you use Roboflow datasets):
```bash
pip3 install roboflow
```

## Install sklearn
```bash
pip install sklearn
```

## Install sklearn
```bash
pip install scikit-learn
```

## Install tensorflow
```bash
pip install tensorflow
```

## Install pycocotools
```bash
pip install pycocotools
```

## Install Mask R-CNN
```bash
pip install git+https://github.com/matterport/Mask_RCNN.git
```

## Install tensorflow-hub
```bash
pip install tensorflow-hub
```

## Install keras-cv
```bash
pip install keras-cv
```

### Verify YOLOv8 Installation
```bash
yolo --version

yolo task=detect mode=predict model=yolov8n.pt source='https://ultralytics.com/images/bus.jpg'
```

# Run

## Download the datasets
```bash
python3 download_all_datasets.py
```

## Run the manual data augmentations in the datasets
```bash
python3 data_augmentations/contrast_enhancement.py
python3 data_augmentations/scale.py
python3 data_augmentations/sharp.py
python3 data_augmentations/translate.py
python3 data_augmentations/zoom_in.py
```

## Run the pre-processing techniques in the datasets
```bash
python3 pre_processing/exec_pre_processing_techniques.py
```

## Configure the models to execute
Update config.py with the desired configurations.

## Run the models
```bash
python3 run_models.py
```

## Process the results
1. Extract and process results into a CSV file sorted by F1 score metric
    ```bash
    python3 results_analysis/process_results.py
    ```

2. Process CSV file report, and check top 5 use-cases per metric, save in JSON file, and create other CSV file with the results sorted by the ranking appearances of the results
    ```bash
    python3 results_analysis/results_analysis.py
    ```
