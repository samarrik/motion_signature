#!/bin/bash

echo "Please install NVIDIA drivers, CUDA, and WSL2 using the instructions from the official NVIDIA website."
read -p "Have you completed the installation? (y/n): " installation_done
if [[ "$installation_done" != "y" ]]; then
    echo "Please complete the installation before proceeding."
    exit 1
fi

echo "Installing Python 3.9"
sudo add-apt-repository ppa:deadsnakes/ppa
sudo apt update
sudo apt upgrad
sudo apt install -y python3.9 python3.9-venv python3.9-dev

echo "Setting up Python virtual environment with Python 3.9"
python3.9 -m venv venv_zel
source venv_zel/bin/activate
pip install --upgrade pip

echo "Installing Python requirements"
pip install -qr requirements.txt

echo "Downloading required models"
mkdir -p ./models/mediapipe
sudo wget -O ./models/mediapipe/pose_landmarker.task -q https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/1/pose_landmarker_heavy.task   

echo "Downloading required datasets"
sudo apt install unzip
wget -O datasets/dataset_clips.zip --no-check-certificate -r 'https://drive.usercontent.google.com/download?id=1Nd9i3Wsr4FIySUrQJN-6m3eKLhjG8ute&export=download&authuser=0&confirm=t&uuid=46b14fe2-59c9-4e50-9c9f-b9720984f5bd&at=AENtkXYKL__f2SiUg7n8cYMWbI4u:1730645134616'
unzip datasets/dataset_clips.zip
rm datasets/dataset_clips.zip

# wget -O datasets/datasets/extracted_correlations.csv --no-check-certificate -r 'https://drive.google.com/uc?export=download&id=[ID]'

echo "Installing OpenFace"
sudo add-apt-repository ppa:kisak/kisak-mesa
sudo apt update
sudo apt upgrade
sudo apt install -y \
  build-essential \
  cmake \
  wget \
  libopenblas-dev \
  libopencv-dev \
  libdlib-dev \
  libboost-all-dev \
  libsqlite3-dev
wget -O install_openface.py https://raw.githubusercontent.com/GuillaumeRochette/OpenFace/master/install.py
mkdir utils
sed -i 's|DEFAULT_INSTALL_PATH = Path(os.environ\["HOME"\]) / "OpenFace"|DEFAULT_INSTALL_PATH = Path(os.getcwd()) / "utils/OpenFace"|g' install_openface.py
python install_openface.py
sudo rm install_openface.py

echo "Everything is ready, run main.py"
