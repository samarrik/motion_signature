#!/bin/bash

echo "Please install NVIDIA drivers, CUDA, and WSL2 using the instructions from the official NVIDIA website."
read -p "Have you completed the installation? (y/n): " installation_done
if [[ "$installation_done" != "y" && "$installation_done" != "Y" && "$installation_done" != "yes" && "$installation_done" != "Yes" && "$installation_done" != "yea" && "$installation_done" != "yeah" && "$installation_done" != "YES" ]]; then
    echo "Please complete the installation before proceeding."
    exit 1
fi

echo "Installing Python 3.9"
sudo add-apt-repository ppa:deadsnakes/ppa
sudo apt update && sudo apt upgrade -y
sudo apt install -y python3.9 python3.9-venv python3.9-dev

echo "Setting up Python virtual environment with Python 3.9"
python3.9 -m venv venv_ms
source venv_ms/bin/activate
pip install --upgrade pip

echo "Insatlling cmake"
sudo apt install cmake

echo "Make sure the compiler supports C++11"
sudo add-apt-repository -y ppa:ubuntu-toolchain-r/test
sudo apt install g++-11

echo "Installing Python requirements"
pip install -qr requirements.txt

echo "Downloading required models"
download_models
# mkdir -p .data/models/mediapipe
# sudo wget -O .data/models/mediapipe/pose_landmarker.task -q https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/1/pose_landmarker_heavy.task   

echo "Everything is ready, run main.py"
