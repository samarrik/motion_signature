#!/bin/bash

echo "Please install NVIDIA drivers, CUDA, and WSL2 using the instructions from the official NVIDIA website."
read -p "Have you completed the installation? (y/n): " installation_done
if [[ "$installation_done" != "y" && "$installation_done" != "Y" && "$installation_done" != "yes" && "$installation_done" != "Yes" ]]; then
    echo "Please complete the installation before proceeding."
    exit 1
fi

echo "Checking for Python 3.9 installation..."
if ! command -v python3.9 &>/dev/null; then
    echo "Python 3.9 is not available. Installing Python 3.9 locally."
    mkdir -p $HOME/python-install
    cd $HOME/python-install
    wget https://www.python.org/ftp/python/3.9.18/Python-3.9.18.tgz
    tar -xzf Python-3.9.18.tgz
    cd Python-3.9.18
    ./configure --prefix=$HOME/python-3.9 --enable-optimizations
    make -j$(nproc)
    make install
    export PATH="$HOME/python-3.9/bin:$PATH"
    cd -
else
    echo "Python 3.9 is already installed."
fi

echo "Setting up Python virtual environment with Python 3.9"
if ! command -v python3.9 &>/dev/null; then
    echo "Error: Python 3.9 installation failed or is not in PATH."
    exit 1
fi

python3.9 -m venv venv_ms
if [[ ! -d "venv_ms" ]]; then
    echo "Virtual environment creation failed."
    exit 1
fi
source venv_ms/bin/activate

echo "Installing Python requirements"
if [[ ! -f "requirements.txt" ]]; then
    echo "requirements.txt file not found. Please provide a valid file."
    exit 1
fi
pip install -r requirements.txt

echo "Downloading required models"
download_models
# mkdir -p .data/models/mediapipe
# wget -O .data/models/mediapipe/pose_landmarker.task -q https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/1/pose_landmarker_heavy.task

echo "Everything is ready. Activate the virtual environment and run main.py."
echo "Run the following command to activate the virtual environment:"
echo "source venv_ms/bin/activate"
