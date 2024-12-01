#!/bin/bash

echo "Please install NVIDIA drivers, CUDA, and WSL2 using the instructions from the official NVIDIA website."
read -p "Have you completed the installation? (y/n): " installation_done
if [[ "$installation_done" != "y" && "$installation_done" != "Y" && "$installation_done" != "yes" && "$installation_done" != "Yes" && "$installation_done" != "yea" && "$installation_done" != "yeah" && "$installation_done" != "YES" ]]; then
    echo "Please complete the installation before proceeding."
    exit 1
fi

# Ensure wget is available
if ! command -v wget &>/dev/null; then
    echo "wget is not installed. Please install wget before proceeding."
    exit 1
fi

echo "Installing Python 3.9 locally"
PYTHON_DIR="$HOME/python39"
if [[ ! -d "$PYTHON_DIR" ]]; then
    mkdir -p "$PYTHON_DIR"
    wget https://www.python.org/ftp/python/3.9.18/Python-3.9.18.tgz -O Python-3.9.18.tgz
    tar -xzf Python-3.9.18.tgz
    cd Python-3.9.18
    ./configure --prefix="$PYTHON_DIR"
    make -j$(nproc)
    make install
    cd ..
    rm -rf Python-3.9.18 Python-3.9.18.tgz
else
    echo "Python 3.9 is already installed locally."
fi

echo "Setting up Python virtual environment with Python 3.9"
python3.9 -m venv venv_ms
source venv_ms/bin/activate

echo "Installing CMake locally"
CMAKE_DIR="$HOME/local/cmake"
if [[ ! -d "$CMAKE_DIR" ]]; then
    mkdir -p "$CMAKE_DIR"
    cd "$CMAKE_DIR"
    wget https://github.com/Kitware/CMake/releases/download/v3.27.6/cmake-3.27.6-linux-x86_64.tar.gz -O cmake.tar.gz
    tar -xzf cmake.tar.gz --strip-components=1
    export PATH="$CMAKE_DIR/bin:$PATH"
    cd -
else
    echo "CMake is already installed locally."
    export PATH="$CMAKE_DIR/bin:$PATH"
fi

echo "Upgrading pip, setuptools, and wheel"
pip install --upgrade pip setuptools wheel || { echo "Failed to upgrade pip and setuptools"; exit 1; }


echo "Installing Python requirements"
pip install -r requirements.txt || { echo "Failed to install Python requirements"; exit 1; }

echo "Downloading required models"
download_models
# mkdir -p "$MODEL_DIR"
# wget -O "$MODEL_DIR/pose_landmarker.task" -q https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/1/pose_landmarker_heavy.task || { echo "Failed to download model"; exit 1; }

echo "Everything is ready. Activate the virtual environment and run main.py."
echo "Run the following command to activate the virtual environment:"
echo "source venv_ms/bin/activate"
