#!/bin/bash

echo "Please install NVIDIA drivers, CUDA, and WSL2 using the instructions from the official NVIDIA website."
read -p "Have you completed the installation? (y/n): " installation_done
if [[ "$installation_done" != "y" && "$installation_done" != "Y" ]]; then
    echo "Please complete the installation before proceeding."
    exit 1
fi

echo "Installing Python 3.9 locally using pyenv..."
if ! command -v pyenv &>/dev/null; then
    echo "Downloading and installing pyenv..."
    wget -O pyenv-installer https://pyenv.run
    bash pyenv-installer
    export PATH="$HOME/.pyenv/bin:$PATH"
    eval "$(pyenv init --path)"
    eval "$(pyenv virtualenv-init -)"
fi

pyenv install 3.9.10
pyenv virtualenv 3.9.10 venv_ms
pyenv activate venv_ms

echo "Upgrading pip and installing requirements"
pip install --upgrade pip
pip install -qr requirements.txt

echo "Downloading required models"
download_models
# mkdir -p .data/models/mediapipe
# wget -O .data/models/mediapipe/pose_landmarker.task \
#     https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/1/pose_landmarker_heavy.task

echo "Everything is ready, run main.py"
