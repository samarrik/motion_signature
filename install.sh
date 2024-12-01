#!/bin/bash

echo "Please install NVIDIA drivers, CUDA, and WSL2 using the instructions from the official NVIDIA website."
read -p "Have you completed the installation? (y/n): " installation_done
if [[ "$installation_done" != "y" && "$installation_done" != "Y" && "$installation_done" != "yes" && "$installation_done" != "Yes" && "$installation_done" != "yea" && "$installation_done" != "yeah" && "$installation_done" != "YES" ]]; then
    echo "Please complete the installation before proceeding."
    exit 1
fi

echo "Installing Python 3.9 using pyenv"
if ! command -v pyenv &>/dev/null; then
    echo "pyenv is not installed. Installing pyenv locally."
    curl https://pyenv.run | bash
    export PATH="$HOME/.pyenv/bin:$PATH"
    eval "$(pyenv init --path)"
    eval "$(pyenv virtualenv-init -)"
fi

export PATH="$HOME/.pyenv/bin:$PATH"
eval "$(pyenv init --path)"
eval "$(pyenv virtualenv-init -)"

if ! pyenv versions | grep -q "3.9"; then
    echo "Python 3.9 is not installed. Installing it using pyenv."
    pyenv install 3.9.18
fi

pyenv local 3.9.18

echo "Setting up Python virtual environment with Python 3.9"
python3.9 -m venv venv_ms
source venv_ms/bin/activate
pip install --upgrade pip

echo "Installing Python requirements"
pip install -r requirements.txt

echo "Downloading required models"
download_models
# mkdir -p .data/models/mediapipe
# wget -O .data/models/mediapipe/pose_landmarker.task -q https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/1/pose_landmarker_heavy.task

echo "Everything is ready. Activate the virtual environment and run main.py."
echo "Run the following command to activate the virtual environment:"
echo "source venv_ms/bin/activate"
