#!/bin/bash

# Setting up the structure
mkdir data/dataset
mkdir data/extracted

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

echo "Download conda and build it"
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh
rm Miniconda3-latest-Linux-x86_64.sh
conda init
source ~/.bashrc
conda update conda
conda --version

echo "Setting up Python virtual environment with all dependencies"
conda env create -f environment.yml
conda init
conda activate ms-env

# Create symbolic links to NVIDIA shared libraries
pushd $(dirname $(python -c 'print(__import__("tensorflow").__file__)'))
ln -svf ../nvidia/*/lib/*.so* .
popd

echo "Downloading required models"
download_models

echo "Everything is ready. Activate the virtual environment and run main.py."
echo "Run the following command to activate the virtual environment:"
echo "conda activate ms-env"
