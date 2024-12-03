#!/bin/bash

echo "Please install CUDA and cuDNN using NVIDIA's instructions."
read -p "Have you completed the installation? (yes/y): " installation_done
if [[ "$installation_done" != "yes" && "$installation_done" != "y" ]]; then
    echo "Complete the installation before proceeding."
    exit 1
fi

# Setting up the structure
mkdir -p data/dataset
mkdir -p data/extracted

# Ensure wget is available
if ! command -v wget &>/dev/null; then
    echo "wget is not installed. Please install wget before proceeding."
    exit 1
fi

echo "Downloading Miniconda and setting it up..."
mkdir -p ~/miniconda3
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh
bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
rm ~/miniconda3/miniconda.sh
source ~/miniconda3/bin/activate
conda init --all

echo "Creating Conda environment..."
if [ -f "environment.yml" ]; then
    conda env create -f environment.yml
else
    echo "Error: environment.yml not found."
    exit 1
fi

conda activate ms-env

# Create symbolic links to NVIDIA shared libraries (if applicable)
if python -c "import tensorflow" &>/dev/null; then
    pushd $(dirname $(python -c 'print(__import__("tensorflow").__file__)'))
    ln -svf ../nvidia/*/lib/*.so* . || echo "NVIDIA libraries not linked. Verify CUDA installation."
    popd
else
    echo "Error: TensorFlow not installed. Verify your environment.yml or install manually."
    exit 1
fi

echo "Downloading required models..."
download_models

echo "Everything is ready. Activate the virtual environment and run main.py."
echo "Run the following command to activate the virtual environment:"
echo "conda activate ms-env"