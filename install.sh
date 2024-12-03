#!/bin/bash

echo "Please install CUDA, and cuDNN using either modules or instructions from nvidia website."
read -p "Have you completed the installation? (y/n): " installation_done
if [[ "$installation_done" != "y" && "$installation_done" != "Y" && "$installation_done" != "yes" && "$installation_done" != "Yes" && "$installation_done" != "yea" && "$installation_done" != "yeah" && "$installation_done" != "YES" ]]; then
@@ -13,44 +17,28 @@ if ! command -v wget &>/dev/null; then
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
wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O Miniconda3.sh
bash Miniconda3.sh -b -p $HOME/miniconda
rm Miniconda3.sh

# Initialize Conda
export PATH="$HOME/miniconda/bin:$PATH"
source $HOME/miniconda/etc/profile.d/conda.sh

# Setting up Python virtual environment with all dependencies
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
