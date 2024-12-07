#!/bin/bash

# Header
echo -e "\n\e[1;34m========================================================"
echo -e "  Talking Motion Project: Installation and Setup Script"
echo -e "========================================================\e[0m\n"

# Step 1: Create Necessary Directories
echo -e "\e[1;33m[1/6] Initializing directories for data storage...\e[0m"
mkdir -p data/dataset
echo -e "\e[1;32m    [✔] Directories initialized successfully.\e[0m\n"

# Step 2: Check for wget Availability
echo -e "\e[1;33m[2/6] Verifying system dependencies...\e[0m"
if ! command -v wget &>/dev/null; then
    echo -e "\e[1;31m    [✖] wget is not installed. Please install wget before proceeding.\e[0m"
    exit 1
fi
echo -e "\e[1;32m    [✔] wget is available.\e[0m\n"

# Step 3: Download and Install Miniconda
echo -e "\e[1;33m[3/6] Downloading and configuring Miniconda...\e[0m"
mkdir -p ~/miniconda3
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh --quiet
bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
rm ~/miniconda3/miniconda.sh
source ~/miniconda3/bin/activate
conda init --all
echo -e "\e[1;32m    [✔] Miniconda installed and initialized.\e[0m\n"

# Step 4: Create Conda Environment
echo -e "\e[1;33m[4/6] Creating the Conda environment from configuration...\e[0m"
if [ -f "environment.yml" ]; then
    conda env create -f environment.yml
    echo -e "\e[1;32m    [✔] Conda environment created successfully.\e[0m\n"
else
    echo -e "\e[1;31m    [✖] Configuration file environment.yml not found. Ensure it is present in the working directory.\e[0m"
    exit 1
fi

# Step 5: Upgrade Dependencies
echo -e "\e[1;33m[5/6] Activating environment and upgrading Python dependencies...\e[0m"
conda activate ms-env
pip install --upgrade pip setuptools wheel
echo -e "\e[1;32m    [✔] Dependencies upgraded successfully.\e[0m\n"

# Step 6: Install Development Version of py-feat
echo -e "\e[1;33m[6/6] Installing the py-feat library in development mode...\e[0m"
cd ~/miniconda3/envs/ms-env/lib
git clone https://github.com/cosanlab/feat.git
cd feat
pip install -e .
echo -e "\e[1;32m    [✔] py-feat successfully installed.\e[0m\n"

# Completion Message
echo -e "\e[1;34m========================================================="
echo -e "  Installation Complete"
echo -e "========================================================="
echo -e "  Activate the Conda environment with the following command:"
echo -e "      conda activate ms-env"
echo -e "  Proceed with your analysis by running main.py."
echo -e "=========================================================\e[0m\n"
