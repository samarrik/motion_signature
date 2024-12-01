# Ensure PowerShell is running with administrator privileges.

Write-Host "Please install NVIDIA drivers, CUDA, and WSL2 using the instructions from the official NVIDIA website."
$installation_done = Read-Host "Have you completed the installation? (y/n)"
if ($installation_done -notin @("y", "Y", "yes", "Yes", "YES")) {
    Write-Host "Please complete the installation before proceeding."
    exit 1
}

# Install Python 3.9 using Chocolatey
Write-Host "Installing Python 3.9..."
if (!(Get-Command "choco" -ErrorAction SilentlyContinue)) {
    Write-Host "Chocolatey is not installed. Please install Chocolatey first from https://chocolatey.org/install."
    exit 1
}
choco install python --version=3.9 -y

# Add Python to PATH if not already added
$pythonPath = "C:\Python39\Scripts"
if (-not ($Env:Path -like "*$pythonPath*")) {
    [System.Environment]::SetEnvironmentVariable("Path", $Env:Path + ";$pythonPath", [System.EnvironmentVariableTarget]::Machine)
    Write-Host "Added Python to PATH. You might need to restart PowerShell."
}

# Verify Python installation
Write-Host "Verifying Python installation..."
python --version
if ($LASTEXITCODE -ne 0) {
    Write-Host "Python installation failed. Please check and try again."
    exit 1
}

# Set up Python virtual environment
Write-Host "Setting up Python virtual environment..."
python -m venv venv_ms
.\venv_ms\Scripts\activate
pip install --upgrade pip

# Install CMake using Chocolatey
Write-Host "Installing CMake..."
choco install cmake -y

# Check for C++ Compiler
Write-Host "Checking for C++ compiler..."
if (!(Get-Command "g++" -ErrorAction SilentlyContinue)) {
    Write-Host "No compatible C++ compiler found. Please install a compiler that supports C++11 (e.g., MSYS2 or Visual Studio)."
    exit 1
}

# Install Python requirements
Write-Host "Installing Python requirements..."
if (-not (Test-Path "requirements.txt")) {
    Write-Host "requirements.txt file not found. Please ensure it's in the current directory."
    exit 1
}
pip install -qr requirements.txt

# Download required models
Write-Host "Downloading required models..."
# Placeholder for model download command
# Example:
# Invoke-WebRequest -Uri "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/1/pose_landmarker_heavy.task" -OutFile ".\data\models\mediapipe\pose_landmarker.task"

Write-Host "Everything is ready. Activate the virtual environment and run main.py."
Write-Host "Run the following command to activate the virtual environment:"
Write-Host ".\venv_ms\Scripts\activate"
