#!/bin/bash
set -e  # Exit on error

echo "Starting download process for Reverie model fine-tuning..."

# Check for required packages
echo "Checking for required packages..."
PYTHON_VERSION=$(python3 --version 2>&1 | cut -d " " -f 2 | cut -d "." -f 1,2)
PYTHON_MAJOR=$(echo $PYTHON_VERSION | cut -d "." -f 1)
PYTHON_MINOR=$(echo $PYTHON_VERSION | cut -d "." -f 2)
VENV_PACKAGE="python3-venv"

# Check if we're on a Debian/Ubuntu system
if [ -f /etc/os-release ]; then
    . /etc/os-release
    if [[ "$ID" == "ubuntu" || "$ID" == "debian" || "$ID_LIKE" == *"ubuntu"* || "$ID_LIKE" == *"debian"* ]]; then
        echo "Detected Debian/Ubuntu-based system: $PRETTY_NAME"

        # Check if python3-venv is installed
        if ! dpkg -l | grep -q $VENV_PACKAGE; then
            echo "The $VENV_PACKAGE package is not installed."
            echo "This is required to create virtual environments."
            echo "Would you like to install it now? (y/n)"
            read -r answer
            if [[ "$answer" == "y" || "$answer" == "Y" ]]; then
                echo "Installing $VENV_PACKAGE..."
                sudo apt-get update
                sudo apt-get install -y $VENV_PACKAGE python3-dev
                if [ $? -ne 0 ]; then
                    echo "Failed to install $VENV_PACKAGE. Please install it manually:"
                    echo "sudo apt-get update && sudo apt-get install -y $VENV_PACKAGE python3-dev"
                    exit 1
                fi
            else
                echo "Please install the required package and try again:"
                echo "sudo apt-get update && sudo apt-get install -y $VENV_PACKAGE python3-dev"
                exit 1
            fi
        fi
    else
        echo "Not a Debian/Ubuntu system. Make sure you have the necessary packages installed for creating Python virtual environments."
    fi
else
    echo "Unable to determine OS. Make sure you have the necessary packages installed for creating Python virtual environments."
fi

echo "Step 1: Creating directories..."
mkdir -p ./resources
mkdir -p ./base_Models
mkdir -p ./reverie/checkpoints
mkdir -p ./reverie/model
echo "Directories created successfully."

echo "Step 2: Setting up virtual environment..."
if [ -d "./venv" ]; then
    echo "Virtual environment already exists."
else
    echo "Creating virtual environment..."
    python3 -m venv venv
    if [ $? -ne 0 ]; then
        echo "Error creating virtual environment."
        echo "If you're on Ubuntu/Debian, try installing the required package:"
        echo "sudo apt-get update && sudo apt-get install -y python3-venv python3-dev"
        exit 1
    fi
    echo "Virtual environment created successfully."
fi

echo "Step 3: Installing dependencies..."
source ./venv/bin/activate
echo "Upgrading pip..."
python -m pip install --upgrade pip
if [ $? -ne 0 ]; then
    echo "Error upgrading pip"
    exit 1
fi

echo "Installing required packages..."
pip install torch transformers>=4.38.0 datasets accelerate peft bitsandbytes huggingface_hub trl sentencepiece protobuf tensorboard
if [ $? -ne 0 ]; then
    echo "Error installing dependencies"
    exit 1
fi
echo "Dependencies installed successfully."

echo "Step 4: Downloading datasets..."
python download_datasets.py
if [ $? -ne 0 ]; then
    echo "Error downloading datasets"
    exit 1
fi

echo "Step 5: Downloading base model..."
python download_model.py
if [ $? -ne 0 ]; then
    echo "Error downloading base model"
    exit 1
fi

echo "All downloads completed successfully!"
echo "You can now run run_training.sh to start the training process"
