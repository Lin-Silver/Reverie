#!/bin/bash
set -e  # Exit on error

echo "Starting Reverie model fine-tuning process..."

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
        echo "Error creating virtual environment"
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

echo "Step 6: Starting model training..."
python train_model.py
if [ $? -ne 0 ]; then
    echo "Error during model training"
    exit 1
fi

echo "All steps completed successfully!"
echo "The fine-tuned model is available in the ./reverie/model directory"
