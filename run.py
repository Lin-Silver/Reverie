#!/usr/bin/env python3
"""
Unified script to run the Reverie model fine-tuning process on any platform.
This script detects the operating system and runs the appropriate commands.
"""

import os
import sys
import platform
import subprocess
import argparse

def print_header(message):
    """Print a header message."""
    print("\n" + "=" * 80)
    print(f" {message}")
    print("=" * 80)

def create_directories():
    """Create necessary directories."""
    directories = ["./resources", "./base_Models", "./reverie", "./reverie/checkpoints", "./reverie/model"]
    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory)
            print(f"Created directory: {directory}")
        else:
            print(f"Directory already exists: {directory}")

def setup_virtual_environment():
    """Set up a virtual environment based on the platform."""
    system = platform.system().lower()
    
    # Check if virtual environment already exists
    venv_dir = "./venv"
    if os.path.exists(venv_dir):
        print("Virtual environment already exists.")
    else:
        print("Creating virtual environment...")
        
        # Create virtual environment
        if system in ["linux", "darwin"]:  # Linux or macOS
            result = subprocess.run([sys.executable, "-m", "venv", venv_dir], capture_output=True, text=True)
        else:  # Windows
            result = subprocess.run([sys.executable, "-m", "venv", venv_dir], capture_output=True, text=True)
        
        if result.returncode != 0:
            print(f"Error creating virtual environment: {result.stderr}")
            sys.exit(1)
        
        print("Virtual environment created successfully.")
    
    # Activate virtual environment and install dependencies
    print("Installing dependencies...")
    
    # Determine the pip path based on the platform
    if system in ["linux", "darwin"]:  # Linux or macOS
        pip_path = os.path.join(venv_dir, "bin", "pip")
        python_path = os.path.join(venv_dir, "bin", "python")
    else:  # Windows
        pip_path = os.path.join(venv_dir, "Scripts", "pip")
        python_path = os.path.join(venv_dir, "Scripts", "python")
    
    # Upgrade pip
    print("Upgrading pip...")
    result = subprocess.run([pip_path, "install", "--upgrade", "pip"], capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Error upgrading pip: {result.stderr}")
        sys.exit(1)
    
    # Install dependencies
    print("Installing required packages...")
    packages = [
        "torch", 
        "transformers>=4.38.0", 
        "datasets", 
        "accelerate", 
        "peft", 
        "bitsandbytes", 
        "huggingface_hub", 
        "trl", 
        "sentencepiece", 
        "protobuf", 
        "tensorboard"
    ]
    
    for package in packages:
        print(f"Installing {package}...")
        result = subprocess.run([pip_path, "install", package], capture_output=True, text=True)
        if result.returncode != 0:
            print(f"Warning: Error installing {package}: {result.stderr}")
    
    print("Dependencies installed successfully.")
    
    return python_path

def download_datasets(python_path):
    """Download the datasets."""
    print_header("Downloading datasets")
    result = subprocess.run([python_path, "download_datasets.py"], capture_output=True, text=True)
    print(result.stdout)
    if result.stderr:
        print(f"Errors: {result.stderr}")
    if result.returncode != 0:
        print("Error downloading datasets")
        sys.exit(1)

def download_model(python_path):
    """Download the model."""
    print_header("Downloading model")
    result = subprocess.run([python_path, "download_model.py"], capture_output=True, text=True)
    print(result.stdout)
    if result.stderr:
        print(f"Errors: {result.stderr}")
    if result.returncode != 0:
        print("Error downloading model")
        sys.exit(1)

def train_model(python_path):
    """Train the model."""
    print_header("Training model")
    result = subprocess.run([python_path, "train_model.py"], capture_output=True, text=True)
    print(result.stdout)
    if result.stderr:
        print(f"Errors: {result.stderr}")
    if result.returncode != 0:
        print("Error training model")
        sys.exit(1)

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Run the Reverie model fine-tuning process.")
    parser.add_argument("--download-only", action="store_true", help="Only download the datasets and model without training")
    args = parser.parse_args()
    
    print_header("Starting Reverie model fine-tuning process")
    print(f"Running on {platform.system()} ({platform.platform()})")
    
    # Create directories
    print_header("Creating directories")
    create_directories()
    
    # Set up virtual environment
    print_header("Setting up virtual environment")
    python_path = setup_virtual_environment()
    
    # Download datasets
    download_datasets(python_path)
    
    # Download model
    download_model(python_path)
    
    # Train model if not download-only
    if not args.download_only:
        train_model(python_path)
        print_header("All steps completed successfully!")
        print("The fine-tuned model is available in the ./reverie/model directory")
    else:
        print_header("Download completed successfully!")
        print("You can now run 'python run.py' to start the training process")

if __name__ == "__main__":
    main()
