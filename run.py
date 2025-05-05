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

def check_and_install_venv():
    """Check if python3-venv is installed on Debian/Ubuntu systems and install if needed."""
    system = platform.system().lower()

    if system != "linux":
        return True  # Only relevant for Linux

    # Check if we're on a Debian/Ubuntu system
    if not os.path.exists("/etc/os-release"):
        return True  # Not a standard Linux distribution, assume it's fine

    # Read OS information
    with open("/etc/os-release", "r") as f:
        os_info = {}
        for line in f:
            if "=" in line:
                key, value = line.strip().split("=", 1)
                os_info[key] = value.strip('"')

    # Check if it's Debian/Ubuntu
    is_debian_based = False
    if "ID" in os_info and os_info["ID"] in ["ubuntu", "debian"]:
        is_debian_based = True
    elif "ID_LIKE" in os_info and any(id in os_info["ID_LIKE"] for id in ["ubuntu", "debian"]):
        is_debian_based = True

    if not is_debian_based:
        return True  # Not Debian-based, assume it's fine

    # Check if python3-venv is installed
    try:
        result = subprocess.run(["dpkg", "-l", "python3-venv"],
                               stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if "ii  python3-venv" in result.stdout:
            print("python3-venv is already installed.")
            return True
    except Exception:
        pass  # Continue with installation attempt

    print("The python3-venv package is not installed.")
    print("This is required to create virtual environments.")

    try:
        # Try to determine the Python version
        python_version = platform.python_version().split(".")
        if len(python_version) >= 2:
            venv_package = f"python3.{python_version[1]}-venv"
            print(f"Detected Python version: {platform.python_version()}")
            print(f"Recommended package: {venv_package}")
        else:
            venv_package = "python3-venv"
    except Exception:
        venv_package = "python3-venv"

    print(f"Would you like to install {venv_package} and python3-dev now? (y/n)")
    answer = input().strip().lower()

    if answer in ["y", "yes"]:
        print(f"Installing {venv_package} and python3-dev...")
        try:
            subprocess.run(["sudo", "apt-get", "update"], check=True)
            subprocess.run(["sudo", "apt-get", "install", "-y", venv_package, "python3-dev"], check=True)
            print("Installation successful.")
            return True
        except subprocess.CalledProcessError as e:
            print(f"Failed to install required packages: {e}")
            print(f"Please install them manually:")
            print(f"sudo apt-get update && sudo apt-get install -y {venv_package} python3-dev")
            return False
    else:
        print(f"Please install the required packages manually:")
        print(f"sudo apt-get update && sudo apt-get install -y {venv_package} python3-dev")
        return False

def setup_virtual_environment():
    """Set up a virtual environment based on the platform."""
    system = platform.system().lower()

    # Check if virtual environment already exists
    venv_dir = "./venv"
    if os.path.exists(venv_dir):
        print("Virtual environment already exists.")
    else:
        print("Creating virtual environment...")

        # For Linux, check if python3-venv is installed
        if system == "linux" and not check_and_install_venv():
            print("Cannot proceed without the required packages.")
            sys.exit(1)

        # Create virtual environment
        try:
            result = subprocess.run([sys.executable, "-m", "venv", venv_dir],
                                   capture_output=True, text=True)

            if result.returncode != 0:
                print(f"Error creating virtual environment: {result.stderr}")
                if system == "linux":
                    print("If you're on Ubuntu/Debian, make sure you have the correct python3-venv package installed.")
                    python_version = platform.python_version().split(".")
                    if len(python_version) >= 2:
                        print(f"For Python {platform.python_version()}, you might need: python3.{python_version[1]}-venv")
                sys.exit(1)

            print("Virtual environment created successfully.")
        except Exception as e:
            print(f"Error creating virtual environment: {str(e)}")
            sys.exit(1)

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
