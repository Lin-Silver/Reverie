import os
import subprocess
import logging
import shutil
import platform
import sys

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def create_directories():
    """Create necessary directories if they don't exist."""
    directories = ["./resources", "./base_Models", "./reverie", "./reverie/checkpoints", "./reverie/model"]
    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory)
            logger.info(f"Created directory: {directory}")
        else:
            logger.info(f"Directory already exists: {directory}")

def check_git_lfs():
    """Check if Git LFS is installed and install it if needed."""
    try:
        # Check if git-lfs is installed
        result = subprocess.run(["git", "lfs", "version"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if result.returncode == 0:
            logger.info("Git LFS is already installed")
            return True
        else:
            logger.warning("Git LFS not found, attempting to install...")
    except Exception:
        logger.warning("Git LFS not found, attempting to install...")

    # Try to install Git LFS based on the platform
    system = platform.system().lower()
    try:
        if system == "linux":
            # For Ubuntu/Debian
            logger.info("Attempting to install Git LFS on Linux...")
            subprocess.run(["sudo", "apt-get", "update"], check=True)
            subprocess.run(["sudo", "apt-get", "install", "-y", "git-lfs"], check=True)
        elif system == "darwin":
            # For macOS
            logger.info("Attempting to install Git LFS on macOS...")
            subprocess.run(["brew", "install", "git-lfs"], check=True)
        elif system == "windows":
            logger.info("On Windows, please install Git LFS manually if not already installed.")
            logger.info("Visit https://git-lfs.github.com/ for installation instructions.")
            # We'll still try to run git-lfs install in case it's installed but not configured
            subprocess.run(["git", "lfs", "install"], check=True)
            return True
        else:
            logger.error(f"Unsupported platform: {system}")
            return False

        # Initialize Git LFS
        subprocess.run(["git", "lfs", "install"], check=True)
        logger.info("Git LFS installed and initialized successfully")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Error installing Git LFS: {str(e)}")
        logger.info("Please install Git LFS manually: https://git-lfs.github.com/")
        return False
    except Exception as e:
        logger.error(f"Unexpected error installing Git LFS: {str(e)}")
        return False

def download_qwen3_model():
    """Download Qwen3 4B model using Git."""
    model_repo = "https://huggingface.co/Qwen/Qwen3-4B"
    model_dir = "./base_Models/Qwen3-4B"

    # Check if Git LFS is available
    if not check_git_lfs():
        logger.warning("Proceeding without Git LFS. This may result in incomplete model files.")

    if os.path.exists(model_dir):
        logger.info(f"Model directory already exists: {model_dir}")
        logger.info("Updating the repository...")
        try:
            subprocess.run(["git", "-C", model_dir, "pull"], check=True)
            logger.info("Model repository updated successfully")
        except subprocess.CalledProcessError as e:
            logger.error(f"Error updating model repository: {str(e)}")
    else:
        logger.info(f"Cloning model repository: {model_repo}")
        try:
            # Make sure Git LFS is initialized
            subprocess.run(["git", "lfs", "install"], check=True)

            # Clone the repository
            subprocess.run(["git", "clone", model_repo, model_dir], check=True)
            logger.info(f"Model cloned successfully to {model_dir}")

            # Verify LFS files were properly downloaded
            logger.info("Verifying LFS files...")
            os.chdir(model_dir)
            subprocess.run(["git", "lfs", "pull"], check=True)
            os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # Return to original directory
            logger.info("LFS files verified")
        except subprocess.CalledProcessError as e:
            logger.error(f"Error cloning model repository: {str(e)}")
        except Exception as e:
            logger.error(f"Unexpected error during model download: {str(e)}")

if __name__ == "__main__":
    logger.info("Starting model download process")
    logger.info(f"Running on {platform.system()} ({platform.platform()})")
    create_directories()
    download_qwen3_model()
    logger.info("Model download process completed")
