import os
import subprocess
import logging
import shutil

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def create_directories():
    """Create necessary directories if they don't exist."""
    directories = ["./resources", "./base_Models"]
    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory)
            logger.info(f"Created directory: {directory}")
        else:
            logger.info(f"Directory already exists: {directory}")

def download_qwen3_model():
    """Download Qwen3 4B model using Git."""
    model_repo = "https://huggingface.co/Qwen/Qwen3-4B"
    model_dir = "./base_Models/Qwen3-4B"

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
            subprocess.run(["git", "lfs", "install"], check=True)
            subprocess.run(["git", "clone", model_repo, model_dir], check=True)
            logger.info(f"Model cloned successfully to {model_dir}")
        except subprocess.CalledProcessError as e:
            logger.error(f"Error cloning model repository: {str(e)}")

if __name__ == "__main__":
    logger.info("Starting model download process")
    create_directories()
    download_qwen3_model()
    logger.info("Model download process completed")
