import os
import datasets
from datasets import load_dataset
from huggingface_hub import login
import logging
import platform

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

def download_datasets():
    """Download datasets from Hugging Face and save them locally."""
    datasets_info = [
        {
            "name": "openerotica/erotica-analysis",
            "save_path": "./resources/erotica-analysis"
        },
        {
            "name": "ystemsrx/Erotic_Literature_Collection",
            "save_path": "./resources/Erotic_Literature_Collection"
        }
    ]

    for dataset_info in datasets_info:
        dataset_name = dataset_info["name"]
        save_path = dataset_info["save_path"]

        logger.info(f"Downloading dataset: {dataset_name}")
        try:
            # Try to download with authentication first
            try:
                # Check if HF_TOKEN environment variable is set
                hf_token = os.environ.get("HF_TOKEN")
                if hf_token:
                    logger.info("Using Hugging Face token from environment variable")
                    login(token=hf_token)

                dataset = load_dataset(dataset_name)
            except Exception as auth_error:
                logger.warning(f"Error downloading with authentication: {str(auth_error)}")
                logger.info("Trying to download without authentication...")
                dataset = load_dataset(dataset_name, use_auth_token=False)

            logger.info(f"Dataset {dataset_name} loaded successfully")

            # Save the dataset locally
            dataset.save_to_disk(save_path)
            logger.info(f"Dataset saved to {save_path}")
        except Exception as e:
            logger.error(f"Error downloading dataset {dataset_name}: {str(e)}")
            logger.error("If the dataset requires authentication, set the HF_TOKEN environment variable with your Hugging Face token")
            logger.error("You can get a token from https://huggingface.co/settings/tokens")

if __name__ == "__main__":
    logger.info("Starting dataset download process")
    logger.info(f"Running on {platform.system()} ({platform.platform()})")
    create_directories()
    download_datasets()
    logger.info("Dataset download process completed")
