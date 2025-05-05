import os
import datasets
from datasets import load_dataset
from huggingface_hub import login
import logging

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
            dataset = load_dataset(dataset_name)
            logger.info(f"Dataset {dataset_name} loaded successfully")
            
            # Save the dataset locally
            dataset.save_to_disk(save_path)
            logger.info(f"Dataset saved to {save_path}")
        except Exception as e:
            logger.error(f"Error downloading dataset {dataset_name}: {str(e)}")

if __name__ == "__main__":
    logger.info("Starting dataset download process")
    create_directories()
    download_datasets()
    logger.info("Dataset download process completed")
