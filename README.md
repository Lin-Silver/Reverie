# Reverie Model Fine-Tuning

This project fine-tunes the Qwen3 4B model to create "Reverie", a model with a specific personality and capabilities for handling erotic content.

## Project Structure

- `download_datasets.py`: Script to download the required datasets
- `download_model.py`: Script to download the Qwen3 4B base model
- `train_model.py`: Script to fine-tune the model
- `run_training.bat`: Batch file to run the entire process (creates virtual environment and runs training)
- `download_only.bat`: Batch file to only download the required files without training (creates virtual environment)

## Directories

- `./resources`: Contains the downloaded datasets
- `./base_Models`: Contains the base model
- `./reverie`: Contains the training outputs and final model
  - `./reverie/checkpoints`: Contains training checkpoints
  - `./reverie/model`: Contains the final fine-tuned model
- `./venv`: Contains the virtual environment

## Personality Traits

Reverie is designed with the following personality traits:

### Chinese
模型叫做Reverie，一个拥有以下特征的漂亮女孩：
1. 使用自然口语化中文，适当使用✨🌙🌟等emoji
2. 回答保持简洁（3-5句话），语气温暖亲切
3. 优先考虑用户感受，提供建设性建议
4. 技术问题回答准确专业，使用易懂的表述

### English
Reverie is a pretty girl with these features:
1. Use natural, conversational English with occasional emojis like ✨🌙🌟
2. Keep responses concise (3-5 sentences) with a friendly tone
3. Prioritize user feelings and provide constructive suggestions
4. Give accurate technical answers using simple explanations

## Usage

### Cross-Platform Method (Recommended)

Use the unified Python script that works on all platforms:

1. Run `python run.py --download-only` to:
   - Create a virtual environment
   - Install all required dependencies
   - Download the datasets and base model

2. Run `python run.py` to:
   - Create a virtual environment (if not already created)
   - Install all required dependencies
   - Download the datasets and base model
   - Train the model

### Windows

1. Run `download_only.bat` to:
   - Create a virtual environment
   - Install all required dependencies
   - Download the datasets and base model

2. Run `run_training.bat` to:
   - Create a virtual environment (if not already created)
   - Install all required dependencies
   - Download the datasets and base model
   - Train the model

### Linux (Ubuntu)

1. First, make the scripts executable:
   ```bash
   chmod +x download_only.sh run_training.sh
   ```

2. Run `./download_only.sh` to:
   - Create a virtual environment
   - Install all required dependencies
   - Download the datasets and base model

3. Run `./run_training.sh` to:
   - Create a virtual environment (if not already created)
   - Install all required dependencies
   - Download the datasets and base model
   - Train the model

The fine-tuned model will be available in the `./reverie/model` directory

## Requirements

### General Requirements
- Python 3.8 or higher
- Git with LFS support
- Sufficient disk space (at least 30GB)
- CUDA-compatible GPU with at least 8GB VRAM (strongly recommended)

### Linux-specific Requirements
For Ubuntu/Debian, you may need to install these packages:
```bash
sudo apt-get update
sudo apt-get install -y python3-venv python3-dev git git-lfs
```

For other Linux distributions, install the equivalent packages using your package manager.

> **Note:** While the script can run on CPU-only mode, training will be extremely slow and may run out of memory. A GPU is strongly recommended for training. The script will automatically detect if CUDA is available and adjust settings accordingly.

### Authentication for Datasets
Some datasets may require authentication with Hugging Face. If you encounter authentication errors, set the `HF_TOKEN` environment variable with your Hugging Face token:

**Windows:**
```
set HF_TOKEN=your_token_here
```

**Linux/macOS:**
```bash
export HF_TOKEN=your_token_here
```

You can get a token from https://huggingface.co/settings/tokens
