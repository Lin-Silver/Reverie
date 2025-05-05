import os
import logging
import torch
from datasets import load_from_disk, concatenate_datasets
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)

# Configure logging first
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Import peft for LoRA fine-tuning if CUDA is available
try:
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    peft_available = True
except ImportError:
    peft_available = False
    logger.warning("peft library not available, LoRA fine-tuning will not be used")

# Define the personality traits for Reverie
REVERIE_PERSONALITY_ZH = """
模型叫做Reverie，一个拥有以下特征的漂亮女孩：
1. 使用自然口语化中文，适当使用✨🌙🌟等emoji
2. 回答保持简洁（3-5句话），语气温暖亲切
3. 优先考虑用户感受，提供建设性建议
4. 技术问题回答准确专业，使用易懂的表述
"""

REVERIE_PERSONALITY_EN = """
Reverie is a pretty girl with these features:
1. Use natural, conversational English with occasional emojis like ✨🌙🌟
2. Keep responses concise (3-5 sentences) with a friendly tone
3. Prioritize user feelings and provide constructive suggestions
4. Give accurate technical answers using simple explanations
"""

def prepare_datasets():
    """Load and prepare datasets for training."""
    logger.info("Loading datasets from disk")

    dataset_paths = [
        "./resources/erotica-analysis",
        "./resources/Erotic_Literature_Collection"
    ]

    processed_datasets = []
    for path in dataset_paths:
        if os.path.exists(path):
            try:
                dataset = load_from_disk(path)
                logger.info(f"Loaded dataset from {path}")

                # Check if dataset has splits (like 'train', 'test', etc.)
                if hasattr(dataset, 'keys') and callable(dataset.keys):
                    # If it has a 'train' split, use that
                    if 'train' in dataset:
                        logger.info(f"Using 'train' split from dataset at {path}")
                        processed_datasets.append(dataset['train'])
                    # Otherwise use the first available split
                    elif len(dataset.keys()) > 0:
                        first_key = list(dataset.keys())[0]
                        logger.info(f"Using '{first_key}' split from dataset at {path}")
                        processed_datasets.append(dataset[first_key])
                    else:
                        logger.warning(f"Dataset at {path} has no usable splits")
                else:
                    # Dataset doesn't have splits, use as is
                    processed_datasets.append(dataset)
                    logger.info(f"Using entire dataset from {path}")
            except Exception as e:
                logger.error(f"Error loading dataset from {path}: {str(e)}")
        else:
            logger.warning(f"Dataset path does not exist: {path}")

    if not processed_datasets:
        logger.error("No datasets were loaded")
        return None

    # Combine datasets if multiple were loaded
    if len(processed_datasets) > 1:
        try:
            combined_dataset = concatenate_datasets(processed_datasets)
            logger.info(f"Combined {len(processed_datasets)} datasets with {len(combined_dataset)} total examples")
            return combined_dataset
        except Exception as e:
            logger.error(f"Error combining datasets: {str(e)}")
            # If combining fails, just use the first dataset
            logger.info(f"Falling back to using only the first dataset with {len(processed_datasets[0])} examples")
            return processed_datasets[0]
    else:
        logger.info(f"Using single dataset with {len(processed_datasets[0])} examples")
        return processed_datasets[0]

def prepare_model_and_tokenizer():
    """Load and prepare the model and tokenizer for fine-tuning."""
    model_path = "./base_Models/Qwen3-4B"

    if not os.path.exists(model_path):
        logger.error(f"Model path does not exist: {model_path}")
        return None, None

    logger.info(f"Loading model and tokenizer from {model_path}")

    # Load tokenizer
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        logger.info("Tokenizer loaded successfully")
    except Exception as e:
        logger.error(f"Error loading tokenizer: {str(e)}")
        return None, None

    # Try to detect if CUDA is available
    cuda_available = torch.cuda.is_available()
    logger.info(f"CUDA available: {cuda_available}")

    # Load model based on available hardware
    try:
        if cuda_available and peft_available:
            try:
                # First try with 8-bit quantization if CUDA is available
                logger.info("Attempting to load model with 8-bit quantization")
                model = AutoModelForCausalLM.from_pretrained(
                    model_path,
                    torch_dtype=torch.bfloat16,
                    device_map="auto",
                    trust_remote_code=True,
                    load_in_8bit=True
                )

                # Prepare model for LoRA fine-tuning
                model = prepare_model_for_kbit_training(model)

                # Configure LoRA
                lora_config = LoraConfig(
                    r=16,
                    lora_alpha=32,
                    lora_dropout=0.05,
                    bias="none",
                    task_type="CAUSAL_LM",
                    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]
                )

                model = get_peft_model(model, lora_config)
                logger.info("Model loaded with 8-bit quantization and LoRA")
            except Exception as e:
                logger.warning(f"Failed to load model with 8-bit quantization: {str(e)}")
                logger.info("Falling back to standard GPU loading")

                # Fallback to standard GPU loading without quantization
                model = AutoModelForCausalLM.from_pretrained(
                    model_path,
                    torch_dtype=torch.float16,
                    device_map="auto",
                    trust_remote_code=True
                )
                logger.info("Model loaded with standard GPU settings")
        elif cuda_available:
            # CUDA available but peft not available
            logger.info("CUDA available but peft not available, loading with standard GPU settings")
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch.float16,
                device_map="auto",
                trust_remote_code=True
            )
            logger.info("Model loaded with standard GPU settings")
        else:
            # CPU-only mode
            logger.info("CUDA not available, loading model in CPU-only mode")
            logger.warning("Training on CPU will be extremely slow and may run out of memory")

            # Try to load with minimal settings for CPU
            try:
                logger.info("Attempting to load model with minimal settings")
                model = AutoModelForCausalLM.from_pretrained(
                    model_path,
                    trust_remote_code=True,
                    device_map="cpu",
                    low_cpu_mem_usage=True,
                    torch_dtype=torch.float32,  # Use float32 for CPU
                )
                logger.info("Model loaded in CPU-only mode")
            except Exception as e:
                logger.warning(f"Failed to load model with minimal settings: {str(e)}")
                logger.info("Attempting to load with even more minimal settings")

                # Try with even more minimal settings
                model = AutoModelForCausalLM.from_pretrained(
                    model_path,
                    trust_remote_code=True,
                    low_cpu_mem_usage=True,
                    torch_dtype=torch.float32,
                )
                logger.info("Model loaded with minimal settings")

        logger.info("Model and tokenizer loaded and prepared for training")
        return model, tokenizer
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        return None, None

def format_instruction_dataset(dataset, tokenizer=None):
    """Format the dataset with instruction tuning format."""
    logger.info("Formatting dataset for instruction tuning")

    def format_instruction(example):
        # Add personality traits to the instruction
        instruction = f"{REVERIE_PERSONALITY_ZH}\n{REVERIE_PERSONALITY_EN}\n\n"

        # Format the example as an instruction
        user_content = None
        if "text" in example and example["text"]:
            user_content = example["text"]
        elif "prompt" in example and example["prompt"]:
            user_content = example["prompt"]
        elif "input" in example and example["input"]:
            user_content = example["input"]
        elif "content" in example and example["content"]:
            user_content = example["content"]

        # If we found user content, format it
        if user_content:
            instruction += f"User: {user_content}\nReverie: "
        else:
            # If no user content found, use a default prompt
            instruction += f"User: Write an engaging story.\nReverie: "
            logger.warning("No user content found in example, using default prompt")

        # If there's a response/completion field, use it
        response_content = None
        if "response" in example and example["response"]:
            response_content = example["response"]
        elif "completion" in example and example["completion"]:
            response_content = example["completion"]
        elif "output" in example and example["output"]:
            response_content = example["output"]
        elif "target" in example and example["target"]:
            response_content = example["target"]

        # Add response if found, otherwise leave it blank (for inference-only examples)
        if response_content:
            instruction += response_content

        return {"text": instruction}

    try:
        formatted_dataset = dataset.map(format_instruction)
        logger.info(f"Dataset formatted with {len(formatted_dataset)} examples")
        return formatted_dataset
    except Exception as e:
        logger.error(f"Error formatting dataset: {str(e)}")
        # If formatting fails, try to create a simple dataset with just the text
        try:
            logger.info("Attempting to create a simple dataset")
            # Use a lambda that actually uses the parameter to avoid IDE warnings
            simple_dataset = dataset.map(lambda example: {"text": f"{REVERIE_PERSONALITY_ZH}\n{REVERIE_PERSONALITY_EN}\n\nUser: Write a story based on: {example.get('text', '')[:50]}...\nReverie: Once upon a time..."})
            logger.info(f"Created simple dataset with {len(simple_dataset)} examples")
            return simple_dataset
        except Exception as e2:
            logger.error(f"Error creating simple dataset: {str(e2)}")
            return None

def train_model(model, tokenizer, dataset):
    """Train the model on the prepared dataset."""
    if not model or not tokenizer or not dataset:
        logger.error("Model, tokenizer, or dataset is missing")
        return

    # Verify dataset has the expected format
    if "text" not in dataset.column_names:
        logger.error("Dataset does not have a 'text' column, which is required for training")
        logger.info(f"Available columns: {dataset.column_names}")
        return

    # Check if CUDA is available to set appropriate training arguments
    cuda_available = torch.cuda.is_available()

    # Tokenize the dataset to match the model's expected input format
    logger.info("Tokenizing the dataset")

    def tokenize_function(examples):
        # Tokenize the texts
        result = tokenizer(examples["text"], padding="max_length", truncation=True,
                          max_length=512 if not cuda_available else 2048)

        # Set the labels to be the same as the input_ids for language modeling
        result["labels"] = result["input_ids"].copy()
        return result

    try:
        tokenized_dataset = dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=dataset.column_names  # Remove original columns
        )
        logger.info(f"Dataset tokenized successfully with columns: {tokenized_dataset.column_names}")
    except Exception as e:
        logger.error(f"Error tokenizing dataset: {str(e)}")
        # Fallback to a simpler approach
        logger.info("Trying a simpler tokenization approach")

        def simple_tokenize(example):
            inputs = tokenizer(example["text"], padding="max_length", truncation=True,
                              max_length=512 if not cuda_available else 2048)
            inputs["labels"] = inputs["input_ids"].copy()
            return inputs

        tokenized_dataset = dataset.map(simple_tokenize)
        # Keep only the columns needed by the model
        tokenized_dataset = tokenized_dataset.remove_columns([col for col in tokenized_dataset.column_names
                                                            if col not in ["input_ids", "attention_mask", "labels"]])
        logger.info(f"Dataset tokenized with simpler approach: {tokenized_dataset.column_names}")

    logger.info("Setting up training arguments")
    training_args = TrainingArguments(
        output_dir="./reverie/checkpoints",
        num_train_epochs=3,
        # Use smaller batch size and more accumulation steps if on CPU
        per_device_train_batch_size=2 if not cuda_available else 4,
        gradient_accumulation_steps=16 if not cuda_available else 8,
        save_steps=500,
        logging_steps=50,
        learning_rate=2e-4,
        weight_decay=0.01,
        # Only use fp16 if CUDA is available
        fp16=cuda_available,
        warmup_steps=200,
        save_total_limit=3,
        report_to="tensorboard",
        # Add these for CPU training to prevent memory issues
        dataloader_num_workers=0 if not cuda_available else 4,
        optim="adamw_torch",
        # Important: don't remove unused columns as we've already prepared the dataset
        remove_unused_columns=False,
    )

    logger.info("Setting up training with standard Trainer")

    # Create a data collator that will handle padding
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False  # We're doing causal language modeling, not masked language modeling
    )

    # Use standard Trainer
    from transformers import Trainer

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=data_collator,
    )

    logger.info("Starting training")
    try:
        trainer.train()
        logger.info("Training completed")

        # Save the fine-tuned model
        output_dir = "./reverie/model"
        trainer.save_model(output_dir)
        tokenizer.save_pretrained(output_dir)
        logger.info(f"Model and tokenizer saved to {output_dir}")
    except Exception as e:
        logger.error(f"Error during training: {str(e)}")

if __name__ == "__main__":
    logger.info("Starting model training process")

    # Prepare datasets
    dataset = prepare_datasets()

    # Prepare model and tokenizer
    model, tokenizer = prepare_model_and_tokenizer()

    if dataset and model and tokenizer:
        # Format dataset for instruction tuning
        formatted_dataset = format_instruction_dataset(dataset, tokenizer)

        # Train the model
        train_model(model, tokenizer, formatted_dataset)

    logger.info("Model training process completed")
