@echo off
echo Starting download process for Reverie model fine-tuning...

echo Step 1: Creating directories...
if not exist ".\resources" mkdir .\resources
if not exist ".\base_Models" mkdir .\base_Models
if not exist ".\reverie" mkdir .\reverie
if not exist ".\reverie\checkpoints" mkdir .\reverie\checkpoints
if not exist ".\reverie\model" mkdir .\reverie\model
echo Directories created successfully.

echo Step 2: Setting up virtual environment...
if exist ".\venv" (
    echo Virtual environment already exists.
) else (
    echo Creating virtual environment...
    python -m venv venv
    if %ERRORLEVEL% NEQ 0 (
        echo Error creating virtual environment
        exit /b %ERRORLEVEL%
    )
    echo Virtual environment created successfully.
)

echo Step 3: Installing dependencies...
call .\venv\Scripts\activate.bat
echo Upgrading pip...
python -m pip install --upgrade pip
if %ERRORLEVEL% NEQ 0 (
    echo Error upgrading pip
    exit /b %ERRORLEVEL%
)

echo Installing required packages...
pip install torch transformers>=4.38.0 datasets accelerate peft bitsandbytes huggingface_hub trl sentencepiece protobuf tensorboard
if %ERRORLEVEL% NEQ 0 (
    echo Error installing dependencies
    exit /b %ERRORLEVEL%
)
echo Dependencies installed successfully.

echo Step 4: Downloading datasets...
python download_datasets.py
if %ERRORLEVEL% NEQ 0 (
    echo Error downloading datasets
    exit /b %ERRORLEVEL%
)

echo Step 5: Downloading base model...
python download_model.py
if %ERRORLEVEL% NEQ 0 (
    echo Error downloading base model
    exit /b %ERRORLEVEL%
)

echo All downloads completed successfully!
echo You can now run run_training.bat to start the training process

pause
