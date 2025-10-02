@echo off
REM Setup script for Uzbek TTS training environment on Windows

echo Setting up Uzbek TTS training environment...

REM Check Python version
python --version
if %errorlevel% neq 0 (
    echo Error: Python is not installed or not in PATH
    pause
    exit /b 1
)

echo Python found ✓

REM Create virtual environment
echo Creating virtual environment...
py -3.11 -m venv .venv311
.\.venv311\Scripts\activate

REM Upgrade pip
echo Upgrading pip...
python -m pip install --upgrade pip

pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128

REM Install Python dependencies
echo Installing Python dependencies...
pip install -r requirements.txt

REM Install TTS
echo Installing TTS...
pip install TTS

REM Verify installations
echo Verifying installations...
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import TTS; print(f'TTS: {TTS.__version__}')"
python -c "import mlflow; print(f'MLflow: {mlflow.__version__}')"
python -c "import wandb; print(f'WandB: {wandb.__version__}')"

REM Create directories
echo Creating project directories...
if not exist "data\raw" mkdir data\raw
if not exist "data\processed" mkdir data\processed
if not exist "data\preprocessed" mkdir data\preprocessed
if not exist "output" mkdir output
if not exist "logs" mkdir logs
if not exist "models" mkdir models


$ python src/data_preprocessing.py --skip_phonemization --metadata_file train_metadata.csv

@REM echo Environment setup completed!
@REM echo To start training, run:
@REM echo python train_uzbek_tts.py --preprocess --evaluate --generate_samples

pip install pystoi pesq openai-whisper
