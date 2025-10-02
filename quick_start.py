#!/usr/bin/env python3
"""
Quick start script for Uzbek TTS training.
This script helps you get started with the training pipeline.
"""

import os
import sys
import subprocess
from pathlib import Path
import argparse

def check_requirements():
    """Check if required packages are installed."""
    required_packages = [
        'torch', 'TTS', 'mlflow', 'wandb', 'librosa', 
        'soundfile', 'phonemizer', 'pandas', 'numpy'
    ]
    
    missing_packages = []
    for package in required_packages:
        try:
            __import__(package)
            print(f"✓ {package}")
        except ImportError:
            missing_packages.append(package)
            print(f"✗ {package} - MISSING")
    
    if missing_packages:
        print(f"\nMissing packages: {', '.join(missing_packages)}")
        print("Please install them with: pip install -r requirements.txt")
        return False
    
    return True

def check_data():
    """Check if data is available."""
    data_dir = Path("data/processed")
    metadata_file = data_dir / "metadata.csv"
    
    if metadata_file.exists():
        print(f"✓ Data found: {metadata_file}")
        return True
    else:
        print(f"✗ Data not found: {metadata_file}")
        print("Please ensure your Common Voice data is in data/processed/")
        return False

def run_preprocessing():
    """Run data preprocessing."""
    print("\n" + "="*50)
    print("RUNNING DATA PREPROCESSING")
    print("="*50)
    
    # Step 1: Convert raw Common Voice data to processed format
    print("Step 1: Converting raw Common Voice data...")
    try:
        result = subprocess.run([
            sys.executable, "data/preprocess.py"
        ], check=True, capture_output=True, text=True)
        
        print("Raw data conversion completed successfully!")
        print(result.stdout)
        
    except subprocess.CalledProcessError as e:
        print(f"Raw data conversion failed: {e}")
        print(f"Error output: {e.stderr}")
        return False
    
    # Step 2: Add phonemization for TTS training
    print("\nStep 2: Adding phonemization...")
    try:
        # Try with fallback phonemization first
        result = subprocess.run([
            sys.executable, "src/data_preprocessing.py",
            "--data_dir", "./data/processed",
            "--output_dir", "./data/preprocessed",
            "--use_fallback"
        ], check=True, capture_output=True, text=True)
        
        print("Phonemization completed successfully!")
        print(result.stdout)
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"Phonemization failed, trying raw text mode: {e}")
        try:
            # Fallback to raw text mode
            result = subprocess.run([
                sys.executable, "src/data_preprocessing.py",
                "--data_dir", "./data/processed",
                "--output_dir", "./data/preprocessed",
                "--skip_phonemization"
            ], check=True, capture_output=True, text=True)
            
            print("Raw text preprocessing completed successfully!")
            print(result.stdout)
            return True
            
        except subprocess.CalledProcessError as e2:
            print(f"Raw text preprocessing also failed: {e2}")
            print(f"Error output: {e2.stderr}")
            return False

def run_training():
    """Run model training."""
    print("\n" + "="*50)
    print("STARTING MODEL TRAINING")
    print("="*50)
    
    try:
        result = subprocess.run([
            sys.executable, "train_uzbek_tts.py",
            "--config", "config.yaml",
            "--data_dir", "./data/preprocessed",
            "--evaluate",
            "--generate_samples"
        ], check=True)
        
        print("Training completed successfully!")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"Training failed: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Quick start for Uzbek TTS training")
    parser.add_argument("--skip_preprocessing", action="store_true", help="Skip data preprocessing")
    parser.add_argument("--skip_training", action="store_true", help="Skip model training")
    parser.add_argument("--check_only", action="store_true", help="Only check requirements and data")
    
    args = parser.parse_args()
    
    print("🇺🇿 Uzbek TTS - Quick Start")
    print("="*50)
    
    # Check requirements
    print("Checking requirements...")
    if not check_requirements():
        print("\n❌ Requirements check failed!")
        return
    
    # Check data
    print("\nChecking data...")
    if not check_data():
        print("\n❌ Data check failed!")
        return
    
    print("\n✅ All checks passed!")
    
    if args.check_only:
        print("\nCheck completed. Ready to start training!")
        return
    
    # Run preprocessing
    if not args.skip_preprocessing:
        if not run_preprocessing():
            print("\n❌ Preprocessing failed!")
            return
    
    # Run training
    if not args.skip_training:
        if not run_training():
            print("\n❌ Training failed!")
            return
    
    print("\n🎉 Uzbek TTS training pipeline completed successfully!")
    print("\nNext steps:")
    print("1. Check output/ directory for trained models")
    print("2. Run evaluation: python scripts/evaluate_model.py --model_path output/best_model.pth")
    print("3. Generate samples: Check output/samples/ directory")
    print("4. View experiment logs in MLflow: mlflow ui")

if __name__ == "__main__":
    main()

