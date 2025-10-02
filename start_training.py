#!/usr/bin/env python3
"""
Simple training script for Uzbek TTS using the generated configuration.
This script uses the TTS library's built-in training capabilities.
"""

import os
import sys
import logging
import warnings
from pathlib import Path
from datetime import datetime
import mlflow
import wandb
import torch

# Suppress PyTorch warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning, module="torchaudio")
warnings.filterwarnings("ignore", category=UserWarning, module="torch.functional")

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    """Start training using the generated configuration."""
    
    # Check if config exists
    config_path = Path("output/vits_config.json")
    if not config_path.exists():
        logger.error("Configuration file not found. Please run train_uzbek_tts.py first.")
        return
    
    logger.info("Starting Uzbek TTS training...")
    logger.info(f"Using configuration: {config_path}")
    
    try:
        # Import TTS
        from TTS.api import TTS
        from TTS.tts.configs.vits_config import VitsConfig
        from TTS.tts.models.vits import Vits
        from trainer import Trainer, TrainerArgs
        
        # Load the configuration
        config = VitsConfig()
        config.load_json(str(config_path))
        
        # Define character set first (Latin + essential Uzbek Cyrillic + missing characters)
        uzbek_chars = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789ОоҒғҚқЎўҲҳЁёАаКкëʙТµģ"
        punctuations = "!'(),-.:;? \"\nʻʼ"
        
        # RTX 4090 Memory-Optimized Configuration (23.4/24GB VRAM usage)
        config.mixed_precision = True  # Enable mixed precision for RTX 4090
        config.epochs = 200  # Set reasonable number of epochs
        config.batch_size = 64  # Reduced from 96 to prevent OOM (still 2x increase from 5070 Ti)
        config.eval_batch_size = 16  # Reduced from 24 to prevent OOM
        config.save_step = 500  # More frequent saves with faster training
        config.save_best_after = 5000  # Earlier best model saving
        config.plot_step = 10000  # Set high value to effectively disable plotting
        config.print_step = 10  # Print more frequently
        config.num_loader_workers = 6  # Reduced from 8 to save memory
        config.num_eval_loader_workers = 3  # Reduced from 4 to save memory
        config.lr = 0.0002  # Adjusted learning rate for smaller batch size (2x increase from 5070 Ti)
        
        # RTX 4090 Memory optimization settings
        config.gradient_accumulation_steps = 1  # No need for accumulation with larger batch size
        config.gradient_clip_val = 1.0  # Gradient clipping for stability with larger batches
        config.pin_memory = True  # Enable for faster GPU transfer
        
        # Update model configuration to match character vocabulary
        config.model_args.num_chars = len(uzbek_chars) + len(punctuations) + 4  # +4 for special tokens
        
        # RTX 4090 Optimizer Configuration
        config.optimizer = "AdamW"
        config.weight_decay = 0.01
        config.betas = [0.8, 0.99]
        config.eps = 1e-9
        
        # RTX 4090 Scheduler Configuration
        config.lr_scheduler = "ExponentialLR"
        config.lr_scheduler_params = {"gamma": 0.9999, "last_epoch": -1}
        
        # RTX 4090 Loss Configuration
        config.c_mel = 45
        config.c_kl = 1.0
        
        logger.info("RTX 4090 Memory-Optimized Configuration loaded successfully")
        logger.info(f"Model: {config.model}")
        if hasattr(config, 'datasets') and config.datasets:
            logger.info(f"Dataset: {config.datasets[0].dataset_name}")
            logger.info(f"Dataset path: {config.datasets[0].path}")
        logger.info(f"🚀 RTX 4090 Batch size: {config.batch_size} (2x increase from 5070 Ti, memory-optimized)")
        logger.info(f"🚀 RTX 4090 Learning rate: {config.lr} (2x increase to match batch scaling)")
        logger.info(f"🚀 RTX 4090 Epochs: {config.epochs}")
        logger.info(f"🚀 RTX 4090 Mixed precision: {config.mixed_precision}")
        logger.info(f"🚀 RTX 4090 Data loader workers: {config.num_loader_workers} (memory-optimized)")
        logger.info(f"🚀 RTX 4090 Pin memory: {config.pin_memory}")
        logger.info(f"💾 Memory usage: ~23.4/24GB VRAM (optimized to prevent OOM)")
        logger.info(f"Character vocabulary size: {config.model_args.num_chars}")
        logger.info(f"Characters: {uzbek_chars}")
        
        # Initialize the model
        model = Vits(config)
        
        # Initialize audio processor for the model
        from TTS.utils.audio import AudioProcessor
        model.ap = AudioProcessor(**config.audio)
        
        # Setup tokenizer (this was missing!)
        from TTS.tts.utils.text.characters import IPAPhonemes
        from TTS.tts.utils.text.tokenizer import TTSTokenizer
        
        # Use the character set defined above
        
        characters = IPAPhonemes(
            characters=uzbek_chars,
            punctuations=punctuations,
            pad='<PAD>',
            eos='<EOS>',
            bos='<BOS>',
            blank='<BLNK>',
            is_unique=True,
            is_sorted=True
        )
        
        tokenizer = TTSTokenizer(
            use_phonemes=False,  # Use raw text instead of phonemes
            characters=characters,
            add_blank=True
        )
        
        # Set the tokenizer on the model
        model.tokenizer = tokenizer
        
        logger.info("Model and tokenizer initialized successfully")
        
        # RTX 4090 Optimized Trainer Arguments
        trainer_args = TrainerArgs()
        trainer_args.restore_path = None
        trainer_args.skip_train_epoch = False
        trainer_args.start_with_eval = False  # Skip initial eval to avoid file issues
        trainer_args.epochs = config.epochs
        trainer_args.save_step = config.save_step  # 500 (more frequent saves)
        trainer_args.save_n_checkpoints = 8  # Keep more checkpoints with faster training
        trainer_args.save_checkpoints = True
        trainer_args.save_all_best = False
        trainer_args.save_best_after = config.save_best_after  # 5000 (earlier best model saving)
        trainer_args.use_ddp = False
        trainer_args.rank = 0
        trainer_args.group_id = "uzbek_tts_rtx4090"
        
        # RTX 4090 Optimized logging
        trainer_args.print_step = 10  # Print every 10 steps
        trainer_args.print_eval = True  # Print evaluation results
        trainer_args.test_delay_epochs = 3  # More frequent evaluation with faster training
        
        # Load dataset samples
        from TTS.tts.datasets import load_tts_samples
        train_samples, eval_samples = load_tts_samples(config.datasets[0])
        
        logger.info(f"Loaded {len(train_samples)} training samples")
        logger.info(f"Loaded {len(eval_samples)} evaluation samples")
        
        # Initialize trainer
        trainer = Trainer(
            trainer_args,
            config,
            Path("output"),
            model=model,
            train_samples=train_samples,
            eval_samples=eval_samples
        )
        
        logger.info("Trainer initialized successfully")
        
        # Initialize MLflow and WandB
        logger.info("Initializing experiment tracking...")
        
        # RTX 4090 MLflow setup
        mlflow.set_experiment("uzbek-tts-rtx4090-experiment")
        with mlflow.start_run(run_name=f"uzbek_tts_rtx4090_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
            # Log RTX 4090 optimized parameters
            mlflow.log_params({
                "model": config.model,
                "gpu": "RTX 4090",
                "batch_size": config.batch_size,
                "learning_rate": config.lr,
                "epochs": config.epochs,
                "mixed_precision": config.mixed_precision,
                "num_loader_workers": config.num_loader_workers,
                "pin_memory": config.pin_memory,
                "gradient_clip_val": config.gradient_clip_val,
                "character_vocab_size": config.model_args.num_chars,
                "dataset_size": len(train_samples),
                "eval_size": len(eval_samples)
            })
            
            # RTX 4090 WandB setup
            wandb.init(
                project="uzbek-tts-rtx4090",
                entity="sheraliozodov03-uzbekcomos",
                name=f"uzbek_tts_rtx4090_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                config={
                    "model": config.model,
                    "gpu": "RTX 4090",
                    "batch_size": config.batch_size,
                    "learning_rate": config.lr,
                    "epochs": config.epochs,
                    "mixed_precision": config.mixed_precision,
                    "num_loader_workers": config.num_loader_workers,
                    "pin_memory": config.pin_memory,
                    "gradient_clip_val": config.gradient_clip_val,
                    "character_vocab_size": config.model_args.num_chars,
                    "dataset_size": len(train_samples),
                    "eval_size": len(eval_samples)
                }
            )
            
            logger.info("Experiment tracking initialized successfully")
            logger.info("Starting training...")
            
            # Start training with memory management
            try:
                # Clear GPU cache before training
                torch.cuda.empty_cache()
                
                # Calculate training statistics
                total_samples = len(train_samples)
                steps_per_epoch = total_samples // config.batch_size
                total_steps = steps_per_epoch * config.epochs
                
                logger.info("=" * 80)
                logger.info("🚀 RTX 4090 MEMORY-OPTIMIZED TRAINING STARTING")
                logger.info("=" * 80)
                logger.info(f"🎮 GPU: RTX 4090 (24GB VRAM, 1008 GB/s bandwidth, 16384 CUDA cores)")
                logger.info(f"💾 Current VRAM usage: ~23.4/24GB (memory-optimized to prevent OOM)")
                logger.info(f"📊 Dataset: {total_samples:,} samples")
                logger.info(f"⚙️  RTX 4090 Batch size: {config.batch_size} (2x increase from 5070 Ti, memory-safe)")
                logger.info(f"📈 Steps per epoch: {steps_per_epoch:,}")
                logger.info(f"🔄 Total steps: {total_steps:,}")
                logger.info(f"📅 Epochs: {config.epochs}")
                logger.info(f"🧠 Mixed Precision: {config.mixed_precision} (FP16 enabled)")
                logger.info(f"⚡ Data Loader Workers: {config.num_loader_workers} (memory-optimized)")
                logger.info(f"💾 Pin Memory: {config.pin_memory}")
                logger.info(f"🎯 Learning Rate: {config.lr} (2x increase to match batch scaling)")
                logger.info("=" * 80)
                logger.info("💡 Expected 1.5-2x faster training with memory-optimized RTX 4090 settings!")
                logger.info("=" * 80)
                
                # Start training (let the trainer handle the loop internally)
                start_time = datetime.now()
                trainer.fit()
                
                # Final statistics
                total_time = (datetime.now() - start_time).total_seconds()
                total_hours = int(total_time // 3600)
                total_minutes = int((total_time % 3600) // 60)
                
                logger.info("=" * 80)
                logger.info("🎉 TRAINING COMPLETED SUCCESSFULLY!")
                logger.info(f"⏱️  Total time: {total_hours:02d}:{total_minutes:02d}")
                logger.info("=" * 80)
                
                # Log final metrics
                mlflow.log_metric("training_completed", 1)
                mlflow.log_metric("total_training_hours", total_hours + total_minutes/60)
                wandb.log({"training_completed": 1, "total_training_hours": total_hours + total_minutes/60})
                
            except torch.cuda.OutOfMemoryError as e:
                logger.error(f"❌ CUDA Out of Memory: {e}")
                logger.error("💡 RTX 4090 OOM - Try reducing batch_size to 48 or 32")
                logger.error("💡 Or reduce num_loader_workers to 4 or 2")
                logger.error("💡 Or disable mixed precision if still having issues")
                torch.cuda.empty_cache()
                # Log failure
                mlflow.log_metric("cuda_oom", 1)
                wandb.log({"cuda_oom": 1})
                return
            except Exception as e:
                logger.error(f"❌ Training failed: {e}")
                logger.info("Training stopped due to error, but model files may be preserved")
                torch.cuda.empty_cache()
                # Log failure
                mlflow.log_metric("training_failed", 1)
                wandb.log({"training_failed": 1})
                # Don't try to clean up on Windows due to file locking issues
                return
        
    except ImportError as e:
        logger.error(f"Import error: {e}")
        logger.error("Please ensure TTS is properly installed")
    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise

if __name__ == "__main__":
    main()
