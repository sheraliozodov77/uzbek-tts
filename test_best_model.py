#!/usr/bin/env python3
"""
Quick test script to generate audio samples from the best trained model.
"""

import os
import sys
import torch
import logging
from pathlib import Path
import warnings

# Suppress warnings
warnings.filterwarnings("ignore")

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_best_model():
    """Test the best model with some Uzbek sentences."""
    
    # Check if best model exists
    best_model_path = Path("output/uzbek_tts_20250916_165454-September-29-2025_12+25PM-0000000/best_model.pth")
    config_path = Path("output/uzbek_tts_20250916_165454-September-29-2025_12+25PM-0000000/config.json")
    
    if not best_model_path.exists():
        logger.error(f"Best model not found at: {best_model_path}")
        return
    
    if not config_path.exists():
        logger.error(f"Config not found at: {config_path}")
        return
    
    logger.info("🎵 Testing Best Uzbek TTS Model")
    logger.info(f"Model: {best_model_path}")
    logger.info(f"Config: {config_path}")
    
    try:
        # Import TTS
        from TTS.api import TTS
        from TTS.tts.configs.vits_config import VitsConfig
        from TTS.tts.models.vits import Vits
        from TTS.utils.audio import AudioProcessor
        
        # Load config
        config = VitsConfig()
        config.load_json(str(config_path))
        
        # Define character set (Latin Uzbek only)
        uzbek_chars = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
        punctuations = "!'(),-.:;? \"\nʻʼ"
        
        # Initialize model
        model = Vits(config)
        
        # Load the best model
        logger.info("Loading best model...")
        checkpoint = torch.load(best_model_path, map_location='cpu')
        model.load_state_dict(checkpoint['model'])
        model.eval()
        
        # Initialize audio processor
        model.ap = AudioProcessor(**config.audio)
        
        # Setup tokenizer (same as training)
        from TTS.tts.utils.text.characters import IPAPhonemes
        from TTS.tts.utils.text.tokenizer import TTSTokenizer
        
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
            use_phonemes=False,
            characters=characters,
            add_blank=True
        )
        
        model.tokenizer = tokenizer
        
        logger.info("✅ Model loaded successfully!")
        
        # Test sentences
        test_sentences = [
            "Salom, qandaysiz? Bugun ob-havo juda yaxshi. Men Toshkentda yashayman. Uzbek tilida gapirishni o'rganyapman. Rahmat, sizga ham!"
        ]
        
        # Create output directory for samples
        output_dir = Path("test_samples")
        output_dir.mkdir(exist_ok=True)
        
        logger.info("🎤 Generating audio samples...")
        
        # Generate audio for each sentence
        for i, sentence in enumerate(test_sentences):
            try:
                logger.info(f"Generating: '{sentence}'")
                
                # Generate audio
                with torch.no_grad():
                    # Tokenize text
                    token_ids = tokenizer.text_to_ids(sentence)
                    token_ids = torch.LongTensor(token_ids).unsqueeze(0)
                    
                    # Generate audio
                    outputs = model.inference(token_ids)
                    audio = outputs['model_outputs'].squeeze().cpu().numpy()
                
                # Save audio
                output_file = output_dir / f"test_sample_{i+1:02d}.wav"
                model.ap.save_wav(audio, str(output_file))
                
                logger.info(f"✅ Saved: {output_file}")
                
            except Exception as e:
                logger.error(f"❌ Failed to generate '{sentence}': {e}")
                continue
        
        logger.info("🎉 Testing complete!")
        logger.info(f"📁 Audio samples saved in: {output_dir}")
        logger.info("🎧 You can now listen to the generated audio files!")
        
    except ImportError as e:
        logger.error(f"Import error: {e}")
        logger.error("Make sure TTS is properly installed")
    except Exception as e:
        logger.error(f"Testing failed: {e}")
        raise

if __name__ == "__main__":
    test_best_model()
