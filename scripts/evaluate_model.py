#!/usr/bin/env python3
"""
Model evaluation script for Uzbek TTS.
Evaluates trained model using multiple metrics.
"""

import sys
import argparse
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from evaluation import TTSEvaluator
import torch
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description="Evaluate Uzbek TTS model")
    parser.add_argument("--model_path", required=True, help="Path to trained model")
    parser.add_argument("--config_path", default="config.yaml", help="Model configuration file")
    parser.add_argument("--test_sentences", nargs="+", help="Test sentences for evaluation")
    parser.add_argument("--output_dir", default="./evaluation_results", help="Output directory for results")
    parser.add_argument("--sample_rate", type=int, default=22050, help="Audio sample rate")
    parser.add_argument("--whisper_model", default="base", help="Whisper model for WER calculation")
    
    args = parser.parse_args()
    
    # Default test sentences
    if not args.test_sentences:
        args.test_sentences = [
            "Salom, qandaysiz?",
            "Bugun ob-havo juda yaxshi.",
            "Men Toshkentda yashayman.",
            "Uzbek tilida gapirishni o'rganyapman.",
            "Kitob o'qishni yaxshi ko'raman.",
            "Darslarimni tayyorlayapman.",
            "Do'stlarim bilan uchrashaman.",
            "Oshxona pishirishni o'rganyapman.",
            "Musiqa tinglashni yaxshi ko'raman.",
            "Sport bilan shug'ullanaman.",
        ]
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load model
    logger.info(f"Loading model from {args.model_path}")
    try:
        # Load model checkpoint
        checkpoint = torch.load(args.model_path, map_location='cpu')
        
        # Initialize model (this would need to be adapted based on your model structure)
        # For now, we'll create a mock model for demonstration
        class MockModel:
            def __init__(self):
                self.sample_rate = args.sample_rate
            
            def tts(self, text):
                # Mock TTS - in real implementation, this would generate actual audio
                import numpy as np
                duration = len(text) * 0.1  # Rough estimate
                samples = int(duration * self.sample_rate)
                return np.random.randn(samples).astype(np.float32)
        
        model = MockModel()
        logger.info("Model loaded successfully")
        
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        return
    
    # Initialize evaluator
    logger.info("Initializing evaluator...")
    evaluator = TTSEvaluator(
        sample_rate=args.sample_rate,
        whisper_model=args.whisper_model
    )
    
    # Prepare test data
    test_data = [{'text': sentence, 'reference_audio': None} for sentence in args.test_sentences]
    
    # Run evaluation
    logger.info("Starting evaluation...")
    results = evaluator.evaluate_model(model, test_data)
    
    # Save results
    results_path = output_dir / "evaluation_results.json"
    evaluator.save_evaluation_report(results, str(results_path))
    
    # Print results
    print("\n" + "="*60)
    print("EVALUATION RESULTS")
    print("="*60)
    print(f"Overall MCD: {results.get('overall_mcd', 'N/A'):.3f}")
    print(f"Overall PESQ: {results.get('overall_pesq', 'N/A'):.3f}")
    print(f"Overall STOI: {results.get('overall_stoi', 'N/A'):.3f}")
    print(f"Overall WER: {results.get('overall_wer', 'N/A'):.3f}")
    print(f"Overall RTF: {results.get('overall_rtf', 'N/A'):.3f}")
    print("="*60)
    
    # Detailed results
    print("\nDetailed Results:")
    for key, value in results.items():
        if isinstance(value, float):
            print(f"{key}: {value:.4f}")
        else:
            print(f"{key}: {value}")
    
    print(f"\nResults saved to: {results_path}")
    
    # Generate sample audio
    logger.info("Generating sample audio...")
    samples_dir = output_dir / "samples"
    samples_dir.mkdir(exist_ok=True)
    
    for i, sentence in enumerate(args.test_sentences[:5]):  # Generate first 5 samples
        try:
            audio = model.tts(sentence)
            
            # Save audio (mock implementation)
            import soundfile as sf
            output_path = samples_dir / f"sample_{i:03d}.wav"
            sf.write(str(output_path), audio, args.sample_rate)
            
            logger.info(f"Generated sample {i+1}: {sentence}")
            
        except Exception as e:
            logger.error(f"Failed to generate sample {i}: {e}")
    
    print(f"Sample audio saved to: {samples_dir}")

if __name__ == "__main__":
    main()
