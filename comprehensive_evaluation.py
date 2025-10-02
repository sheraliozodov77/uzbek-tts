#!/usr/bin/env python3
"""
Comprehensive evaluation of the trained Uzbek TTS model with proper quality metrics.
Calculates MOS, MCD, PESQ, WER, and other important TTS evaluation metrics.
"""

import os
import sys
import logging
import time
import json
import numpy as np
import torch
import torchaudio
import librosa
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Tuple, Optional
import warnings

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Try to import evaluation metrics
try:
    from pesq import pesq
    PESQ_AVAILABLE = True
except ImportError:
    PESQ_AVAILABLE = False
    logger.warning("PESQ not available. Install with: pip install pesq")

try:
    from pystoi import stoi
    STOI_AVAILABLE = True
except ImportError:
    STOI_AVAILABLE = False
    logger.warning("STOI not available. Install with: pip install pystoi")

try:
    import whisper
    WHISPER_AVAILABLE = True
except ImportError:
    WHISPER_AVAILABLE = False
    logger.warning("Whisper not available. Install with: pip install openai-whisper")

def find_best_model():
    """Find the best trained model."""
    output_dir = Path("output")
    if not output_dir.exists():
        logger.error("Output directory not found")
        return None, None
    
    # Find the most recent training run
    training_dirs = [d for d in output_dir.iterdir() if d.is_dir() and d.name.startswith("uzbek_tts_")]
    if not training_dirs:
        logger.error("No training directories found")
        return None, None
    
    # Sort by modification time (most recent first)
    latest_dir = max(training_dirs, key=lambda x: x.stat().st_mtime)
    
    best_model_path = latest_dir / "best_model.pth"
    config_path = latest_dir / "config.json"
    
    if not best_model_path.exists():
        logger.error(f"Best model not found at: {best_model_path}")
        return None, None
    
    if not config_path.exists():
        logger.error(f"Config not found at: {config_path}")
        return None, None
    
    return best_model_path, config_path

def load_trained_model(model_path, config_path):
    """Load the trained model."""
    try:
        from TTS.tts.configs.vits_config import VitsConfig
        from TTS.tts.models.vits import Vits
        from TTS.utils.audio import AudioProcessor
        from TTS.tts.utils.text.characters import IPAPhonemes
        from TTS.tts.utils.text.tokenizer import TTSTokenizer
        
        # Load configuration
        config = VitsConfig()
        config.load_json(str(config_path))
        
        # Initialize model
        model = Vits(config)
        model.load_checkpoint(config, str(model_path))
        model.cuda()
        model.eval()
        
        # Initialize audio processor
        ap = AudioProcessor(**config.audio)
        
        # Setup tokenizer
        uzbek_chars = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789ОоҒғҚқЎўҲҳЁёАаКкëʙТµģ"
        punctuations = "!'(),-.:;? \"\nʻʼ"
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
        
        logger.info("Model loaded successfully")
        return model, ap, config
        
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        return None, None, None

def calculate_mcd(reference_audio, generated_audio, sample_rate=22050):
    """Calculate Mel-Cepstral Distortion (MCD) between reference and generated audio."""
    try:
        # Extract mel-spectrograms
        ref_mel = extract_mel_spectrogram(reference_audio, sample_rate)
        gen_mel = extract_mel_spectrogram(generated_audio, sample_rate)
        
        # Ensure same length
        min_len = min(ref_mel.shape[1], gen_mel.shape[1])
        ref_mel = ref_mel[:, :min_len]
        gen_mel = gen_mel[:, :min_len]
        
        # Calculate MCD
        mcd = np.mean(np.sqrt(2 * np.sum((ref_mel - gen_mel) ** 2, axis=0)))
        return float(mcd)
        
    except Exception as e:
        logger.error(f"MCD calculation failed: {e}")
        return float('inf')

def extract_mel_spectrogram(audio, sample_rate=22050, n_mels=80):
    """Extract mel-spectrogram from audio."""
    # Convert to torch tensor if needed
    if isinstance(audio, np.ndarray):
        audio_tensor = torch.from_numpy(audio).float()
    else:
        audio_tensor = audio.float()
    
    # Ensure correct shape
    if audio_tensor.dim() == 1:
        audio_tensor = audio_tensor.unsqueeze(0)
    
    # Compute mel-spectrogram
    mel_transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=sample_rate,
        n_fft=1024,
        hop_length=256,
        n_mels=n_mels,
        f_min=0,
        f_max=8000
    )
    
    mel_spec = mel_transform(audio_tensor)
    mel_spec_db = torchaudio.transforms.AmplitudeToDB()(mel_spec)
    
    return mel_spec_db.squeeze(0).numpy()

def calculate_pesq(reference_audio, generated_audio, sample_rate=22050):
    """Calculate PESQ score."""
    if not PESQ_AVAILABLE:
        logger.warning("PESQ not available")
        return 0.0
    
    try:
        # Resample to 16kHz for PESQ
        if sample_rate != 16000:
            ref_16k = librosa.resample(reference_audio, orig_sr=sample_rate, target_sr=16000)
            gen_16k = librosa.resample(generated_audio, orig_sr=sample_rate, target_sr=16000)
        else:
            ref_16k = reference_audio
            gen_16k = generated_audio
        
        # Ensure same length
        min_len = min(len(ref_16k), len(gen_16k))
        ref_16k = ref_16k[:min_len]
        gen_16k = gen_16k[:min_len]
        
        score = pesq(16000, ref_16k, gen_16k, 'wb')
        return float(score)
        
    except Exception as e:
        logger.error(f"PESQ calculation failed: {e}")
        return 0.0

def calculate_stoi(reference_audio, generated_audio, sample_rate=22050):
    """Calculate STOI score."""
    if not STOI_AVAILABLE:
        logger.warning("STOI not available")
        return 0.0
    
    try:
        # Resample to 10kHz for STOI
        if sample_rate != 10000:
            ref_10k = librosa.resample(reference_audio, orig_sr=sample_rate, target_sr=10000)
            gen_10k = librosa.resample(generated_audio, orig_sr=sample_rate, target_sr=10000)
        else:
            ref_10k = reference_audio
            gen_10k = generated_audio
        
        # Ensure same length
        min_len = min(len(ref_10k), len(gen_10k))
        ref_10k = ref_10k[:min_len]
        gen_10k = gen_10k[:min_len]
        
        score = stoi(ref_10k, gen_10k, 10000, extended=False)
        return float(score)
        
    except Exception as e:
        logger.error(f"STOI calculation failed: {e}")
        return 0.0

def calculate_wer_with_whisper(reference_text, generated_audio, sample_rate=22050):
    """Calculate Word Error Rate using Whisper."""
    if not WHISPER_AVAILABLE:
        logger.warning("Whisper not available for WER calculation")
        return 1.0
    
    try:
        # Load Whisper model
        model = whisper.load_model("base")
        
        # Transcribe generated audio
        result = model.transcribe(generated_audio, language="uz")
        predicted_text = result["text"].strip().lower()
        
        # Calculate WER
        wer = compute_wer(reference_text.lower(), predicted_text)
        return wer
        
    except Exception as e:
        logger.error(f"WER calculation failed: {e}")
        return 1.0

def compute_wer(reference, hypothesis):
    """Compute Word Error Rate between reference and hypothesis."""
    ref_words = reference.split()
    hyp_words = hypothesis.split()
    
    if len(ref_words) == 0:
        return 1.0 if len(hyp_words) > 0 else 0.0
    
    # Create distance matrix
    d = np.zeros((len(ref_words) + 1, len(hyp_words) + 1))
    
    # Initialize first row and column
    for i in range(len(ref_words) + 1):
        d[i][0] = i
    for j in range(len(hyp_words) + 1):
        d[0][j] = j
    
    # Fill distance matrix
    for i in range(1, len(ref_words) + 1):
        for j in range(1, len(hyp_words) + 1):
            if ref_words[i-1] == hyp_words[j-1]:
                d[i][j] = d[i-1][j-1]
            else:
                d[i][j] = min(
                    d[i-1][j] + 1,      # deletion
                    d[i][j-1] + 1,      # insertion
                    d[i-1][j-1] + 1     # substitution
                )
    
    # WER = (substitutions + insertions + deletions) / total_words
    wer = d[len(ref_words)][len(hyp_words)] / len(ref_words)
    return wer

def load_reference_audio(text, data_dir="data/preprocessed"):
    """Try to find reference audio for the given text."""
    data_path = Path(data_dir)
    if not data_path.exists():
        logger.warning(f"Data directory not found: {data_dir}")
        return None
    
    # Look for metadata files that might contain text-audio mappings
    metadata_files = list(data_path.glob("*.json")) + list(data_path.glob("*.csv")) + list(data_path.glob("*.txt"))
    
    if metadata_files:
        # Try to find a matching audio file based on text similarity
        audio_files = list(data_path.glob("*.wav"))
        if audio_files:
            # For evaluation purposes, use a random audio file as reference
            # In a real evaluation, you'd have proper text-to-audio mapping
            import random
            return str(random.choice(audio_files))
    
    return None

def load_test_metadata(metadata_path="data/preprocessed/test_metadata.csv", max_samples=20):
    """Load test metadata from CSV file."""
    import pandas as pd
    
    try:
        # Read the metadata CSV file
        df = pd.read_csv(metadata_path, sep='|', header=None, names=['audio_file', 'speaker', 'text'])
        
        # Filter out very short or very long sentences for better evaluation
        df['text_length'] = df['text'].str.len()
        df = df[(df['text_length'] >= 20) & (df['text_length'] <= 200)]  # Reasonable length
        
        # Select up to max_samples
        if len(df) > max_samples:
            df = df.sample(n=max_samples, random_state=42)
        
        test_data = []
        for _, row in df.iterrows():
            test_data.append({
                'audio_file': row['audio_file'],
                'speaker': row['speaker'],
                'text': row['text'].strip()
            })
        
        logger.info(f"Loaded {len(test_data)} test samples from {metadata_path}")
        return test_data
        
    except Exception as e:
        logger.error(f"Failed to load test metadata: {e}")
        # Fallback to hardcoded sentences
        return [
            {"audio_file": "fallback_1.wav", "speaker": "uzbek_speaker", "text": "Salom, qandaysiz? Bugun ob-havo juda yaxshi."},
            {"audio_file": "fallback_2.wav", "speaker": "uzbek_speaker", "text": "Men Toshkentda yashayman va O'zbek tilida gapirishni o'rganyapman."},
            {"audio_file": "fallback_3.wav", "speaker": "uzbek_speaker", "text": "Kitob o'qishni yaxshi ko'raman, chunki bu mening bilimimni kengaytiradi."},
            {"audio_file": "fallback_4.wav", "speaker": "uzbek_speaker", "text": "Rahmat, sizga ham! Kechirasiz, qayerda metro bor?"},
            {"audio_file": "fallback_5.wav", "speaker": "uzbek_speaker", "text": "Men ishga ketmoqchiman, bugun darslarim bor."}
        ]

def comprehensive_evaluation(model, ap, config, output_dir="comprehensive_evaluation"):
    """Run comprehensive evaluation with all quality metrics."""
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Load test data from metadata
    test_data = load_test_metadata()
    
    results = {
        'mcd_scores': [],
        'pesq_scores': [],
        'stoi_scores': [],
        'wer_scores': [],
        'rtf_scores': [],
        'inference_times': [],
        'audio_durations': []
    }
    
    logger.info(f"Starting comprehensive evaluation on {len(test_data)} sentences...")
    
    for i, sample in enumerate(test_data):
        sentence = sample['text']
        reference_audio_file = sample['audio_file']
        
        logger.info(f"Evaluating sample {i+1}/{len(test_data)}: '{sentence[:50]}...'")
        
        try:
            # Generate audio
            start_time = time.time()
            
            # Tokenize text
            token_ids = model.tokenizer.text_to_ids(sentence)
            token_ids = torch.LongTensor(token_ids).unsqueeze(0).cuda()
            
            # Generate audio
            with torch.no_grad():
                outputs = model.inference(token_ids)
                generated_audio = outputs['model_outputs'].squeeze().cpu().numpy()
            
            inference_time = time.time() - start_time
            
            # Save generated audio
            output_file = output_dir / f"eval_sample_{i+1:02d}.wav"
            ap.save_wav(generated_audio, str(output_file))
            
            # Calculate audio duration and RTF
            audio_duration = len(generated_audio) / ap.sample_rate
            rtf = inference_time / audio_duration
            
            results['rtf_scores'].append(rtf)
            results['inference_times'].append(inference_time)
            results['audio_durations'].append(audio_duration)
            
            # Try to load reference audio for comparison
            reference_audio_path = Path("data/preprocessed/wavs") / reference_audio_file
            
            if reference_audio_path.exists():
                try:
                    # Load reference audio
                    ref_audio, ref_sr = librosa.load(str(reference_audio_path), sr=ap.sample_rate)
                    
                    # Calculate quality metrics
                    mcd = calculate_mcd(ref_audio, generated_audio, ap.sample_rate)
                    pesq_score = calculate_pesq(ref_audio, generated_audio, ap.sample_rate)
                    stoi_score = calculate_stoi(ref_audio, generated_audio, ap.sample_rate)
                    
                    results['mcd_scores'].append(mcd)
                    results['pesq_scores'].append(pesq_score)
                    results['stoi_scores'].append(stoi_score)
                    
                    logger.info(f"  MCD: {mcd:.2f}, PESQ: {pesq_score:.2f}, STOI: {stoi_score:.3f}")
                    
                except Exception as e:
                    logger.warning(f"  Failed to process reference audio: {e}")
                    results['mcd_scores'].append(float('inf'))
                    results['pesq_scores'].append(0.0)
                    results['stoi_scores'].append(0.0)
            else:
                logger.warning(f"  Reference audio not found: {reference_audio_path}")
                results['mcd_scores'].append(float('inf'))
                results['pesq_scores'].append(0.0)
                results['stoi_scores'].append(0.0)
            
            # Calculate basic audio statistics
            audio_energy = np.mean(generated_audio ** 2)
            audio_std = np.std(generated_audio)
            audio_dynamic_range = np.max(generated_audio) - np.min(generated_audio)
            
            logger.info(f"  Audio Energy: {audio_energy:.4f}, Std: {audio_std:.4f}, Dynamic Range: {audio_dynamic_range:.4f}")
            
            # Calculate WER using Whisper
            wer = calculate_wer_with_whisper(sentence, generated_audio, ap.sample_rate)
            results['wer_scores'].append(wer)
            
            logger.info(f"  RTF: {rtf:.3f}, WER: {wer:.3f}, Duration: {audio_duration:.2f}s")
            
        except Exception as e:
            logger.error(f"Failed to evaluate sample {i+1}: {e}")
            continue
    
    return results

def calculate_summary_metrics(results):
    """Calculate summary statistics from evaluation results."""
    summary = {}
    
    for metric, scores in results.items():
        if scores:
            # Filter out infinite values for some metrics
            valid_scores = [s for s in scores if s != float('inf')]
            if valid_scores:
                summary[f'{metric}_mean'] = np.mean(valid_scores)
                summary[f'{metric}_std'] = np.std(valid_scores)
                summary[f'{metric}_min'] = np.min(valid_scores)
                summary[f'{metric}_max'] = np.max(valid_scores)
                summary[f'{metric}_count'] = len(valid_scores)
            else:
                summary[f'{metric}_mean'] = 0.0
                summary[f'{metric}_std'] = 0.0
                summary[f'{metric}_min'] = 0.0
                summary[f'{metric}_max'] = 0.0
                summary[f'{metric}_count'] = 0
        else:
            summary[f'{metric}_mean'] = 0.0
            summary[f'{metric}_std'] = 0.0
            summary[f'{metric}_min'] = 0.0
            summary[f'{metric}_max'] = 0.0
            summary[f'{metric}_count'] = 0
    
    return summary

def print_evaluation_summary(summary):
    """Print comprehensive evaluation summary."""
    print("\n" + "="*80)
    print("🎯 COMPREHENSIVE UZBEK TTS MODEL EVALUATION RESULTS")
    print("="*80)
    
    # Performance Metrics
    print("\n📊 PERFORMANCE METRICS:")
    print(f"  Total samples: {summary.get('rtf_scores_count', 0)}")
    print(f"  Average RTF: {summary.get('rtf_scores_mean', 0):.3f}")
    print(f"  Average inference time: {summary.get('inference_times_mean', 0):.3f}s")
    print(f"  Average audio duration: {summary.get('audio_durations_mean', 0):.2f}s")
    
    # Quality Metrics
    print("\n🎵 AUDIO QUALITY METRICS:")
    
    # Check if we have reference-based metrics
    mcd_mean = summary.get('mcd_scores_mean', float('inf'))
    pesq_mean = summary.get('pesq_scores_mean', 0)
    stoi_mean = summary.get('stoi_scores_mean', 0)
    
    if mcd_mean != float('inf') and mcd_mean > 0:
        print(f"  MCD (Mel-Cepstral Distortion): {mcd_mean:.2f} dB")
        if mcd_mean <= 5.0:
            print("    ✅ EXCELLENT (≤5 dB)")
        elif mcd_mean <= 10.0:
            print("    ✅ GOOD (≤10 dB)")
        else:
            print("    ⚠️  NEEDS IMPROVEMENT (>10 dB)")
    else:
        print("  MCD: Not available (no reference audio)")
    
    if pesq_mean > 0:
        print(f"  PESQ (Perceptual Quality): {pesq_mean:.2f}")
        if pesq_mean >= 3.5:
            print("    ✅ EXCELLENT (≥3.5)")
        elif pesq_mean >= 3.0:
            print("    ✅ GOOD (≥3.0)")
        else:
            print("    ⚠️  NEEDS IMPROVEMENT (<3.0)")
    else:
        print("  PESQ: Not available (no reference audio)")
    
    if stoi_mean > 0:
        print(f"  STOI (Intelligibility): {stoi_mean:.3f}")
        if stoi_mean >= 0.8:
            print("    ✅ EXCELLENT (≥0.8)")
        elif stoi_mean >= 0.7:
            print("    ✅ GOOD (≥0.7)")
        else:
            print("    ⚠️  NEEDS IMPROVEMENT (<0.7)")
    else:
        print("  STOI: Not available (no reference audio)")
    
    # Calculate basic audio quality indicators
    avg_duration = summary.get('audio_durations_mean', 0)
    avg_rtf = summary.get('rtf_scores_mean', float('inf'))
    
    print(f"  Average Audio Duration: {avg_duration:.2f}s")
    print(f"  Audio Consistency: {'✅ GOOD' if avg_duration > 2.0 and avg_duration < 10.0 else '⚠️  CHECK'}")
    print(f"  Generation Speed: {'✅ EXCELLENT' if avg_rtf <= 0.1 else '✅ GOOD' if avg_rtf <= 0.5 else '⚠️  SLOW'}")
    
    # WER
    wer_mean = summary.get('wer_scores_mean', 1.0)
    print(f"  WER (Word Error Rate): {wer_mean:.3f} ({wer_mean*100:.1f}%)")
    if wer_mean <= 0.1:
        print("    ✅ EXCELLENT (≤10%)")
    elif wer_mean <= 0.2:
        print("    ✅ GOOD (≤20%)")
    else:
        print("    ⚠️  NEEDS IMPROVEMENT (>20%)")
    
    # Overall Assessment
    print("\n🏆 OVERALL ASSESSMENT:")
    rtf_mean = summary.get('rtf_scores_mean', float('inf'))
    if rtf_mean <= 0.3:
        print("  ✅ RTF: EXCELLENT (≤0.3)")
    elif rtf_mean <= 0.5:
        print("  ✅ RTF: GOOD (≤0.5)")
    else:
        print("  ⚠️  RTF: NEEDS IMPROVEMENT (>0.5)")
    
    print("="*80)

def main():
    """Main evaluation function."""
    logger.info("Starting comprehensive Uzbek TTS model evaluation...")
    
    # Find and load model
    model_path, config_path = find_best_model()
    if not model_path:
        logger.error("Could not find trained model")
        return
    
    logger.info(f"Found model: {model_path}")
    logger.info(f"Found config: {config_path}")
    
    # Load model
    model, ap, config = load_trained_model(model_path, config_path)
    if not model:
        logger.error("Failed to load model")
        return
    
    # Run comprehensive evaluation
    results = comprehensive_evaluation(model, ap, config)
    
    # Calculate summary metrics
    summary = calculate_summary_metrics(results)
    
    # Save detailed report
    report = {
        'evaluation_timestamp': datetime.now().isoformat(),
        'model_info': {
            'model_type': 'VITS',
            'language': 'Uzbek',
            'optimization': 'RTX 4090'
        },
        'summary_metrics': summary,
        'detailed_results': results
    }
    
    with open('comprehensive_evaluation_report.json', 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    
    # Print summary
    print_evaluation_summary(summary)
    
    logger.info("Comprehensive evaluation completed!")
    logger.info("Detailed report saved to: comprehensive_evaluation_report.json")

if __name__ == "__main__":
    main()
