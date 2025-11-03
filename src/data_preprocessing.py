"""
Data preprocessing pipeline for Uzbek TTS training.
Handles phonemization, text cleaning, and dataset preparation.
"""

import os
import re
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Tuple, Dict, Optional
import logging

# Optional imports
try:
    from phonemizer import phonemize
    from phonemizer.backend import EspeakBackend
    PHONEMIZER_AVAILABLE = True
except ImportError:
    PHONEMIZER_AVAILABLE = False
    print("Warning: phonemizer not available. Install with: pip install phonemizer")


try:
    import unidecode
    UNIDECODE_AVAILABLE = True
except ImportError:
    UNIDECODE_AVAILABLE = False
    print("Warning: unidecode not available. Install with: pip install unidecode")

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class UzbekTextProcessor:
    """Uzbek text processing and phonemization."""
    
    def __init__(self, use_espeak: bool = True):
        # Check if we're on Windows and disable espeak-ng
        import platform
        if platform.system() == "Windows":
            logger.info("Windows detected - using fallback phonemization (espeak-ng not available)")
            self.use_espeak = False
        else:
            self.use_espeak = use_espeak and PHONEMIZER_AVAILABLE
            if self.use_espeak:
                try:
                    self.backend = EspeakBackend('uz', preserve_punctuation=True)
                    logger.info("Initialized espeak-ng backend for Uzbek")
                except Exception as e:
                    logger.warning(f"Failed to initialize espeak-ng: {e}")
                    self.use_espeak = False
        
        # Uzbek-specific text cleaning patterns
        self.cleaning_patterns = [
            (r'[^\w\s\'\-.,!?;:]', ''),  # Remove special chars except basic punctuation
            (r'\s+', ' '),  # Normalize whitespace
            (r'^\s+|\s+$', ''),  # Trim
        ]
        
        # Common Uzbek abbreviations and their expansions
        self.abbreviations = {
            'dr.': 'doktor',
            'prof.': 'professor',
            'mr.': 'janob',
            'mrs.': 'xonim',
            'etc.': 'va boshqalar',
            'vs.': 'qarshi',
            'kg': 'kilogramm',
            'km': 'kilometr',
            'm': 'metr',
            'cm': 'santimetr',
            'mm': 'millimetr',
            'g': 'gramm',
            'ml': 'millilitr',
            'l': 'litr',
        }
    
    def clean_text(self, text: str) -> str:
        """Clean and normalize Uzbek text."""
        if not text or pd.isna(text):
            return ""
        
        text = str(text).strip()
        
        # Minimal cleaning - just normalize whitespace
        text = re.sub(r'\s+', ' ', text)  # Normalize whitespace
        text = text.strip()
        
        return text
    
    def _convert_numbers_to_words(self, text: str) -> str:
        """Convert numbers to Uzbek words (basic implementation)."""
        # Simple number to word conversion for common numbers
        number_map = {
            '0': 'nol', '1': 'bir', '2': 'ikki', '3': 'uch', '4': 'to\'rt',
            '5': 'besh', '6': 'olti', '7': 'yetti', '8': 'sakkiz', '9': 'to\'qqiz',
            '10': 'o\'n', '20': 'yigirma', '30': 'o\'ttiz', '40': 'qirq',
            '50': 'ellik', '60': 'oltmish', '70': 'yetmish', '80': 'sakson',
            '90': 'to\'qson', '100': 'yuz', '1000': 'ming'
        }
        
        for num, word in number_map.items():
            text = re.sub(r'\b' + num + r'\b', word, text)
        
        return text
    
    
    def phonemize_text(self, text: str) -> str:
        """Convert text to phonemes."""
        if not text:
            return ""
        
        try:
            if self.use_espeak:
                # Try espeak-ng for phonemization
                try:
                    phonemes = phonemize(
                        text,
                        language='uz',
                        backend='espeak',
                        separator=' ',
                        preserve_punctuation=True,
                        njobs=1
                    )
                    # Check if phonemization actually worked
                    if phonemes and len(phonemes.strip()) > 0:
                        return phonemes
                    else:
                        logger.warning(f"espeak-ng returned empty phonemes for: {text}")
                        return self._fallback_phonemization(text)
                except Exception as e:
                    logger.warning(f"espeak-ng failed for '{text}': {e}")
                    return self._fallback_phonemization(text)
            else:
                # Fallback: return cleaned text
                logger.warning("espeak-ng not available, using fallback phonemization")
                return self._fallback_phonemization(text)
                
        except Exception as e:
            logger.error(f"Phonemization failed for text '{text}': {e}")
            return self._fallback_phonemization(text)
    
    def _fallback_phonemization(self, text: str) -> str:
        """Fallback phonemization when espeak-ng is not available."""
        # Simple character-based phonemization for Uzbek
        # This is a basic fallback - not perfect but better than nothing
        
        # Clean the text first
        cleaned = self.clean_text(text)
        
        # Basic Uzbek character mapping to approximate phonemes
        char_to_phoneme = {
            'a': 'a', 'b': 'b', 'c': 'ts', 'd': 'd', 'e': 'e', 'f': 'f', 'g': 'g',
            'h': 'h', 'i': 'i', 'j': 'j', 'k': 'k', 'l': 'l', 'm': 'm', 'n': 'n',
            'o': 'o', 'p': 'p', 'q': 'q', 'r': 'r', 's': 's', 't': 't', 'u': 'u',
            'v': 'v', 'w': 'w', 'x': 'x', 'y': 'y', 'z': 'z',
            'ʻ': 'ʔ', 'ʼ': 'ʔ', 'ʼ': 'ʔ',  # Uzbek glottal stops
            'oʻ': 'oʔ', 'gʻ': 'gʔ', 'ʻ': 'ʔ',  # Uzbek special characters
        }
        
        # Convert text to phonemes
        phonemes = []
        i = 0
        while i < len(cleaned):
            char = cleaned[i].lower()
            
            # Handle special Uzbek characters
            if i < len(cleaned) - 1:
                two_char = cleaned[i:i+2].lower()
                if two_char in char_to_phoneme:
                    phonemes.append(char_to_phoneme[two_char])
                    i += 2
                    continue
            
            # Handle single characters
            if char in char_to_phoneme:
                phonemes.append(char_to_phoneme[char])
            elif char.isalpha():
                phonemes.append(char)  # Keep unknown letters as-is
            elif char == ' ':
                phonemes.append(' ')
            # Skip punctuation for now
            
            i += 1
        
        result = ' '.join(phonemes).strip()
        return result if result else cleaned


class DatasetPreprocessor:
    """Main dataset preprocessing class."""
    
    def __init__(self, data_dir: str, output_dir: str):
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.text_processor = UzbekTextProcessor()
        
        # Create output directories
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / "wavs").mkdir(exist_ok=True)
    
    def load_metadata(self, metadata_file: str) -> pd.DataFrame:
        """Load metadata from CSV file."""
        metadata_path = self.data_dir / metadata_file
        if not metadata_path.exists():
            raise FileNotFoundError(f"Metadata file not found: {metadata_path}")
        
        # Try 3-column format first, fallback to 2-column
        try:
            df = pd.read_csv(metadata_path, sep='|', header=None, names=['audio_file', 'speaker', 'text'])
            # Check if text column is empty (2-column format)
            if df['text'].isna().all() or df['text'].eq('').all():
                # Re-read as 2-column format
                df = pd.read_csv(metadata_path, sep='|', header=None, names=['audio_file', 'text'])
                df['speaker'] = 'uzbek_speaker'  # Add default speaker
        except:
            # Fallback to 2-column format
            df = pd.read_csv(metadata_path, sep='|', header=None, names=['audio_file', 'text'])
            df['speaker'] = 'uzbek_speaker'  # Add default speaker
        logger.info(f"Loaded {len(df)} samples from {metadata_file}")
        return df
    
    def preprocess_metadata(self, df: pd.DataFrame) -> pd.DataFrame:
        """Preprocess metadata with text cleaning and phonemization."""
        logger.info("Starting text preprocessing and phonemization...")
        
        processed_data = []
        
        for idx, row in df.iterrows():
            if idx % 1000 == 0:
                logger.info(f"Processing sample {idx}/{len(df)}")
            
            audio_file = row['audio_file']
            speaker = row['speaker']
            text = row['text']
            
            # Clean text
            cleaned_text = self.text_processor.clean_text(text)
            
            # Use cleaned text as-is (Latin Uzbek only)
            latin_text = cleaned_text
            
            # Phonemize
            phonemes = self.text_processor.phonemize_text(latin_text)
            
            # Skip if phonemization failed or text is too short
            if not phonemes or len(phonemes.strip()) < 2:
                logger.warning(f"Skipping sample {idx}: invalid phonemes")
                continue
            
            processed_data.append({
                'audio_file': audio_file,
                'speaker': speaker,
                'text': cleaned_text,
                'phonemes': phonemes,
                'original_text': text
            })
        
        result_df = pd.DataFrame(processed_data)
        logger.info(f"Preprocessed {len(result_df)} samples successfully")
        return result_df
    
    def split_dataset(self, df: pd.DataFrame, train_ratio: float = 0.8, val_ratio: float = 0.1) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Split dataset into train/validation/test sets."""
        # Shuffle the dataset
        df_shuffled = df.sample(frac=1, random_state=42).reset_index(drop=True)
        
        n_samples = len(df_shuffled)
        n_train = int(n_samples * train_ratio)
        n_val = int(n_samples * val_ratio)
        
        train_df = df_shuffled[:n_train]
        val_df = df_shuffled[n_train:n_train + n_val]
        test_df = df_shuffled[n_train + n_val:]
        
        logger.info(f"Dataset split: Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)}")
        return train_df, val_df, test_df
    
    def save_metadata(self, df: pd.DataFrame, filename: str, use_phonemes: bool = True):
        """Save metadata in TTS format."""
        output_path = self.output_dir / filename
        
        # Select text column (phonemes or original text)
        text_col = 'phonemes' if use_phonemes else 'text'
        
        # Format: audio_file|speaker|text
        formatted_data = []
        for _, row in df.iterrows():
            line = f"{row['audio_file']}|{row['speaker']}|{row[text_col]}"
            formatted_data.append(line)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(formatted_data))
        
        logger.info(f"Saved {len(formatted_data)} samples to {output_path}")
    
    def create_character_set(self, df: pd.DataFrame) -> str:
        """Create character set from phonemes."""
        if 'phonemes' not in df.columns or len(df) == 0:
            logger.warning("No phonemes data available, using default character set")
            return "abcdefghijklmnopqrstuvwxyzʔ"
        
        all_phonemes = ' '.join(df['phonemes'].tolist())
        unique_chars = sorted(set(all_phonemes))
        
        # Remove empty strings and whitespace
        unique_chars = [c for c in unique_chars if c.strip()]
        
        logger.info(f"Found {len(unique_chars)} unique characters in phonemes")
        return ''.join(unique_chars)
    
    def process_dataset(self, metadata_file: str = "metadata.csv"):
        """Main processing pipeline."""
        logger.info("Starting dataset preprocessing...")
        
        # Load metadata
        df = self.load_metadata(metadata_file)
        
        # Preprocess text and phonemes
        processed_df = self.preprocess_metadata(df)
        
        # Split dataset
        train_df, val_df, test_df = self.split_dataset(processed_df)
        
        # Save metadata files
        self.save_metadata(train_df, "train_metadata.csv", use_phonemes=True)
        self.save_metadata(val_df, "val_metadata.csv", use_phonemes=True)
        self.save_metadata(test_df, "test_metadata.csv", use_phonemes=True)
        
        # Also save with original text for comparison
        self.save_metadata(train_df, "train_metadata_text.csv", use_phonemes=False)
        self.save_metadata(val_df, "val_metadata_text.csv", use_phonemes=False)
        
        # Create character set
        char_set = self.create_character_set(processed_df)
        char_set_path = self.output_dir / "character_set.txt"
        with open(char_set_path, 'w', encoding='utf-8') as f:
            f.write(char_set)
        
        # Save processing statistics
        stats = {
            'total_samples': len(processed_df),
            'train_samples': len(train_df),
            'val_samples': len(val_df),
            'test_samples': len(test_df),
            'unique_characters': len(char_set),
            'character_set': char_set,
            'average_phoneme_length': processed_df['phonemes'].str.len().mean() if len(processed_df) > 0 else 0,
            'average_text_length': processed_df['text'].str.len().mean() if len(processed_df) > 0 else 0,
        }
        
        stats_path = self.output_dir / "preprocessing_stats.json"
        import json
        with open(stats_path, 'w', encoding='utf-8') as f:
            json.dump(stats, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Preprocessing completed. Statistics saved to {stats_path}")
        return stats
    
    def process_dataset_raw_text(self, metadata_file: str = "metadata.csv"):
        """Process dataset without phonemization - use raw text."""
        logger.info("Starting dataset preprocessing (raw text mode)...")
        
        # Load metadata
        df = self.load_metadata(metadata_file)
        
        # Clean text but don't phonemize
        logger.info("Cleaning text (no phonemization)...")
        processed_data = []
        
        for idx, row in df.iterrows():
            if idx % 1000 == 0:
                logger.info(f"Processing sample {idx}/{len(df)}")
            
            audio_file = row['audio_file']
            speaker = row['speaker']
            text = row['text']
            
            # Clean text only
            cleaned_text = self.text_processor.clean_text(text)
            
            # Use cleaned text as-is (Latin Uzbek only)
            latin_text = cleaned_text
            
            # Debug output
            if hasattr(self, 'debug') and self.debug and idx < 10:
                print(f"Sample {idx}:")
                print(f"  Original: '{text}'")
                print(f"  Cleaned: '{cleaned_text}'")
                print(f"  Latin: '{latin_text}'")
                print(f"  Length: {len(latin_text.strip())}")
                print("-" * 50)
            
            # Skip if text is too short (more lenient threshold)
            if not latin_text or len(latin_text.strip()) < 2:
                logger.warning(f"Skipping sample {idx}: text too short (length: {len(latin_text.strip()) if latin_text else 0})")
                continue
            
            processed_data.append({
                'audio_file': audio_file,
                'speaker': speaker,
                'text': latin_text,
                'phonemes': latin_text,  # Use text as phonemes
                'original_text': text
            })
        
        result_df = pd.DataFrame(processed_data)
        logger.info(f"Preprocessed {len(result_df)} samples successfully")
        
        # Split dataset
        train_df, val_df, test_df = self.split_dataset(result_df)
        
        # Save metadata files (using text as phonemes)
        self.save_metadata(train_df, "train_metadata.csv", use_phonemes=True)
        self.save_metadata(val_df, "val_metadata.csv", use_phonemes=True)
        self.save_metadata(test_df, "test_metadata.csv", use_phonemes=True)
        
        # Also save with original text for comparison
        self.save_metadata(train_df, "train_metadata_text.csv", use_phonemes=False)
        self.save_metadata(val_df, "val_metadata_text.csv", use_phonemes=False)
        
        # Create character set from text
        char_set = self.create_character_set(result_df)
        char_set_path = self.output_dir / "character_set.txt"
        with open(char_set_path, 'w', encoding='utf-8') as f:
            f.write(char_set)
        
        # Save processing statistics
        stats = {
            'total_samples': len(result_df),
            'train_samples': len(train_df),
            'val_samples': len(val_df),
            'test_samples': len(test_df),
            'unique_characters': len(char_set),
            'character_set': char_set,
            'average_text_length': result_df['text'].str.len().mean() if len(result_df) > 0 else 0,
            'phonemization_mode': 'raw_text'
        }
        
        stats_path = self.output_dir / "preprocessing_stats.json"
        import json
        with open(stats_path, 'w', encoding='utf-8') as f:
            json.dump(stats, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Raw text preprocessing completed. Statistics saved to {stats_path}")
        return stats


def main():
    """Main preprocessing script."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Preprocess Uzbek TTS dataset")
    parser.add_argument("--data_dir", default="./data/processed", help="Input data directory")
    parser.add_argument("--output_dir", default="./data/preprocessed", help="Output directory")
    parser.add_argument("--metadata_file", default="metadata.csv", help="Metadata file name")
    parser.add_argument("--test_phonemization", action="store_true", help="Test phonemization on sample texts")
    parser.add_argument("--skip_phonemization", action="store_true", help="Skip phonemization and use raw text")
    parser.add_argument("--use_fallback", action="store_true", help="Use fallback phonemization instead of espeak-ng")
    parser.add_argument("--debug", action="store_true", help="Show debug information about text processing")
    
    args = parser.parse_args()
    
    # Test phonemization if requested
    if args.test_phonemization:
        print("Testing phonemization...")
        processor = UzbekTextProcessor(use_espeak=not args.use_fallback)
        test_texts = [
            "Salom, qandaysiz?",
            "Bugun ob-havo juda yaxshi.",
            "Men Toshkentda yashayman.",
            "Uzbek tilida gapirishni o'rganyapman.",
        ]
        
        for text in test_texts:
            phonemes = processor.phonemize_text(text)
            print(f"Text: {text}")
            print(f"Phonemes: {phonemes}")
            print("-" * 50)
        return
    
    # Initialize preprocessor
    preprocessor = DatasetPreprocessor(args.data_dir, args.output_dir)
    
    # Set debug flag
    if args.debug:
        preprocessor.debug = True
    
    # Process dataset
    if args.skip_phonemization:
        print("Skipping phonemization - using raw text for training")
        stats = preprocessor.process_dataset_raw_text(args.metadata_file)
    else:
        stats = preprocessor.process_dataset(args.metadata_file)
    
    print("\n" + "="*50)
    print("PREPROCESSING COMPLETED")
    print("="*50)
    print(f"Total samples: {stats['total_samples']}")
    print(f"Train samples: {stats['train_samples']}")
    print(f"Validation samples: {stats['val_samples']}")
    print(f"Test samples: {stats['test_samples']}")
    print(f"Unique characters: {stats['unique_characters']}")
    if 'average_phoneme_length' in stats:
        print(f"Average phoneme length: {stats['average_phoneme_length']:.1f}")
    print(f"Average text length: {stats['average_text_length']:.1f}")
    print("="*50)


if __name__ == "__main__":
    main()
