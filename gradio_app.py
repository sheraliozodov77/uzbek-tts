#!/usr/bin/env python3
"""
Beautiful Gradio web interface for Uzbek TTS model.
Perfect for portfolio showcase!
"""

import os
import sys
import logging
import time
import torch
import numpy as np
from pathlib import Path
import warnings
import gradio as gr

# Suppress warnings
warnings.filterwarnings("ignore")

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global variables for model
model = None
ap = None
config = None
device = "cuda" if torch.cuda.is_available() else "cpu"

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

def load_model():
    """Load the trained TTS model."""
    global model, ap, config
    
    if model is not None:
        logger.info("Model already loaded")
        return True
    
    try:
        from TTS.tts.configs.vits_config import VitsConfig
        from TTS.tts.models.vits import Vits
        from TTS.utils.audio import AudioProcessor
        from TTS.tts.utils.text.characters import IPAPhonemes
        from TTS.tts.utils.text.tokenizer import TTSTokenizer
        
        logger.info("🔍 Finding best model...")
        model_path, config_path = find_best_model()
        
        if model_path is None or config_path is None:
            logger.error("❌ Model not found!")
            return False
        
        logger.info(f"📦 Loading model from: {model_path}")
        logger.info(f"📋 Loading config from: {config_path}")
        
        # Load configuration
        config = VitsConfig()
        config.load_json(str(config_path))
        
        # Initialize model
        model = Vits(config)
        model.load_checkpoint(config, str(model_path))
        
        if device == "cuda":
            model.cuda()
        else:
            model.cpu()
        
        model.eval()
        
        # Initialize audio processor
        ap = AudioProcessor(**config.audio)
        
        # Setup tokenizer (Latin Uzbek only)
        uzbek_chars = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
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
        
        logger.info("✅ Model loaded successfully!")
        return True
        
    except Exception as e:
        logger.error(f"❌ Failed to load model: {e}")
        import traceback
        traceback.print_exc()
        return False

def generate_speech(text: str, progress=gr.Progress()):
    """Generate speech from text."""
    global model, ap, config
    
    if model is None:
        return None, "❌ Model not loaded! Please wait for initialization or check logs."
    
    if not text or not text.strip():
        return None, "⚠️ Please enter some text to generate speech."
    
    try:
        progress(0.1, desc="Tokenizing text...")
        
        # Tokenize text
        token_ids = model.tokenizer.text_to_ids(text.strip())
        token_ids = torch.LongTensor(token_ids).unsqueeze(0)
        
        if device == "cuda":
            token_ids = token_ids.cuda()
        
        progress(0.5, desc="Generating audio...")
        
        # Generate audio
        start_time = time.time()
        with torch.no_grad():
            outputs = model.inference(token_ids)
            audio = outputs['model_outputs'].squeeze()
            
            if isinstance(audio, torch.Tensor):
                audio = audio.cpu().numpy()
        
        inference_time = time.time() - start_time
        audio_duration = len(audio) / config.audio['sample_rate']
        rtf = inference_time / audio_duration if audio_duration > 0 else 0
        
        progress(0.9, desc="Processing audio...")
        
        progress(1.0, desc="Complete!")
        
        # Create info message
        speed_multiplier = int(1/rtf) if rtf > 0 and rtf < 1 else 0
        speed_text = f"⚡ {speed_multiplier}x faster" if speed_multiplier > 0 else "N/A"
        
        info = f"""
        ✅ **Audio Generated Successfully!**
        
        📊 **Performance Metrics:**
        - ⏱️ **Inference Time**: {inference_time:.3f} seconds
        - 🎵 **Audio Duration**: {audio_duration:.2f} seconds
        - 🚀 **Real-Time Factor (RTF)**: {rtf:.3f} ({speed_text})
        - 🔊 **Sample Rate**: {config.audio['sample_rate']} Hz
        """
        
        # Return audio as (sample_rate, audio_array) tuple for Gradio
        # Gradio expects audio as numpy array with values in [-1, 1] range
        return (config.audio['sample_rate'], audio), info
        
    except Exception as e:
        logger.error(f"❌ Generation failed: {e}")
        import traceback
        traceback.print_exc()
        return None, f"❌ Error generating speech: {str(e)}"

# Example sentences
EXAMPLE_SENTENCES = [
    "Salom, qandaysiz? Bugun ob-havo juda yaxshi.",
    "Men Toshkentda yashayman. Uzbek tilida gapirishni o'rganyapman.",
    "Rahmat, sizga ham! Yaxshi kunlar tilayman.",
    "Bu juda chiroyli shahar. Men bu yerda do'stlarim bilan birga yashayman.",
    "Uzbek tilida gapirish oson emas, lekin men har kuni mashq qilaman.",
    "Bugun maktabda yangi dars o'qidim. O'qituvchi juda yaxshi tushuntirdi.",
    "Kechki paytda do'stlarim bilan bog'da yurishga boramiz.",
    "Ovqat tayyorlashni juda yaxshi ko'raman. Milliy taomlarimizni tayyorlashni o'rganyapman.",
    "Kitob o'qish meni juda qiziqtiradi. Har hafta yangi kitob o'qiyman.",
    "Uzbekiston - bu go'zal mamlakat. Bu yerda ko'p tarixiy joylar bor."
]

def get_model_info():
    """Get model information for display."""
    if model is None:
        return "Model not loaded yet..."
    
    model_path, config_path = find_best_model()
    
    info = f"""
    ### Model Information
    
    **Architecture**: VITS (Variational Inference with adversarial learning for end-to-end Text-to-Speech)
    
    **Training**
    - Status: Production Ready
    - Training Time: 25.45 hours
    - Total Epochs: 200
    - Total Steps: ~130,000+
    
    **Performance**
    - Real-Time Factor: 0.019 (53x faster than real-time)
    - Average Inference: 0.110 seconds
    - Sample Rate: {config.audio['sample_rate']} Hz
    
    **Hardware**
    - GPU: RTX 4090 (24GB VRAM)
    - Memory Usage: 23.4/24GB (optimal utilization)
    
    **Supported Scripts**
    - Latin Uzbek
    
    **Model Location**: `{model_path.parent.name}`
    """
    
    return info

# Load model on startup
logger.info("🚀 Starting Uzbek TTS Gradio App...")
load_model()

# Custom CSS for beautiful UI with #fcd86a as main color
custom_css = """
    .gradio-container {
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    .main-header {
        text-align: center;
        background: #fcd86a;
        color: #2c3e50;
        padding: 20px;
        border-radius: 10px;
        margin-bottom: 20px;
    }
    .info-box {
        background: #fff9e6;
        padding: 15px;
        border-radius: 8px;
        margin: 10px 0;
    }
    .example-btn {
        margin: 5px;
    }
    button.primary {
        background: linear-gradient(135deg, #fcd86a 0%, #f5c842 100%) !important;
        color: #2c3e50 !important;
        border: none !important;
        font-weight: 600 !important;
    }
    button.primary:hover {
        background: linear-gradient(135deg, #f5c842 0%, #e8b830 100%) !important;
        transform: translateY(-1px);
        box-shadow: 0 4px 8px rgba(252, 216, 106, 0.3) !important;
    }
    button.secondary {
        border: 2px solid #fcd86a !important;
        color: #2c3e50 !important;
    }
    button.secondary:hover {
        background: #fcd86a !important;
        color: #2c3e50 !important;
    }
    .gr-box {
        border-color: #fcd86a !important;
    }
    .gr-input:focus {
        border-color: #fcd86a !important;
        box-shadow: 0 0 0 2px rgba(252, 216, 106, 0.2) !important;
    }
"""

# Create Gradio interface
with gr.Blocks(css=custom_css, theme=gr.themes.Soft()) as demo:
    
    # Header
    gr.Markdown(
        """
        # Uzbek Text-to-Speech (TTS) Model
        ### Neural Speech Synthesis for Uzbek Language
        
        This is a production-ready VITS-based TTS model trained on RTX 4090 GPU. 
        Enter your Uzbek text below and generate natural-sounding speech!
        """,
        elem_classes=["main-header"]
    )
    
    with gr.Row():
        with gr.Column(scale=2):
            # Text input
            text_input = gr.Textbox(
                label="📝 Enter Uzbek Text (Latin)",
                placeholder="Salom, qandaysiz? Uzbek tilida gapirishni o'rganyapman...",
                lines=5,
                value=EXAMPLE_SENTENCES[0] if EXAMPLE_SENTENCES else ""
            )
            
            # Example buttons
            gr.Markdown("**💡 Example Sentences:**")
            with gr.Row():
                for i, example in enumerate(EXAMPLE_SENTENCES[:5]):  # Show first 5 examples
                    btn = gr.Button(f"Example {i+1}", variant="secondary", size="sm", elem_classes=["example-btn"])
                    # Use lambda with default argument to capture example value
                    btn.click(
                        fn=lambda ex=example: ex,
                        outputs=text_input
                    )
            
            with gr.Row():
                generate_btn = gr.Button("🎵 Generate Speech", variant="primary", size="lg")
                clear_btn = gr.Button("🔄 Clear", variant="secondary")
            
            # Audio output
            audio_output = gr.Audio(
                label="🎧 Generated Audio",
                type="numpy",
                format="wav"
            )
            
            # Info output
            info_output = gr.Markdown(label="📊 Generation Info")
        
        with gr.Column(scale=1):
            # Model info
            model_info = gr.Markdown(get_model_info())
    
    # Event handlers
    generate_btn.click(
        fn=generate_speech,
        inputs=[text_input],
        outputs=[audio_output, info_output]
    )
    
    clear_btn.click(
        fn=lambda: ("", None, ""),
        outputs=[text_input, audio_output, info_output]
    )
    
    # Allow Enter key to generate
    text_input.submit(
        fn=generate_speech,
        inputs=[text_input],
        outputs=[audio_output, info_output]
    )
    
    # Footer
    gr.Markdown(
        """
        ### Project Links
        - **GitHub**: Check the repository for full documentation
        - **Training**: 200 epochs, ~130,000+ steps
        - **Performance**: RTF 0.019, 53x faster than real-time
        
        **Built with**: Gradio, TTS, PyTorch, VITS Architecture
        
        **Developer**: Sherali Ozodov
        """,
        elem_classes=["info-box"]
    )

if __name__ == "__main__":
    # Launch the interface
    demo.launch(
        server_name="0.0.0.0",  # Allow external access
        server_port=7860,        # Default Gradio port
        share=True,             # Set to True for public link
        show_error=True,
        quiet=False
    )

