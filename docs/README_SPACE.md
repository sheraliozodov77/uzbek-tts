---
title: Uzbek Text-to-Speech
emoji: 🎤
colorFrom: yellow
colorTo: orange
sdk: gradio
sdk_version: 4.0.0
app_file: app.py
pinned: false
license: mit
---

# Uzbek Text-to-Speech (TTS) Model

A production-ready VITS-based Text-to-Speech system for the Uzbek language, trained on RTX 4090 GPU with 67+ hours of Common Voice data.

## Features

- 🚀 **Real-time Inference**: 53x faster than real-time (RTF: 0.019)
- ✅ **Production Ready**: Fully trained and optimized
- 🎯 **High Quality**: Natural-sounding Uzbek speech synthesis
- 📝 **Latin Uzbek**: Supports Latin script Uzbek text

## Model Details

- **Architecture**: VITS (Variational Inference with adversarial learning for end-to-end Text-to-Speech)
- **Training**: 200 epochs, ~130,000+ steps, 25.45 hours
- **Performance**: RTF 0.019, Average inference: 0.110 seconds
- **Sample Rate**: 22.05kHz
- **Hardware**: Trained on RTX 4090 (24GB VRAM)

## Usage

Simply enter your Uzbek text (Latin script) and click "Generate Speech" to hear the synthesized audio.

## Example Sentences

Try these example sentences to get started:
- "Salom, qandaysiz? Bugun ob-havo juda yaxshi."
- "Men Toshkentda yashayman. Uzbek tilida gapirishni o'rganyapman."
- "Rahmat, sizga ham! Yaxshi kunlar tilayman."

## Technical Details

Built with:
- Gradio for the web interface
- Coqui TTS framework
- PyTorch
- VITS architecture

## License

MIT License

