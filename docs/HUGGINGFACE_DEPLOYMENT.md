# Deploying to Hugging Face Spaces

This guide will help you deploy your Uzbek TTS model to Hugging Face Spaces for permanent hosting.

## Prerequisites

1. **Hugging Face Account**: Sign up at https://huggingface.co
2. **Git LFS**: For uploading large model files (best_model.pth is ~200-500MB)
3. **Git**: For version control

## Step 1: Create a New Space

1. Go to https://huggingface.co/spaces
2. Click "Create new Space"
3. Fill in the details:
   - **Space name**: `uzbek-tts` (or your preferred name)
   - **SDK**: Select "Gradio"
   - **Visibility**: Public or Private
   - Click "Create Space"

## Step 2: Upload Model Files

You have two options for uploading model files:

### Option A: Using Git LFS (Recommended for Large Files)

```bash
# Install Git LFS if not already installed
git lfs install

# Clone your newly created Space
git clone https://huggingface.co/spaces/YOUR_USERNAME/uzbek-tts
cd uzbek-tts

# Enable Git LFS for model files
git lfs track "*.pth"
git lfs track "*.pth.tar"

# Copy model files from your local project
mkdir -p output/uzbek_tts_model
cp ../uzbek-tts/output/uzbek_tts_*/best_model.pth output/uzbek_tts_model/
cp ../uzbek-tts/output/uzbek_tts_*/config.json output/uzbek_tts_model/

# Or use a simpler structure (recommended):
cp ../uzbek-tts/output/uzbek_tts_*/best_model.pth .
cp ../uzbek-tts/output/uzbek_tts_*/config.json .

# Commit and push
git add .
git commit -m "Add model files and app"
git push
```

### Option B: Using Hugging Face Web Interface

1. Go to your Space page
2. Click "Files" tab
3. Click "Add file" → "Upload file"
4. Upload `best_model.pth` and `config.json` directly
   - Note: Large files (>10MB) should use Git LFS

## Step 3: Upload Application Files

Copy these files to your Space:

### Required Files:

1. **`app.py`** - Main Gradio application (already created for you)
2. **`requirements.txt`** - Python dependencies (use `requirements_space.txt`)
3. **`README.md`** - Space description (use `README_SPACE.md` as template)

### File Structure in Space:

```
uzbek-tts/
├── app.py                    # Main Gradio app
├── requirements.txt          # Dependencies
├── README.md                 # Space description
├── best_model.pth           # Model weights (large file, use Git LFS)
└── config.json              # Model configuration
```

## Step 4: Update app.py for HF Spaces

The `app.py` file I created already includes:
- ✅ Multiple path detection for model files
- ✅ Automatic model loading on startup
- ✅ HF Spaces compatible launch code

However, make sure your `app.py` has this structure:

```python
# At the end of app.py
if __name__ == "__main__":
    # For local development
    demo.launch(...)
else:
    # For HF Spaces (automatic)
    demo.launch()
```

## Step 5: Update Model Path Detection

The `app.py` I created already checks multiple paths:
- `output/` (local development)
- `.` (root directory - HF Spaces)
- `models/` (alternative HF Spaces location)

Make sure your model files are accessible at one of these paths.

## Step 6: Test Locally First

Before deploying, test that everything works:

```bash
# Install dependencies
pip install -r requirements_space.txt

# Run the app
python app.py
```

## Step 7: Deploy to Hugging Face

### Using Git:

```bash
cd uzbek-tts-space  # Your cloned Space directory

# Copy files
cp ../app.py .
cp ../requirements_space.txt requirements.txt
cp ../README_SPACE.md README.md

# Copy model files
cp ../output/uzbek_tts_*/best_model.pth .
cp ../output/uzbek_tts_*/config.json .

# Commit and push
git add .
git commit -m "Initial deployment"
git push
```

### Using Web Interface:

1. Go to your Space → Files tab
2. Upload each file:
   - `app.py`
   - `requirements.txt`
   - `README.md`
   - `best_model.pth` (use Git LFS for large files)
   - `config.json`

## Step 8: Configure Space Settings

1. Go to your Space → Settings
2. Configure:
   - **Hardware**: Select "CPU" or "GPU" (GPU recommended for faster inference)
   - **Sleep time**: Set to prevent Space from sleeping
   - **Environment variables**: Add any needed vars

## Step 9: Monitor Deployment

1. Go to your Space page
2. Check the "Logs" tab for any errors
3. Wait for build to complete (usually 5-10 minutes)
4. Your Space will be available at: `https://huggingface.co/spaces/YOUR_USERNAME/uzbek-tts`

## Troubleshooting

### Issue: Model not found
**Solution**: Make sure `best_model.pth` and `config.json` are in the root directory or update paths in `app.py`

### Issue: Build fails due to dependencies
**Solution**: Check `requirements.txt` - remove Windows-specific packages if any

### Issue: Model file too large
**Solution**: Use Git LFS: `git lfs track "*.pth"`

### Issue: CUDA/GPU errors on HF Spaces
**Solution**: The app automatically falls back to CPU if CUDA is unavailable

### Issue: Slow inference
**Solution**: Upgrade to GPU hardware in Space settings (may require upgraded account)

## Alternative: Using Hugging Face Model Hub

If your model files are very large, consider:

1. Upload model to Hugging Face Model Hub separately
2. Update `app.py` to download model on first run:

```python
from huggingface_hub import hf_hub_download

def download_model():
    model_path = hf_hub_download(
        repo_id="YOUR_USERNAME/uzbek-tts-model",
        filename="best_model.pth",
        cache_dir="models"
    )
    config_path = hf_hub_download(
        repo_id="YOUR_USERNAME/uzbek-tts-model",
        filename="config.json",
        cache_dir="models"
    )
    return model_path, config_path
```

## Next Steps

Once deployed:
1. ✅ Share your Space URL
2. ✅ Add to your portfolio
3. ✅ Monitor usage and performance
4. ✅ Update model versions as needed

## Quick Reference

- **Space URL**: `https://huggingface.co/spaces/YOUR_USERNAME/uzbek-tts`
- **Model files needed**: `best_model.pth`, `config.json`
- **Main app file**: `app.py`
- **Dependencies**: `requirements.txt`
- **Documentation**: `README.md`

Good luck with your deployment! 🚀

