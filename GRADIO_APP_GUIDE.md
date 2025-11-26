# Gradio Web Interface Guide

## Quick Start

### Using Qwen2.5-VL Model (Recommended)

The new Gradio app uses the fine-tuned Qwen2.5-VL model for better recipe generation.

1. **Install dependencies** (if not already installed):
```bash
pip install -r requirements.txt
```

2. **Run the Gradio app**:
```bash
python app_qwen.py
```

3. **Open in browser**:
   - The app will be available at `http://localhost:7860`
   - The URL will be printed in the terminal

### Features

- **Image Upload**: Drag and drop or click to upload food images
- **Advanced Settings**: Adjust generation parameters (temperature, top-p, max tokens)
- **Example Images**: Try pre-loaded example images from your dataset
- **Auto-generation**: Recipes generate automatically when you upload an image

### Model Loading

The app will:
1. First try to load your fine-tuned model from `./qwen_recipe_model_final/`
2. If not found, fall back to the base `Qwen2.5-VL-3B-Instruct` model
3. Automatically use GPU if available, otherwise uses CPU

### Troubleshooting

- **Out of Memory**: Reduce `max_tokens` in the advanced settings
- **Model not loading**: Make sure the model path exists and is complete
- **Import errors**: Install missing packages with `pip install qwen-vl-utils gradio`

### Old CLIP+GPT-2 App

If you want to use the original CLIP+GPT-2 model, run:
```bash
python app.py
```


