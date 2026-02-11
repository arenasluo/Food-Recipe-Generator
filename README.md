---
title: Food to Recipe Generator
emoji: 🍽️
colorFrom: blue
colorTo: red
sdk: gradio
sdk_version: 5.0.0
app_file: app.py
pinned: false
license: mit
---

# Food to Recipe Generation: Vision-Language Models for Recipe Generation

A deep learning system that generates cooking recipes from food images using state-of-the-art vision-language models. This project implements and compares two approaches: (1) CLIP + GPT-2 Transformer, and (2) Qwen2.5-VL-3B-Instruct fine-tuning.

![Project Banner](https://img.shields.io/badge/Python-3.8+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

## 🚀 **Try the Live Demo!**

**👉 [Launch Web App](https://huggingface.co/spaces/arenasluo/food-recipe-generator) 👈**

Upload a food image and get an AI-generated recipe instantly!

## Overview

This project implements cross-modal translation models that:
- Take images of food as input
- Generate complete recipes with titles, ingredients, and step-by-step cooking instructions
- Use visual information (ingredients, cooking method, presentation) to create structured recipe text
- Evaluate recipe quality using multiple metrics (ROUGE-L, BLEU, FIRE-style component metrics)

## Two Approaches Implemented

### Approach 1: CLIP + GPT-2 Transformer (`Food_to_receipe.ipynb`)

**Architecture:**
- **Vision Encoder**: OpenAI CLIP (frozen) to extract visual features from food images
- **Text Decoder**: GPT-2 Transformer fine-tuned on recipe text
- **Training**: Only the GPT-2 decoder is trained; CLIP encoder remains frozen

**Key Features:**
- Modular design separating vision and language components
- Efficient training (only decoder parameters updated)
- Good baseline for comparison

### Approach 2: Qwen2.5-VL-3B-Instruct (`Food_to_receipe_Qwen.ipynb`) ⭐ **Recommended**

**Architecture:**
- **Model**: Qwen2.5-VL-3B-Instruct (3.75B parameters)
- **Type**: Native vision-language model that handles both image understanding and text generation
- **Training**: End-to-end fine-tuning with LoRA (Low-Rank Adaptation) for efficiency

**Key Features:**
- **Native Vision-Language Model**: Unified architecture for better image-text alignment
- **Instruction-Tuned**: Better at following prompts and generating structured recipes
- **Memory Efficient**: LoRA fine-tuning reduces memory requirements
- **Multi-GPU Support**: Optimized for training with DeepSpeed ZeRO Stage 3
- **Better Performance**: Achieves higher scores on evaluation metrics

## Features

- **Food Recognition**: Automatically identifies food items and visual characteristics
- **Recipe Generation**: Generates detailed recipes with:
  - Recipe titles
  - Complete ingredient lists with amounts
  - Step-by-step cooking instructions
- **Comprehensive Evaluation**: 
  - ROUGE-L and BLEU scores for overall quality
  - FIRE-style component metrics (Title LCS, Ingredient Precision/Recall/F1, Instruction ROUGE-L/BLEU)
  - Training/validation loss tracking
- **Web Interface**: Interactive Gradio-based web UI for easy recipe generation
- **Multi-GPU Support**: Efficient training on multiple GPUs with DeepSpeed
- **Memory Optimization**: Gradient checkpointing, mixed precision (BF16/FP16), and CPU offloading

## Installation

### Prerequisites

- Python 3.8 or higher
- CUDA-capable GPU (recommended, 16GB+ VRAM)
- At least 16GB RAM

### Setup

1. Clone the repository:
```bash
git clone https://github.gatech.edu/fgao70/DL_group_project.git
cd DL_group_project
git checkout wluo79
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

### Required Packages

- PyTorch 2.0+
- Transformers (Hugging Face)
- CLIP (OpenAI) - for Approach 1
- Qwen-VL - for Approach 2
- Gradio
- Pillow
- Pandas
- NumPy
- Matplotlib
- Seaborn
- rouge-score
- sacrebleu
- accelerate
- deepspeed
- peft (for LoRA)

## Dataset

The project uses the **Food Ingredients and Recipe Dataset with Image Name Mapping** which contains:
- **13,501 recipes** with titles, ingredients, and instructions
- **13,471 successfully matched food images**
- Recipe format: Structured text with title, ingredients list, and step-by-step instructions

### Dataset Structure
```
CLIP+Transformer/
├── Food Ingredients and Recipe Dataset with Image Name Mapping.csv
└── food_images/
    └── Food Images/
        ├── image1.jpg
        ├── image2.jpg
        └── ...
```

## Usage

### 🌐 Try the Live Demo (No Setup Required!)

**The easiest way to try the model:**

👉 **[https://huggingface.co/spaces/arenasluo/food-recipe-generator](https://huggingface.co/spaces/arenasluo/food-recipe-generator)**

1. Click the link above
2. Upload a food image
3. Click "Generate Recipe"
4. Get your AI-generated recipe!

No installation, no setup - just try it in your browser! 🚀

---

### Training the Models

#### Approach 1: CLIP + GPT-2

Open and run the Jupyter notebook:
```bash
jupyter notebook Food_to_receipe.ipynb
```

#### Approach 2: Qwen2.5-VL-3B (Recommended)

Open and run the Jupyter notebook:
```bash
jupyter notebook Food_to_receipe_Qwen.ipynb
```

**Training Steps:**
1. Load and preprocess the dataset
2. Initialize the model (CLIP+GPT2 or Qwen2.5-VL)
3. Fine-tune on recipe dataset
4. Monitor training/validation loss
5. Evaluate on test set with comprehensive metrics
6. Save model checkpoint

### Using the Web Interface

**Jupyter Notebook:**
1. Run the notebook cells to load the model
2. Run the Gradio interface cell (Section 16 in Qwen notebook)
3. Launch the web server:
```python
demo.launch(server_port=7860)
```
4. Upload a food image through the web interface
5. Click "Generate Recipe" to get the recipe

### Output Format

```
RECIPE: GRILLED CHICKEN

============================================================

INGREDIENTS:

  1. Chicken breast
  2. Olive oil
  3. Salt and pepper
  ...

============================================================

INSTRUCTIONS:

Step 1: Preheat grill to medium-high heat.

Step 2: Season chicken with salt and pepper.

Step 3: Grill chicken for 6-8 minutes per side.

============================================================
NOTE: This recipe was generated by an AI model and may require
refinement. Please verify ingredients and instructions before cooking.
```

## Training Details

### Qwen2.5-VL-3B Fine-Tuning (Approach 2)

**Hyperparameters:**
- **Batch Size**: 1 per GPU (with gradient accumulation of 128 for effective batch size of 128)
- **Learning Rate**: 5e-5
- **Epochs**: 5
- **Warmup Steps**: 100
- **Optimizer**: AdamW
- **Scheduler**: Linear warmup with decay
- **Mixed Precision**: BF16 (if supported) or FP16
- **DeepSpeed**: ZeRO Stage 3 with CPU offloading for memory efficiency
- **LoRA**: Low-rank adaptation for efficient fine-tuning

**Training Performance:**
- **Dataset Size**: 500 training samples (for memory efficiency)
- **Train/Test Split**: 80/20
- **Final Train Loss**: ~2.74
- **Final Validation Loss**: ~2.89

### CLIP + GPT-2 (Approach 1)

**Hyperparameters:**
- **Batch Size**: 4
- **Learning Rate**: 5e-5
- **Epochs**: 5
- **Optimizer**: AdamW
- **Frozen Components**: CLIP encoder (only GPT-2 decoder trained)

## Evaluation Metrics

### Overall Metrics
- **ROUGE-L**: Measures semantic similarity between generated and ground truth recipes
- **BLEU**: Measures n-gram overlap for exact word matching

### FIRE-Style Component Metrics
- **Title LCS**: Longest Common Subsequence for title evaluation
- **Ingredient Precision/Recall/F1**: Set-based metrics for ingredient extraction
- **Instruction ROUGE-L/BLEU**: Metrics specifically for cooking instructions

See `METRICS_COMPARISON_WITH_FIRE.md` for detailed comparison with the FIRE paper.

## Results

### Quantitative Results

**Qwen2.5-VL-3B Approach:**
- Better overall performance on evaluation metrics
- Higher ROUGE-L and BLEU scores
- Better ingredient identification (Ingredient F1)
- More coherent recipe generation

**CLIP + GPT-2 Approach:**
- Good baseline performance
- Efficient training (only decoder parameters)
- Modular architecture allows for component swapping

### Qualitative Results

Both models successfully:
- Learn to associate visual features with recipe text
- Generate structured recipes with ingredients and instructions
- Identify food categories with reasonable accuracy
- Handle diverse food types and cooking styles

## Project Structure

```
CLIP+Transformer/
├── Food_to_receipe.ipynb              # CLIP + GPT-2 approach
├── Food_to_receipe_Qwen.ipynb          # Qwen2.5-VL-3B approach (recommended)
├── Food_to_receipe_Deep_seek_Janus_1B.ipynb  # Additional experiments
├── Food_to_receipe_Minimax_M1.ipynb    # Additional experiments
├── ds_config.json                      # DeepSpeed configuration
├── requirements.txt                    # Python dependencies
├── README.md                           # This file
├── PROJECT_SUMMARY.md                  # Plain-language project summary
├── IMPLEMENTATION_DETAILS.md          # Technical implementation details
├── EVALUATION_AND_RESULTS.md          # Evaluation methodology and results
├── METRICS_COMPARISON_WITH_FIRE.md    # Comparison with FIRE paper
├── food_images/                        # Food images directory
│   └── Food Images/
├── qwen_recipe_model_final/           # Fine-tuned Qwen model
├── janus_recipe_model_final/          # Fine-tuned Janus model
└── Food Ingredients and Recipe Dataset with Image Name Mapping.csv
```

## Limitations

- Generated recipes may need refinement for actual cooking
- Model trained on limited dataset (500-1000 samples for memory efficiency)
- Some repetition in generated text (mitigated by post-processing)
- Quality depends on image clarity and food visibility
- Evaluation metrics may not capture all aspects of recipe quality

## Future Improvements

1. Train on full dataset (all 13,501 recipes)
2. Experiment with larger models (Qwen2.5-VL-7B, Qwen2.5-VL-14B)
3. Implement beam search for better generation quality
4. Add more sophisticated evaluation metrics
5. Fine-tune vision encoder for better food understanding
6. Add data augmentation for robustness
7. Implement recipe validation and refinement steps

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- **Qwen Team** for Qwen2.5-VL models
- **OpenAI** for CLIP vision-language model
- **Hugging Face** for Transformers library and infrastructure
- **DeepSpeed Team** for memory-efficient training
- **Gradio** for the web interface
- Dataset creators for the food recipes and images

## Citation

If you use this project in your research, please cite:

```bibtex
@misc{food-recipe-generator,
  author = {Weijun Luo},
  title = {Food to Recipe Generator using Vision-Language Models},
  year = {2025},
  publisher = {GitHub},
  url = {https://github.com/arenasluo/Food-Recipe-Generator}
}
```

## Contact

For questions or feedback, please open an issue on GitHub.

---

**Note**: This is an academic/research project. Generated recipes should be verified before cooking.
