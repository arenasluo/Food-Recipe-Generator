---
title: Food to Recipe Generator
emoji: 🍽️
colorFrom: yellow
colorTo: red
sdk: gradio
sdk_version: "4.0.0"
app_file: app.py
pinned: false
license: mit
---

# 🍽️ Food to Recipe Generator

Generate detailed cooking recipes from food images using Qwen2.5-VL vision-language model!

## Features

- Upload a photo of any food dish
- Get AI-generated recipes with:
  - Recipe title
  - Ingredients list with measurements
  - Step-by-step cooking instructions
  - Cooking times and temperatures

## Model

This Space uses **Qwen2.5-VL-3B-Instruct** fine-tuned with LoRA on the Recipe1M+ dataset.

- **Base Model**: [Qwen/Qwen2.5-VL-3B-Instruct](https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct)
- **Fine-tuned LoRA**: [arenasluo/qwen-recipe-lora](https://huggingface.co/arenasluo/qwen-recipe-lora)
- **Training Dataset**: Recipe1M+ (food images and recipes)

## Usage

1. Upload a clear photo of food
2. Click "Generate Recipe" or wait for auto-generation
3. Review the AI-generated recipe

**Note**: AI-generated recipes should be verified before cooking. Always check measurements, cooking times, and food safety guidelines.

## Acknowledgments

- Qwen2.5-VL model by Alibaba Cloud
- Recipe1M+ dataset
- Built with Gradio and Hugging Face Transformers
