import os
import sys
import warnings
import torch
from PIL import Image
import gradio as gr

# Suppress warnings
warnings.filterwarnings("ignore")

# Hugging Face Libraries
from transformers import (
    Qwen2_5_VLForConditionalGeneration,
    AutoProcessor
)
from qwen_vl_utils import process_vision_info

try:
    from peft import PeftModel
    PEFT_AVAILABLE = True
except ImportError:
    PEFT_AVAILABLE = False

try:
    from huggingface_hub import snapshot_download
    HF_HUB_AVAILABLE = True
except ImportError:
    HF_HUB_AVAILABLE = False

# Global variables for lazy loading
model = None
processor = None
model_loaded_with_lora = False

# Model configuration
BASE_MODEL_NAME = "Qwen/Qwen2.5-VL-3B-Instruct"
LORA_ADAPTER_REPO = "arenasluo/qwen-recipe-lora"

def load_model():
    """Lazy load the model only when first needed."""
    global model, processor, model_loaded_with_lora

    if model is not None:
        return  # Already loaded

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print("Loading Qwen2.5-VL model...")

    # Try to load fine-tuned LoRA adapter
    if HF_HUB_AVAILABLE and PEFT_AVAILABLE:
        try:
            print(f"Attempting to load LoRA adapter from {LORA_ADAPTER_REPO}...")
            adapter_path = snapshot_download(
                repo_id=LORA_ADAPTER_REPO,
                allow_patterns=["adapter_*", "*.json"]
            )

            base_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                BASE_MODEL_NAME,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                device_map="auto" if torch.cuda.is_available() else None
            )

            model = PeftModel.from_pretrained(base_model, adapter_path)
            model = model.merge_and_unload()
            model_loaded_with_lora = True
            print("✓ Model loaded with fine-tuned LoRA adapter")
        except Exception as e:
            print(f"Could not load LoRA adapter: {e}")
            print("Falling back to base model...")
            model = None

    # Fallback to base model
    if model is None:
        print("Loading base Qwen2.5-VL model...")
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            BASE_MODEL_NAME,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None
        )
        print("✓ Base model loaded")

    # Load processor
    processor = AutoProcessor.from_pretrained(BASE_MODEL_NAME)
    model.eval()
    print(f"✓ Model ready! (LoRA fine-tuned: {model_loaded_with_lora})")

def generate_recipe_from_image(image, max_new_tokens=2048):
    """Generate a recipe from a food image using Qwen2.5-VL."""
    try:
        # Lazy load model on first use
        load_model()

        if image is None:
            return "⚠️ Please upload an image."

        # Convert to PIL Image if needed
        if isinstance(image, str):
            image = Image.open(image)

        # Ensure RGB format
        if image.mode != 'RGB':
            image = image.convert('RGB')

        # Create prompt for recipe generation
        prompt_text = (
            "Generate a detailed cooking recipe from this food image. "
            "Include:\n"
            "- Recipe title\n"
            "- Ingredients list with measurements\n"
            "- Step-by-step cooking instructions\n"
            "- Cooking time and temperature if applicable"
        )

        # Format message for Qwen2.5-VL
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt_text}
                ]
            }
        ]

        # Process inputs
        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt"
        )

        # Move to device
        device = next(model.parameters()).device
        inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                  for k, v in inputs.items()}

        # Set pad token
        pad_token_id = processor.tokenizer.pad_token_id or processor.tokenizer.eos_token_id
        eos_token_id = processor.tokenizer.eos_token_id

        # Generate recipe
        with torch.no_grad():
            generated_ids = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                pad_token_id=pad_token_id,
                eos_token_id=eos_token_id,
                repetition_penalty=1.1,
            )

        # Decode only the newly generated tokens
        input_length = inputs['input_ids'].shape[1]
        generated_tokens = generated_ids[0][input_length:]
        recipe = processor.decode(generated_tokens, skip_special_tokens=True)

        # Format output
        output = "# 🍽️ AI-Generated Recipe\n\n"
        output += "=" * 60 + "\n\n"
        output += recipe
        output += "\n\n" + "=" * 60 + "\n"
        output += "\n**Note:** This recipe was generated by an AI model. "
        output += "Please verify ingredients and instructions before cooking.\n"

        if not model_loaded_with_lora:
            output += "\n*Using base model without fine-tuning. Results may vary in quality.*"

        return output

    except Exception as e:
        error_msg = f"❌ Error generating recipe: {str(e)}\n\n"
        error_msg += "Please try again with a different image."
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return error_msg

# Create Gradio interface
with gr.Blocks(title="Food to Recipe Generator") as demo:
    gr.Markdown("""
    # 🍽️ Food to Recipe Generator
    ### Powered by Qwen2.5-VL Vision-Language Model

    Upload an image of food and get an AI-generated recipe with ingredients and cooking instructions!

    **How it works:**
    1. Upload a food image using the uploader below
    2. Click "Generate Recipe" button (or it will auto-generate on upload)
    3. Get a complete recipe with ingredients and step-by-step instructions!

    ---
    """)

    with gr.Row():
        with gr.Column():
            image_input = gr.Image(
                label="Upload Food Image",
                type="pil",
                height=400
            )
            generate_btn = gr.Button(
                "🚀 Generate Recipe",
                variant="primary",
                size="lg"
            )

            gr.Markdown("""
            **Tips:**
            - Use clear, well-lit photos of food
            - Close-up shots work better
            - Multiple dishes may produce combined recipes
            """)

        with gr.Column():
            recipe_output = gr.Textbox(
                label="Generated Recipe",
                lines=25,
                max_lines=35,
                placeholder="Your recipe will appear here...\n\nUpload an image to get started!"
            )

    gr.Markdown("""
    ---
    **Model Info:**
    - Base: Qwen2.5-VL-3B-Instruct (Alibaba Cloud)
    - Fine-tuned on Recipe1M+ dataset
    - Uses vision-language understanding for recipe generation

    **Disclaimer:** AI-generated recipes should be reviewed before cooking.
    Verify measurements, cooking times, and food safety guidelines.
    """)

    # Connect the function to the interface
    generate_btn.click(
        fn=generate_recipe_from_image,
        inputs=image_input,
        outputs=recipe_output
    )

    # Also generate on image upload
    image_input.upload(
        fn=generate_recipe_from_image,
        inputs=image_input,
        outputs=recipe_output
    )

if __name__ == "__main__":
    print("Starting Food to Recipe Generator...")
    demo.launch()
