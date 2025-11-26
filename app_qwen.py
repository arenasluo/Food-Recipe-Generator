"""
Food to Recipe Generator - Gradio Web App
Using Qwen2.5-VL-3B-Instruct for recipe generation from food images
"""

import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
from PIL import Image
import gradio as gr
import os
import sys

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Global variables for loaded model
model = None
processor = None


@torch.no_grad()
def load_models():
    """Load Qwen model and processor"""
    global model, processor
    
    if model is not None and processor is not None:
        print("Models already loaded, reusing...")
        return model, processor
    
    print("Loading Qwen2.5-VL model...")
    
    # Try to load fine-tuned model first
    model_path = "./qwen_recipe_model_final"
    
    if os.path.exists(model_path):
        print(f"Loading fine-tuned model from {model_path}")
        try:
            model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                model_path,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                device_map="auto" if torch.cuda.is_available() else None,
                trust_remote_code=True
            )
            processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
            print("✓ Fine-tuned model loaded successfully!")
        except Exception as e:
            print(f"⚠️ Error loading fine-tuned model: {e}")
            print("Falling back to base model...")
            model_path = "Qwen/Qwen2.5-VL-3B-Instruct"
            model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                model_path,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                device_map="auto" if torch.cuda.is_available() else None,
                trust_remote_code=True
            )
            processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    else:
        print(f"Fine-tuned model not found at {model_path}")
        print("Loading base Qwen2.5-VL-3B-Instruct model...")
        model_path = "Qwen/Qwen2.5-VL-3B-Instruct"
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None,
            trust_remote_code=True
        )
        processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    
    model.eval()
    
    # Move to device if not using device_map
    if not torch.cuda.is_available():
        model = model.to(device)
    
    print("✓ Models loaded successfully!")
    return model, processor


def generate_recipe(image, max_new_tokens=1024, temperature=0.7, top_p=0.9, progress=None):
    """Generate a recipe from a food image using Qwen2.5-VL"""
    
    if image is None:
        return "⚠️ Please upload a food image."
    
    global model, processor
    
    # Helper function for progress updates
    def update_progress(value, desc):
        if progress is not None:
            try:
                progress(value, desc=desc)
            except:
                pass
    
    try:
        update_progress(0, "Loading model...")
        # Load models if not already loaded
        if model is None or processor is None:
            model, processor = load_models()
        
        update_progress(0.2, "Processing image...")
        
        # Ensure PIL Image
        if not isinstance(image, Image.Image):
            image = Image.fromarray(image)
        
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Create prompt
        prompt_text = "Generate a detailed cooking recipe from this food image. Include ingredients, instructions, cooking times, and temperatures."
        
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
        model_device = next(model.parameters()).device
        inputs = {k: v.to(model_device) for k, v in inputs.items()}
        
        # Get token IDs
        pad_token_id = processor.tokenizer.pad_token_id or processor.tokenizer.eos_token_id
        eos_token_id = processor.tokenizer.eos_token_id
        
        update_progress(0.5, "Generating recipe...")
        
        # Generate
        model.eval()
        with torch.inference_mode():
            generated_ids = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=temperature,
                top_p=top_p,
                pad_token_id=pad_token_id,
                eos_token_id=eos_token_id,
                repetition_penalty=1.1,
            )
        
        # Decode only the newly generated tokens
        input_length = inputs['input_ids'].shape[1]
        generated_tokens = generated_ids[0][input_length:]
        
        update_progress(0.9, "Decoding recipe...")
        
        # Decode
        recipe = processor.decode(generated_tokens, skip_special_tokens=True)
        
        update_progress(1.0, "Complete!")
        
        return recipe.strip()
    
    except Exception as e:
        import traceback
        error_msg = f"❌ Error generating recipe: {str(e)}"
        print(error_msg)
        print(traceback.format_exc())
        return error_msg


# Load models at startup
print("🚀 Loading models...")
try:
    model, processor = load_models()
    print("✓ Models ready!")
except Exception as e:
    print(f"⚠️ Error loading models: {e}")
    print("Models will be loaded on first use.")

# Create Gradio interface
with gr.Blocks(title="Food to Recipe Generator", theme=gr.themes.Soft()) as demo:
    gr.Markdown("""
    # 🍽️ Food to Recipe Generator
    
    Upload an image of food and get an AI-generated recipe with ingredients and instructions!
    
    **Powered by Qwen2.5-VL-3B-Instruct**
    """)
    
    with gr.Row():
        with gr.Column(scale=1):
            image_input = gr.Image(
                label="Upload Food Image",
                type="pil",
                height=400
            )
            
            with gr.Accordion("Advanced Settings", open=False):
                max_tokens = gr.Slider(
                    minimum=256,
                    maximum=2048,
                    value=1024,
                    step=128,
                    label="Max Tokens",
                    info="Maximum length of generated recipe"
                )
                temperature = gr.Slider(
                    minimum=0.1,
                    maximum=2.0,
                    value=0.7,
                    step=0.1,
                    label="Temperature",
                    info="Controls randomness (lower = more deterministic)"
                )
                top_p = gr.Slider(
                    minimum=0.1,
                    maximum=1.0,
                    value=0.9,
                    step=0.05,
                    label="Top-p",
                    info="Nucleus sampling threshold"
                )
            
            generate_btn = gr.Button("🚀 Generate Recipe", variant="primary", size="lg")
            
            gr.Markdown("""
            ### Tips:
            - Upload clear images of food dishes
            - Works best with single dishes
            - Generated recipes may need refinement
            """)
        
        with gr.Column(scale=1):
            recipe_output = gr.Textbox(
                label="Generated Recipe",
                lines=30,
                max_lines=40,
                placeholder="Your recipe will appear here...",
                show_copy_button=True,
                autoscroll=True
            )
    
    # Examples section
    gr.Markdown("## 📸 Example Images")
    
    # Look for example images in the food_images directory
    example_images = []
    food_images_dir = "./food_images/Food Images"
    if os.path.exists(food_images_dir):
        # Get first few images as examples
        import glob
        image_files = glob.glob(os.path.join(food_images_dir, "*.jpg"))[:6]
        if image_files:
            example_images = image_files
    
    if example_images:
        gr.Examples(
            examples=[[img] for img in example_images],
            inputs=image_input,
            outputs=recipe_output,
            fn=lambda img: generate_recipe(img, max_new_tokens=1024, temperature=0.7, top_p=0.9),
            cache_examples=False
        )
    
    # Connect function
    generate_btn.click(
        fn=lambda img, max_t, temp, top_p: generate_recipe(img, max_t, temp, top_p),
        inputs=[image_input, max_tokens, temperature, top_p],
        outputs=recipe_output
    )
    
    # Auto-generate on upload (optional - commented out to prevent auto-runs)
    # image_input.upload(
    #     fn=lambda img, max_t, temp, top_p: generate_recipe(img, max_t, temp, top_p),
    #     inputs=[image_input, max_tokens, temperature, top_p],
    #     outputs=recipe_output
    # )
    
    gr.Markdown("""
    ---
    **Note**: Recipes are AI-generated and may need refinement. Please verify ingredients and instructions before cooking!
    
    Built with [Qwen2.5-VL](https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct) + [Gradio](https://gradio.app)
    """)

# Launch
if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True
    )

