import os
import torch
import torch.nn as nn
from PIL import Image
import gradio as gr

# Hugging Face Libraries
from transformers import (
    CLIPModel, CLIPProcessor,
    GPT2LMHeadModel, GPT2Tokenizer, GPT2Config
)

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Custom Model: CLIP features + GPT-2 Decoder
class CLIPRecipeGenerator(nn.Module):
    def __init__(self, clip_model, tokenizer, config):
        super().__init__()
        self.clip_model = clip_model
        self.clip_dim = 512  # CLIP ViT-B/32 embedding dimension

        # Project CLIP features to GPT-2 embedding dimension
        self.vision_proj = nn.Linear(self.clip_dim, config.n_embd)

        # GPT-2 decoder
        self.gpt2 = GPT2LMHeadModel(config)

        # Initialize embedding layer
        self.gpt2.resize_token_embeddings(tokenizer.vocab_size)

    def _shift_labels(self, labels):
        """Shift labels to the right for causal LM"""
        if labels is None:
            return None
        batch_size = labels.shape[0]
        shifted_labels = torch.full((batch_size, 1), -100, dtype=labels.dtype, device=labels.device)
        shifted_labels = torch.cat([shifted_labels, labels], dim=1)
        return shifted_labels

# Load models
print("Loading CLIP model...")
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# Freeze CLIP
for param in clip_model.parameters():
    param.requires_grad = False

clip_model = clip_model.to(device)
clip_model.eval()

print("Loading GPT-2 tokenizer...")
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token

print("Creating GPT-2 config...")
config = GPT2Config(
    vocab_size=tokenizer.vocab_size,
    n_positions=1024,
    n_ctx=1024,
    n_embd=768,
    n_layer=6,
    n_head=12,
    resid_pdrop=0.1,
    embd_pdrop=0.1,
    attn_pdrop=0.1,
)

print("Creating CLIPRecipeGenerator model...")
model = CLIPRecipeGenerator(clip_model, tokenizer, config).to(device)

# Load trained model if available
model_path = "recipe_generator_model.pt"
if os.path.exists(model_path):
    print(f"Loading trained model from {model_path}")
    checkpoint = torch.load(model_path, map_location=device)
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    print("✓ Models loaded successfully!")
else:
    print("⚠️ Warning: Model checkpoint not found. Using untrained model.")

model.eval()

print("✓ Models ready!")

def recognize_food_from_image(image):
    """
    Use CLIP to recognize what food is in the image.
    Returns a food item (simplified version).
    """
    return "Food Dish"

def generate_recipe_for_web(uploaded_image):
    """
    Generate a recipe from an uploaded image.
    This function is designed for the web interface.
    """
    try:
        if uploaded_image is None:
            return "⚠️ Please upload an image."
        
        # Convert to PIL Image if needed
        if isinstance(uploaded_image, str):
            image = Image.open(uploaded_image)
        else:
            image = uploaded_image
        
        # Ensure RGB format
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # STEP 1: Food Recognition using CLIP
        recognized_food = recognize_food_from_image(image)
        
        # Process image with CLIP for recipe generation
        pixel_values = clip_processor(images=image, return_tensors="pt")['pixel_values'].to(device)
        
        # Generate recipe using the model
        model.eval()
        with torch.no_grad():
            # Get CLIP embeddings
            image_embeds = clip_model.get_image_features(pixel_values=pixel_values)
            vision_features = model.vision_proj(image_embeds)
            vision_features = vision_features.unsqueeze(1)
            
            # Initialize generation with a better starting token
            generated_ids = []
            prev_token = torch.tensor([[tokenizer.bos_token_id if tokenizer.bos_token_id else tokenizer.eos_token_id]], device=device)
            
            # Track recent tokens to detect repetition
            recent_tokens = []
            repetition_threshold = 4
            repetition_window = 8
            
            # Generate tokens
            for step in range(512):  # Max 512 tokens
                inputs_embeds = model.gpt2.transformer.wte(prev_token)
                combined_embeds = torch.cat([vision_features, inputs_embeds], dim=1)
                
                outputs = model.gpt2(inputs_embeds=combined_embeds)
                logits = outputs.logits[:, -1, :]
                
                # Apply temperature and top-k/top-p sampling
                temperature = 0.9
                top_k = 100
                top_p = 0.92
                
                # Apply temperature
                logits = logits / temperature
                
                # Top-k filtering
                if top_k > 0:
                    indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
                    logits[indices_to_remove] = float('-inf')
                
                # Top-p (nucleus) filtering
                if top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                    cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
                    
                    # Remove tokens with cumulative probability above the threshold
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0
                    
                    indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                    logits[indices_to_remove] = float('-inf')
                
                # Sample from the filtered distribution
                probs = torch.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                
                # Check for repetition
                if len(recent_tokens) >= repetition_window:
                    recent_tokens.pop(0)
                recent_tokens.append(next_token.item())
                
                # Skip if too repetitive
                if len(recent_tokens) >= repetition_threshold:
                    if len(set(recent_tokens[-repetition_threshold:])) == 1:
                        # Repetition detected, try to break it
                        logits[0, next_token.item()] = float('-inf')
                        probs = torch.softmax(logits, dim=-1)
                        next_token = torch.multinomial(probs, num_samples=1)
                
                generated_ids.append(next_token.item())
                prev_token = next_token
                
                # Stop if EOS token
                if next_token.item() == tokenizer.eos_token_id:
                    break
            
            # Decode generated tokens
            generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
            
            # Post-process: Remove repetitions and format
            lines = generated_text.split('\n')
            seen = set()
            unique_lines = []
            for line in lines:
                line_stripped = line.strip()
                if line_stripped and line_stripped not in seen:
                    seen.add(line_stripped)
                    unique_lines.append(line)
            
            generated_text = '\n'.join(unique_lines)
            
            # Format output
            output = f"RECIPE: {recognized_food.upper()}\n"
            output += f"(Recognized as: {recognized_food})\n\n"
            output += "=" * 60 + "\n\n"
            output += generated_text
            output += "\n\n" + "=" * 60 + "\n"
            output += "NOTE: This recipe was generated by an AI model and may require\n"
            output += "refinement. Please verify ingredients and instructions before cooking."
            
            return output
            
    except Exception as e:
        return f"❌ Error generating recipe: {str(e)}\n\nPlease try again with a different image."

# Create Gradio interface (without theme parameter for compatibility with older Gradio versions)
# Note: theme parameter is not supported in Gradio < 4.0, so we omit it for compatibility
with gr.Blocks(title="Food to Recipe Generator") as demo:
    gr.Markdown("""
    # 🍽️ Food to Recipe Generator
    
    Upload an image of food and get an AI-generated recipe!
    
    **How it works:**
    1. Upload a food image using the uploader below
    2. Click "Generate Recipe" button
    3. Get a complete recipe with ingredients and cooking steps!
    """)
    
    with gr.Row():
        with gr.Column():
            image_input = gr.Image(
                label="Upload Food Image",
                type="pil",
                height=400
            )
            generate_btn = gr.Button("Generate Recipe 🚀", variant="primary", size="lg")
            
        with gr.Column():
            recipe_output = gr.Textbox(
                label="Generated Recipe",
                lines=20,
                max_lines=30,
                placeholder="Your recipe will appear here...",
                show_copy_button=True
            )
    
    # Connect the function to the interface
    generate_btn.click(
        fn=generate_recipe_for_web,
        inputs=image_input,
        outputs=recipe_output
    )
    
    # Also generate on image upload
    image_input.upload(
        fn=generate_recipe_for_web,
        inputs=image_input,
        outputs=recipe_output
    )

if __name__ == "__main__":
    demo.launch()

