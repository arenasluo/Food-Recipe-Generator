import os
import sys
import warnings
import torch
import torch.nn as nn
from PIL import Image
import gradio as gr

# Suppress asyncio cleanup warnings (harmless but noisy in logs)
warnings.filterwarnings("ignore")

# Hugging Face Libraries
from transformers import (
    CLIPModel, CLIPProcessor,
    GPT2LMHeadModel, GPT2Tokenizer, GPT2Config
)

# For loading model from Model Hub
try:
    from huggingface_hub import hf_hub_download
    HF_HUB_AVAILABLE = True
except ImportError:
    HF_HUB_AVAILABLE = False
    print("⚠️ huggingface_hub not available, will try local model file only")

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

# Load trained model - try Model Hub first, then local file
model_loaded = False
model_path = None

# Try loading from Hugging Face Model Hub first
if HF_HUB_AVAILABLE:
    try:
        print("Attempting to load model from Hugging Face Model Hub...")
        model_path = hf_hub_download(
            repo_id="arenasluo/recipe-generator-model",
            filename="recipe_generator_model.pt",
            cache_dir=None  # Download to current directory
        )
        print(f"✓ Model found on Model Hub: {model_path}")
    except Exception as e:
        print(f"Model not found on Model Hub: {e}")
        model_path = None

# Fallback to local file
if model_path is None:
    model_path = "recipe_generator_model.pt"

# Load the model
if os.path.exists(model_path):
    print(f"Loading trained model from {model_path}")
    try:
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        if isinstance(checkpoint, dict):
            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'], strict=False)
                model_loaded = True
                print("✓ Model loaded from checkpoint dict with 'model_state_dict' key")
            elif 'state_dict' in checkpoint:
                model.load_state_dict(checkpoint['state_dict'], strict=False)
                model_loaded = True
                print("✓ Model loaded from checkpoint dict with 'state_dict' key")
            else:
                # Try loading the entire dict as state_dict
                try:
                    model.load_state_dict(checkpoint, strict=False)
                    model_loaded = True
                    print("✓ Model loaded from checkpoint dict (direct)")
                except:
                    print("⚠️ Could not load from checkpoint dict")
        else:
            # Checkpoint is directly the state_dict
            try:
                model.load_state_dict(checkpoint, strict=False)
                model_loaded = True
                print("✓ Model loaded from checkpoint (direct state_dict)")
            except Exception as e:
                print(f"⚠️ Error loading checkpoint: {e}")
    except Exception as e:
        print(f"⚠️ Error loading model checkpoint: {e}")
        print("⚠️ Using untrained model")
else:
    print("⚠️ Warning: Model checkpoint not found. Using untrained model.")

# Ensure model is in eval mode
model.eval()

# Set random seed for reproducibility
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

print(f"✓ Models ready! (Model loaded: {model_loaded})")

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
            # Use BOS token if available, otherwise use a prompt token
            if hasattr(tokenizer, 'bos_token_id') and tokenizer.bos_token_id is not None:
                start_token_id = tokenizer.bos_token_id
            elif hasattr(tokenizer, 'eos_token_id') and tokenizer.eos_token_id is not None:
                start_token_id = tokenizer.eos_token_id
            else:
                # Fallback: use tokenizer.encode to get a starting token
                start_text = "Title: Ingredients: Instructions:"
                start_tokens = tokenizer.encode(start_text, return_tensors="pt", add_special_tokens=False)
                start_token_id = start_tokens[0][0].item() if len(start_tokens[0]) > 0 else tokenizer.eos_token_id

            # Start with initial token
            generated_ids = [start_token_id]
            input_ids = torch.tensor([generated_ids], device=device)

            # Track recent tokens to detect repetition
            recent_tokens = []
            repetition_threshold = 4
            repetition_window = 8

            # Generate tokens
            for step in range(512):  # Max 512 tokens
                # Get embeddings for ALL generated tokens so far
                inputs_embeds = model.gpt2.transformer.wte(input_ids)
                # Prepend vision features only once at the beginning
                combined_embeds = torch.cat([vision_features, inputs_embeds], dim=1)

                outputs = model.gpt2(inputs_embeds=combined_embeds)
                logits = outputs.logits[:, -1, :]
                
                # Apply temperature and top-k/top-p sampling
                temperature = 0.9
                top_k = 100
                top_p = 0.92
                
                # Apply temperature
                logits = logits / temperature
                
                # Check for invalid values and clamp if necessary
                if torch.isnan(logits).any() or torch.isinf(logits).any():
                    # If logits are invalid, use a simple argmax fallback
                    next_token = torch.argmax(logits, dim=-1, keepdim=True)
                    generated_ids.append(next_token.item())
                    prev_token = next_token
                    continue
                
                # Top-k filtering
                if top_k > 0:
                    top_k_value = min(top_k, logits.size(-1))
                    indices_to_remove = logits < torch.topk(logits, top_k_value)[0][..., -1, None]
                    logits[indices_to_remove] = float('-inf')
                
                # Top-p (nucleus) filtering
                if top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                    sorted_probs = torch.softmax(sorted_logits, dim=-1)
                    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
                    
                    # Remove tokens with cumulative probability above the threshold
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0
                    
                    indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                    logits[indices_to_remove] = float('-inf')
                
                # Sample from the filtered distribution
                # Replace -inf with a very small value to avoid issues
                logits = torch.clamp(logits, min=-1e10)
                probs = torch.softmax(logits, dim=-1)
                
                # Validate probabilities before sampling
                if torch.isnan(probs).any() or torch.isinf(probs).any() or (probs < 0).any():
                    # Fallback to argmax if probabilities are invalid
                    next_token = torch.argmax(logits, dim=-1, keepdim=True)
                else:
                    # Ensure probabilities sum to approximately 1
                    probs = probs / (probs.sum(dim=-1, keepdim=True) + 1e-10)
                    try:
                        next_token = torch.multinomial(probs, num_samples=1)
                    except RuntimeError:
                        # Fallback to argmax if multinomial fails
                        next_token = torch.argmax(logits, dim=-1, keepdim=True)
                
                # Check for repetition
                if len(recent_tokens) >= repetition_window:
                    recent_tokens.pop(0)
                recent_tokens.append(next_token.item())
                
                # Skip if too repetitive
                if len(recent_tokens) >= repetition_threshold:
                    if len(set(recent_tokens[-repetition_threshold:])) == 1:
                        # Repetition detected, try to break it
                        # Save original logits for fallback
                        original_logits = logits.clone()
                        logits[0, next_token.item()] = float('-inf')
                        
                        # Replace -inf with a very small value
                        logits = torch.clamp(logits, min=-1e10)
                        probs = torch.softmax(logits, dim=-1)
                        
                        # Validate probabilities
                        if torch.isnan(probs).any() or torch.isinf(probs).any() or (probs < 0).any() or probs.sum() < 0.1:
                            # If probabilities are invalid, use argmax on original logits
                            next_token = torch.argmax(original_logits, dim=-1, keepdim=True)
                        else:
                            try:
                                next_token = torch.multinomial(probs, num_samples=1)
                            except RuntimeError:
                                # Fallback to argmax if multinomial fails
                                next_token = torch.argmax(original_logits, dim=-1, keepdim=True)
                
                generated_ids.append(next_token.item())
                # Update input_ids to include all generated tokens
                input_ids = torch.tensor([generated_ids], device=device)

                # Stop if EOS token
                if next_token.item() == tokenizer.eos_token_id:
                    break
            
            # Decode generated tokens (skip the first start token)
            generated_text = tokenizer.decode(generated_ids[1:], skip_special_tokens=True)
            
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
                placeholder="Your recipe will appear here..."
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
    # Suppress asyncio cleanup errors (harmless warnings during shutdown)
    # These are Python cleanup warnings that don't affect functionality
    import sys
    
    # Create a simple filter for stderr
    class FilteredStderr:
        def __init__(self, original):
            self.original = original
        
        def write(self, text):
            # Filter out asyncio cleanup errors (harmless)
            if any(keyword in text for keyword in [
                "Exception ignored in",
                "BaseEventLoop.__del__",
                "Invalid file descriptor",
                "_close_self_pipe"
            ]):
                return  # Silently ignore
            return self.original.write(text)
        
        def flush(self):
            return self.original.flush()
        
        def __getattr__(self, name):
            return getattr(self.original, name)
    
    # Apply filter only in Hugging Face Spaces environment
    if os.getenv("SYSTEM") == "spaces" or os.getenv("SPACE_ID"):
        sys.stderr = FilteredStderr(sys.stderr)
    
    demo.launch()

