"""
Inference script for trained Food to Recipe model
"""

import torch
from PIL import Image
from pathlib import Path
import argparse
from pipeline import FoodToRecipeModel, RecipeGenerationPipeline


def load_trained_model(checkpoint_path, device='cuda'):
    print(f"Loading model from {checkpoint_path}...")
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = checkpoint['config']
    

    model = FoodToRecipeModel(
        siglip_model_name=config['siglip_model'],
        d_model=config['d_model'],
        nhead=config['nhead'],
        num_decoder_layers=config['num_decoder_layers'],
        dim_feedforward=config['dim_feedforward'],
        dropout=config['dropout'],
        max_seq_len=config['max_length'],
        freeze_encoder=True
    )
    
    # Load weights
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    print(f"[OK] Model loaded (trained for {checkpoint['epoch']} epochs)")
    print(f"[OK] Best validation loss: {checkpoint['val_loss']:.4f}")
    
    return model


def generate_recipe(model, image_path, temperature=0.8, top_k=50, top_p=0.95, max_length=256):
    """Generate recipe from image."""
    # Create pipeline
    device = next(model.parameters()).device
    pipeline = RecipeGenerationPipeline(model, device=str(device))
    
    image = Image.open(image_path).convert('RGB')

    print(f"\nGenerating recipe for: {image_path}")
    print(f"Parameters: temp={temperature}, top_k={top_k}, top_p={top_p}")
    
    recipe = pipeline(
        image,
        max_length=max_length,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p
    )
    
    return recipe


def main():
    parser = argparse.ArgumentParser(description="Generate recipes from food images")
    parser.add_argument('image', type=str, help='Path to food image')
    parser.add_argument('--checkpoint', type=str, default='./checkpoints/best_model.pt',
                       help='Path to model checkpoint')
    parser.add_argument('--temperature', type=float, default=0.8,
                       help='Sampling temperature (0.7-1.0)')
    parser.add_argument('--top-k', type=int, default=50,
                       help='Top-k sampling parameter')
    parser.add_argument('--top-p', type=float, default=0.95,
                       help='Nucleus sampling parameter')
    parser.add_argument('--max-length', type=int, default=256,
                       help='Maximum recipe length')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use (cuda/cpu)')
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("Food to Recipe - Inference")
    print("=" * 80)
    
    if not Path(args.checkpoint).exists():
        print(f"\n[ERROR] Checkpoint not found: {args.checkpoint}")
        print("\nPlease train the model first:")
        print("  uv run train_kaggle_dataset.py")
        return

    if not Path(args.image).exists():
        print(f"\n[ERROR] Image not found: {args.image}")
        return
    

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")

    model = load_trained_model(args.checkpoint, device)

    recipe = generate_recipe(
        model,
        args.image,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        max_length=args.max_length
    )
    

    print("\n" + "=" * 80)
    print("GENERATED RECIPE")
    print("=" * 80)
    print(recipe)
    print("=" * 80)
    
    output_file = Path(args.image).stem + "_recipe.txt"
    with open(output_file, 'w') as f:
        f.write(recipe)
    print(f"\n[OK] Recipe saved to: {output_file}")


if __name__ == "__main__":
    main()

