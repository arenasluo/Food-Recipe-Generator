"""
Training script for Food Ingredients and Recipe Dataset from Kaggle
Dataset: https://www.kaggle.com/datasets/pes12017000148/food-ingredients-and-recipe-dataset-with-images

This script trains the Food to Recipe model with:
- Frozen SigLIP encoder (pretrained weights)
- Trainable decoder only
- Proper train/val split and evaluation
"""

import os
import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from PIL import Image
from pathlib import Path
from tqdm import tqdm
import numpy as np
from pipeline import FoodToRecipeModel, RecipeGenerationPipeline
import yaml



class KaggleFoodRecipeDataset(Dataset):
    """
    Dataset loader for Kaggle Food Ingredients and Recipe Dataset.
    
    Expected dataset structure:
    dataset/
        Food Ingredients and Recipe Dataset with Image/
            Food Ingredients and Recipe Dataset with Image.csv
            Food Images/
                Food Images/
                    image1.jpg
                    image2.jpg
                    ...
    
    Or you can download using Kaggle API:
        kaggle datasets download -d pes12017000148/food-ingredients-and-recipe-dataset-with-images
    """
    
    def __init__(self, data_dir, model, max_length=512, split='train'):
        self.data_dir = Path(data_dir)
        self.model = model
        self.max_length = max_length
        self.split = split
        

        print(f"Loading {split} dataset from {data_dir}...")
        self.data = self._load_dataset()
        print(f"Loaded {len(self.data)} samples")
    
    def _load_dataset(self):
        import pandas as pd
        

        possible_csv_paths = [
            self.data_dir / "Food Ingredients and Recipe Dataset with Image Name Mapping.csv",
            self.data_dir / "Food Ingredients and Recipe Dataset with Image.csv",
            self.data_dir / "Food Ingredients and Recipe Dataset with Image" / "Food Ingredients and Recipe Dataset with Image.csv",
            self.data_dir / "data.csv",
            self.data_dir / "recipes.csv"
        ]
        
        csv_path = None
        for path in possible_csv_paths:
            if path.exists():
                csv_path = path
                break
        
        if csv_path is None:
            raise FileNotFoundError(
                f"Could not find CSV file in {self.data_dir}. "
                "Please download the dataset from Kaggle:\n"
                "kaggle datasets download -d pes12017000148/food-ingredients-and-recipe-dataset-with-images"
            )

        df = pd.read_csv(csv_path)
        
        possible_img_dirs = [
            self.data_dir / "Food Images" / "Food Images",
            self.data_dir / "Food Ingredients and Recipe Dataset with Image" / "Food Images" / "Food Images",
            self.data_dir / "images",
            self.data_dir / "Food Images"
        ]
        
        img_dir = None
        for path in possible_img_dirs:
            if path.exists():
                img_dir = path
                break
        
        if img_dir is None:
            raise FileNotFoundError(f"Could not find image directory in {self.data_dir}")
        
        print(f"Found CSV: {csv_path}")
        print(f"Found images: {img_dir}")
        print(f"Dataset columns: {df.columns.tolist()}")
        
        data = []
        for idx, row in df.iterrows():
            image_name = None
            for col in ['Image_Name', 'image_name', 'Image', 'image']:
                if col in df.columns:
                    image_name = row[col]
                    break
            
            if image_name is None:
                continue

            if not str(image_name).endswith(('.jpg', '.jpeg', '.png')):
                image_name = f"{image_name}.jpg"
            
            image_path = img_dir / image_name
            
            if not image_path.exists():
           
                base_name = Path(image_name).stem
                for ext in ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']:
                    test_path = img_dir / f"{base_name}{ext}"
                    if test_path.exists():
                        image_path = test_path
                        break
            
            if not image_path.exists():
                continue
            

            recipe_text = None
            for col in ['Instructions', 'instructions', 'Directions', 'directions', 'Recipe', 'recipe']:
                if col in df.columns and pd.notna(row.get(col)):
                    recipe_text = str(row[col])
                    break
            

            ingredients = None
            for col in ['Ingredients', 'ingredients']:
                if col in df.columns and pd.notna(row.get(col)):
                    ingredients = str(row[col])
                    break
            

            if recipe_text:
                if ingredients:
                    full_recipe = f"Ingredients: {ingredients}\n\nInstructions: {recipe_text}"
                else:
                    full_recipe = recipe_text
                
                data.append({
                    'image_path': str(image_path),
                    'recipe': full_recipe,
                    'title': row.get('Title', row.get('title', f"Recipe {idx}"))
                })
        
        return data
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]

        try:
            image = Image.open(item['image_path']).convert('RGB')
            inputs = self.model.processor(images=image, return_tensors="pt")
            pixel_values = inputs["pixel_values"].squeeze(0)
        except Exception as e:
            print(f"Error loading image {item['image_path']}: {e}")

            pixel_values = torch.zeros(3, 224, 224)

        recipe_tokens = self.model.tokenizer(
            item['recipe'],
            return_tensors="pt",
            padding="max_length",
            max_length=self.max_length,
            truncation=True
        )["input_ids"].squeeze(0)
        
        return pixel_values, recipe_tokens



def train_epoch(model, dataloader, optimizer, criterion, device, epoch):
    model.train()
    total_loss = 0
    num_batches = len(dataloader)
    
    progress_bar = tqdm(dataloader, desc=f"Epoch {epoch}")
    
    for batch_idx, (images, recipes) in enumerate(progress_bar):
        images = images.to(device)
        recipes = recipes.to(device)

        input_ids = recipes[:, :-1]
        target_ids = recipes[:, 1:]
        

        optimizer.zero_grad()
        logits = model(images, input_ids)
        

        loss = criterion(
            logits.reshape(-1, logits.size(-1)),
            target_ids.reshape(-1)
        )
        

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        total_loss += loss.item()
        

        progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    return total_loss / num_batches


def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    num_batches = len(dataloader)
    
    with torch.no_grad():
        for images, recipes in tqdm(dataloader, desc="Evaluating"):
            images = images.to(device)
            recipes = recipes.to(device)
            
            input_ids = recipes[:, :-1]
            target_ids = recipes[:, 1:]
            
            logits = model(images, input_ids)
            loss = criterion(
                logits.reshape(-1, logits.size(-1)),
                target_ids.reshape(-1)
            )
            
            total_loss += loss.item()
    
    return total_loss / num_batches


def generate_sample_recipes(model, dataset, device, num_samples=3):
    model.eval()
    
    samples = []
    indices = np.random.choice(len(dataset), min(num_samples, len(dataset)), replace=False)
    
    print("\n" + "=" * 80)
    print("Sample Generated Recipes:")
    print("=" * 80)
    
    with torch.no_grad():
        for idx in indices:
            pixel_values, recipe_tokens = dataset[idx]
            pixel_values = pixel_values.unsqueeze(0).to(device)
            

            generated_ids = model.generate(
                pixel_values,
                max_length=256,
                temperature=0.8,
                top_k=50,
                top_p=0.95
            )
            

            generated_text = model.tokenizer.decode(generated_ids[0], skip_special_tokens=True)
            original_text = model.tokenizer.decode(recipe_tokens, skip_special_tokens=True)
            
            print(f"\nSample {idx}:")
            print(f"Generated: {generated_text[:200]}...")
            print(f"Original: {original_text[:200]}...")
            print("-" * 80)
    
    print("=" * 80 + "\n")



def main():
    print("=" * 80)
    print("Food to Recipe - Training on Kaggle Dataset")
    print("Dataset: Food Ingredients and Recipe Dataset with Images")
    print("=" * 80)
    with open('config.yaml','r') as file:
        config = yaml.safe_load(file)
    
    print("\nConfiguration:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    

    print("\n" + "=" * 80)
    print("1. Creating Model")
    print("=" * 80)
    
    model = FoodToRecipeModel(
        siglip_model_name=config['siglip_model'],
        d_model=config['d_model'],
        nhead=config['nhead'],
        num_decoder_layers=config['num_decoder_layers'],
        dim_feedforward=config['dim_feedforward'],
        dropout=config['dropout'],
        max_seq_len=config['max_length'],
        freeze_encoder=True  # FREEZE SIGLIP - only train decoder
    )
    
    device = torch.device(config['device'])
    model = model.to(device)
    
    print(f"\n[OK] Model created with {sum(p.numel() for p in model.parameters()):,} total parameters")
    print(f"[OK] Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    print(f"[OK] Frozen parameters: {sum(p.numel() for p in model.parameters() if not p.requires_grad):,}")
    

    print("\n" + "=" * 80)
    print("2. Loading Dataset")
    print("=" * 80)

    if not Path(config['data_dir']).exists():
        print("\n[WARNING] Dataset not found!")
        print("Please download the dataset from Kaggle:")
        print("  1. Install Kaggle API: pip install kaggle")
        print("  2. Setup API credentials: https://github.com/Kaggle/kaggle-api#api-credentials")
        print("  3. Download dataset:")
        print("     kaggle datasets download -d pes12017000148/food-ingredients-and-recipe-dataset-with-images")
        print("  4. Extract to ./dataset directory")
        print("  5. Run this script again")
        return
    

    full_dataset = KaggleFoodRecipeDataset(
        config['data_dir'],
        model,
        max_length=config['max_length']
    )
    

    train_size = int(config['train_split'] * len(full_dataset))
    val_size = len(full_dataset) - train_size
    
    train_dataset, val_dataset = random_split(
        full_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    print(f"\n[OK] Train samples: {len(train_dataset)}")
    print(f"[OK] Validation samples: {len(val_dataset)}")
    

    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=4,
        pin_memory=True if config['device'] == 'cuda' else False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=4,
        pin_memory=True if config['device'] == 'cuda' else False
    )
    
    print("\n" + "=" * 80)
    print("3. Setup Training")
    print("=" * 80)
    
    criterion = nn.CrossEntropyLoss(ignore_index=model.pad_token_id)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config['learning_rate'],
        weight_decay=config['weight_decay']
    )
    

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=config['num_epochs']
    )
    

    os.makedirs(config['save_dir'], exist_ok=True)
    
    # ========================================================================
    # Training Loop
    # ========================================================================
    
    print("\n" + "=" * 80)
    print("4. Training")
    print("=" * 80)
    
    best_val_loss = float('inf')
    training_history = []
    
    for epoch in range(1, config['num_epochs'] + 1):
        print(f"\n{'='*80}")
        print(f"Epoch {epoch}/{config['num_epochs']}")
        print(f"{'='*80}")
        
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device, epoch)
        

        val_loss = evaluate(model, val_loader, criterion, device)
        

        scheduler.step()
        

        print(f"\nEpoch {epoch} Results:")
        print(f"  Train Loss: {train_loss:.4f}")
        print(f"  Val Loss: {val_loss:.4f}")
        print(f"  Learning Rate: {scheduler.get_last_lr()[0]:.6f}")
        

        training_history.append({
            'epoch': epoch,
            'train_loss': train_loss,
            'val_loss': val_loss,
            'lr': scheduler.get_last_lr()[0]
        })
        
 
        if epoch % 5 == 0:
            generate_sample_recipes(model, full_dataset, device, num_samples=2)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_path = os.path.join(config['save_dir'], 'best_model.pt')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
                'config': config
            }, save_path)
            print(f"\n[OK] Saved best model to {save_path}")
        
        # Save checkpoint
        if epoch % config['save_every'] == 0:
            save_path = os.path.join(config['save_dir'], f'checkpoint_epoch_{epoch}.pt')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
                'config': config
            }, save_path)
            print(f"[OK] Saved checkpoint to {save_path}")
    

    print("\n" + "=" * 80)
    print("5. Training Completed!")
    print("=" * 80)
    

    history_path = os.path.join(config['save_dir'], 'training_history.json')
    with open(history_path, 'w') as f:
        json.dump(training_history, f, indent=2)
    
    print(f"\n[OK] Best validation loss: {best_val_loss:.4f}")
    print(f"[OK] Training history saved to: {history_path}")
    print(f"[OK] Best model saved to: {os.path.join(config['save_dir'], 'best_model.pt')}")
    

    print("\nGenerating final sample recipes...")
    generate_sample_recipes(model, full_dataset, device, num_samples=5)
    
    print("\n" + "=" * 80)
    print("Training Complete!")
    print("=" * 80)
    print("\nTo use the trained model:")
    print("  from food_to_recipe import FoodToRecipeModel, RecipeGenerationPipeline")
    print("  model = FoodToRecipeModel(...)")
    print("  checkpoint = torch.load('./checkpoints/best_model.pt')")
    print("  model.load_state_dict(checkpoint['model_state_dict'])")
    print("  pipeline = RecipeGenerationPipeline(model)")
    print("  recipe = pipeline(your_image)")


if __name__ == "__main__":
    main()

