"""
Pipeline for easy recipe generation from food images
"""

import torch
from PIL import Image
from typing import Union, List
from model import FoodToRecipeModel


class RecipeGenerationPipeline:
    
    def __init__(
        self,
        model: FoodToRecipeModel,
        device: str = None
    ):
        self.model = model
        
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        self.model.to(self.device)
        self.model.eval()
    
    def preprocess_image(self, image: Union[str, Image.Image]) -> torch.Tensor:
        if isinstance(image, str):
            image = Image.open(image).convert('RGB')
        

        inputs = self.model.processor(images=image, return_tensors="pt")
        pixel_values = inputs["pixel_values"].to(self.device)
        
        return pixel_values
    
    def __call__(
        self,
        images: Union[str, Image.Image, List[Union[str, Image.Image]]],
        max_length: int = 256,
        temperature: float = 1.0,
        top_k: int = 50,
        top_p: float = 0.95
    ) -> Union[str, List[str]]:
        single_input = not isinstance(images, list)
        if single_input:
            images = [images]
        

        all_recipes = []
        
        for image in images:
            pixel_values = self.preprocess_image(image)
            

            with torch.no_grad():
                generated_ids = self.model.generate(
                    pixel_values,
                    max_length=max_length,
                    temperature=temperature,
                    top_k=top_k,
                    top_p=top_p
                )
            

            recipe = self.model.tokenizer.decode(generated_ids[0], skip_special_tokens=True)
            all_recipes.append(recipe)
        
        return all_recipes[0] if single_input else all_recipes

