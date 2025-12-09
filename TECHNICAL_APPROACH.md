# Technical Approach: What Did You Do Exactly?

## What Did You Do Exactly? How Did You Solve the Problem?

### The Solution Architecture

**Step 1: Model Selection**
We used **Qwen2.5-VL-3B-Instruct**, a vision-language model that can both understand images and generate text in a single unified model. This model has 3.75 billion parameters and was pre-trained on large amounts of image-text pairs.

**Step 2: Data Preparation**
- Loaded 13,501 recipes from a CSV file
- Matched each recipe to its corresponding food image (13,471 successful matches)
- Formatted recipes into a structured text format:
  ```
  RECIPE: [Title]
  
  INGREDIENTS:
  [List of ingredients with amounts]
  
  INSTRUCTIONS:
  [Step-by-step cooking instructions]
  ```

**Step 3: Fine-Tuning Process**
- Created a custom dataset class (`QwenRecipeDataset`) that:
  - Loads images using PIL
  - Formats prompts asking the model to generate recipes from images
  - Prepares the data in Qwen's expected chat format
- Fine-tuned the model using the Hugging Face `Trainer` API with:
  - Batch size: 1 per GPU (with gradient accumulation of 128 for effective batch size of 128)
  - Learning rate: 5e-5
  - Training for 5 epochs
  - Mixed precision training (FP16/BF16) to save memory
  - Gradient checkpointing to reduce memory usage
  - DeepSpeed ZeRO Stage 2 for multi-GPU training efficiency

**Step 4: Recipe Generation**
- Created a function that:
  - Takes a food image as input
  - Processes it through the model with a prompt: "Generate a detailed cooking recipe from this food image"
  - Uses sampling parameters (temperature=0.7, top_p=0.9) to generate diverse but coherent recipes
  - Returns the generated recipe text

**Step 5: Evaluation**
- Implemented multiple metrics to measure quality:
  - **ROUGE-L**: Measures how similar the generated recipe is to the ground truth
  - **BLEU**: Measures word overlap between generated and actual recipes
  - **Ingredient F1**: Measures how well the model identifies correct ingredients
  - **Instruction F1**: Measures how well the model generates correct cooking steps

### How This Solves the Problem

The key insight is using a **native vision-language model** instead of separate vision and language models. This means:

1. **Single Model**: One model handles both understanding the image and generating the recipe, so it learns the direct connection between visual features (colors, textures, shapes) and recipe components (ingredients, cooking methods).

2. **Instruction Tuning**: Qwen2.5-VL is instruction-tuned, meaning it's trained to follow prompts. When we ask it to "generate a recipe from this image," it knows how to structure the output properly.

3. **Fine-Tuning on Recipe Data**: By training on 13,471 image-recipe pairs, the model learns:
   - What visual features correspond to which ingredients
   - How cooking methods appear in images (grilled vs. baked vs. fried)
   - How to format recipes in a standard way

4. **End-to-End Learning**: Unlike systems that first identify food then search for recipes, this model learns to directly map from visual input to recipe output, capturing subtle relationships that rule-based systems miss.

---

## Why Did You Think It Would Be Successful?

### Theoretical Reasons

1. **Vision-Language Models Work**: Models like Qwen2.5-VL have shown strong performance on tasks requiring both vision and language understanding. They can identify objects, understand scenes, and generate descriptive text.

2. **Instruction Tuning**: The model is already trained to follow instructions, so asking it to generate recipes in a specific format should work better than training a model from scratch.

3. **Sufficient Training Data**: With 13,471 examples, we have enough data for the model to learn patterns. Each example shows the model what a good recipe looks like for a given image.

4. **Structured Output**: Recipes have a clear structure (title, ingredients, instructions), which makes it easier for the model to learn the format compared to free-form text generation.

5. **Visual Clues**: Food images contain many visual clues that correlate with recipes:
   - Color indicates cooking method (browned = roasted/grilled)
   - Texture shows preparation (smooth = pureed, chunky = diced)
   - Presentation suggests cuisine type (arrangement, garnishes)

### Practical Evidence

- The model architecture (Qwen2.5-VL) is state-of-the-art for vision-language tasks
- Similar models have been successfully fine-tuned for other domain-specific tasks
- The evaluation metrics showed the model was learning (loss decreased from ~5.72 to ~2.74 during training)

---

## Is Anything New in Your Approach?

### What's New

1. **Direct Image-to-Recipe Generation**: Most existing systems use a two-step process:
   - Step 1: Identify the food (using CLIP or similar)
   - Step 2: Search for or generate a recipe based on the food name
   
   Our approach uses a single model that directly generates recipes from images, learning the visual-to-recipe mapping end-to-end.

2. **Fine-Tuning a Native Vision-Language Model for Recipes**: While vision-language models exist, fine-tuning them specifically for recipe generation from food images is a novel application. Most recipe generation systems either:
   - Use text-only models (need you to describe the food)
   - Use separate vision and language components
   - Don't learn the visual-to-recipe connection directly

3. **Comprehensive Evaluation Metrics**: We implemented multiple metrics (ROUGE-L, BLEU, Ingredient F1, Instruction F1) to evaluate different aspects of recipe quality, not just overall similarity.

4. **Structured Recipe Format**: We trained the model to generate recipes in a specific structured format (Title, Ingredients, Instructions) rather than free-form text, making the output more usable.

### What's Not New

- Using pre-trained vision-language models (Qwen2.5-VL is from existing research)
- Fine-tuning techniques (standard transfer learning)
- Evaluation metrics (ROUGE and BLEU are standard NLP metrics)
- The general approach of fine-tuning large models for specific tasks

**The novelty is in the application**: Using a native vision-language model for end-to-end recipe generation from food images, rather than combining separate components.

---

## What Problems Did You Anticipate?

### Before Starting

1. **Memory Issues**: 
   - **Anticipated**: Large models (3.75B parameters) require significant GPU memory
   - **Solution Planned**: Use gradient checkpointing, mixed precision (FP16), and DeepSpeed ZeRO

2. **Data Quality**:
   - **Anticipated**: Some images might not match recipes, or recipes might be poorly formatted
   - **Solution Planned**: Validate image-recipe pairs and clean the data

3. **Overfitting**:
   - **Anticipated**: With limited data (13K examples), the model might memorize recipes
   - **Solution Planned**: Use validation split, monitor validation loss, and limit training epochs

4. **Generation Quality**:
   - **Anticipated**: Generated recipes might be repetitive or incoherent
   - **Solution Planned**: Use repetition penalty, temperature sampling, and post-processing

5. **Training Time**:
   - **Anticipated**: Fine-tuning large models takes a long time
   - **Solution Planned**: Use multi-GPU training with DeepSpeed for efficiency

---

## What Problems Did You Encounter?

### Problems That Actually Occurred

1. **Device Mismatch Errors** (Major Issue)
   - **Problem**: When using multiple GPUs, tensors would end up on different devices, causing "device mismatch" errors
   - **Symptoms**: RuntimeError about tensors being on different CUDA devices
   - **Solution**: 
     - Switched to single-GPU training mode
     - Explicitly moved all tensors to the same device (cuda:0)
     - Added device synchronization and explicit device placement in data collator
   - **Evidence**: Code comments show "FORCE SINGLE GPU TO AVOID DEVICE MISMATCH ERRORS"

2. **Image Processing Complexity**
   - **Problem**: Qwen2.5-VL requires specific image processing format with `image_grid_thw` metadata
   - **Symptoms**: Errors about missing or incorrect image grid dimensions
   - **Solution**: 
     - Created custom data collator (`QwenDataCollator`) that properly handles image metadata
     - Used `process_vision_info` utility from `qwen_vl_utils` to format images correctly
     - Added validation to ensure image_grid_thw is properly formatted

3. **Memory Constraints**
   - **Problem**: Even with optimizations, training on full dataset was memory-intensive
   - **Symptoms**: Out-of-memory (OOM) errors during training
   - **Solution**: 
     - Used very small batch size (1) with large gradient accumulation (128)
     - Enabled gradient checkpointing
     - Used FP16/BF16 mixed precision
     - Limited training to subset of data initially (1000 samples for testing)

4. **Label Masking Issues**
   - **Problem**: During training, all labels were being masked (set to -100), causing loss to be None
   - **Symptoms**: Training loss was None or NaN
   - **Solution**: 
     - Fixed label creation in dataset to properly mask only padding tokens, not actual recipe tokens
     - Added validation checks to ensure labels contain non-masked values
     - Debugged label creation process step-by-step

5. **Generation Repetition**
   - **Problem**: Generated recipes would repeat the same phrases or ingredients
   - **Symptoms**: Output like "chicken chicken chicken" or repeated instructions
   - **Solution**: 
     - Increased repetition penalty (1.5)
     - Adjusted temperature and top_p sampling parameters
     - Added post-processing to remove obvious repetitions

6. **DeepSpeed Linker Warnings**
   - **Problem**: Non-critical warnings about missing `libaio` library when importing DeepSpeed
   - **Symptoms**: Warnings (not errors) during import
   - **Solution**: 
     - Documented that warnings are non-critical
     - DeepSpeed works without libaio, just with reduced I/O optimizations
     - Added instructions for optional installation

### Problems That Didn't Occur (But Were Anticipated)

- **Data quality issues**: The dataset was cleaner than expected
- **Model convergence**: The model trained successfully without major convergence issues
- **Evaluation metrics**: All metrics computed successfully

---

## Did the Very First Thing You Tried Work?

### No - Multiple Approaches Were Tried

**First Attempt: Deepseek Janus-Pro-1B**
- **What was tried**: Using Deepseek Janus-Pro-1B, another vision-language model
- **Why it failed**: 
  - Device mismatch errors were more severe
  - Model had issues with image token processing
  - More complex setup requirements
- **Evidence**: The notebook `Food_to_receipe_Deep_seek_Janus_1B.ipynb` shows extensive error handling and device management code, suggesting significant problems

**Second Attempt: Qwen2.5-VL-3B-Instruct** (Current Solution)
- **What was tried**: Switched to Qwen2.5-VL-3B-Instruct
- **Why it worked better**:
  - Better documentation and examples
  - More stable image processing pipeline
  - Better instruction-following capabilities
  - Cleaner API for fine-tuning
- **Result**: This approach worked, though still required fixes for device management and memory optimization

### Evolution of the Solution

1. **Initial Idea**: Use CLIP for food recognition + separate text model for recipe generation
2. **First Implementation**: Deepseek Janus-Pro-1B (encountered device and processing issues)
3. **Second Implementation**: Qwen2.5-VL-3B-Instruct (worked after fixing device and memory issues)
4. **Refinements**: 
   - Custom data collator for proper image handling
   - Single-GPU training to avoid device mismatches
   - Memory optimizations (gradient checkpointing, mixed precision)
   - Comprehensive evaluation metrics

**The first approach (Deepseek) did not work smoothly, but the second approach (Qwen) worked after addressing the encountered problems.**

---

## Code Repositories and Sources Used

### Primary Models and Libraries

1. **Qwen2.5-VL-3B-Instruct**
   - **Source**: Hugging Face Model Hub
   - **URL**: https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct
   - **Citation**: Qwen Team. (2024). Qwen2.5-VL: A Versatile Vision-Language Model. Alibaba Cloud.
   - **What we used**: The pre-trained model weights and processor
   - **Changes made**: 
     - Fine-tuned on our recipe dataset
     - Modified input format to include recipe generation prompts
     - Customized data collator for batch processing

2. **Hugging Face Transformers**
   - **Source**: Hugging Face Transformers Library
   - **URL**: https://github.com/huggingface/transformers
   - **Citation**: Wolf, T., et al. (2020). "Transformers: State-of-the-Art Natural Language Processing." Proceedings of EMNLP.
   - **What we used**: 
     - `Qwen2_5_VLForConditionalGeneration` model class
     - `AutoProcessor` for text and image processing
     - `Trainer` API for training
     - `TrainingArguments` for hyperparameter configuration
   - **Changes made**: 
     - Created custom dataset class (`QwenRecipeDataset`)
     - Created custom data collator (`QwenDataCollator`) to handle image metadata
     - Modified training loop to handle vision-language inputs

3. **qwen-vl-utils**
   - **Source**: Qwen Vision-Language Utilities
   - **URL**: Part of Qwen model package
   - **What we used**: `process_vision_info` function to format images for Qwen
   - **Changes made**: Integrated into our data processing pipeline

4. **DeepSpeed**
   - **Source**: Microsoft DeepSpeed
   - **URL**: https://github.com/microsoft/DeepSpeed
   - **Citation**: Rajbhandari, S., et al. (2020). "ZeRO: Memory Optimizations Toward Training Trillion Parameter Models." SC20.
   - **What we used**: DeepSpeed ZeRO Stage 2 for multi-GPU memory optimization
   - **Changes made**: 
     - Created DeepSpeed configuration file (`ds_config.json`)
     - Configured for ZeRO Stage 2 with CPU offloading
     - Integrated with Hugging Face Trainer

5. **PyTorch**
   - **Source**: PyTorch
   - **URL**: https://pytorch.org/
   - **Citation**: Paszke, A., et al. (2019). "PyTorch: An Imperative Style, High-Performance Deep Learning Library." NeurIPS.
   - **What we used**: Core deep learning framework
   - **Changes made**: Standard usage, no modifications

6. **ROUGE Score**
   - **Source**: Google Research
   - **URL**: https://github.com/google-research/google-research/tree/master/rouge
   - **Citation**: Lin, C. Y. (2004). "ROUGE: A Package for Automatic Evaluation of Summaries." ACL Workshop.
   - **What we used**: `rouge_scorer` for evaluating recipe similarity
   - **Changes made**: Integrated into custom evaluation function

7. **SacreBLEU**
   - **Source**: SacreBLEU
   - **URL**: https://github.com/mjpost/sacrebleu
   - **Citation**: Post, M. (2018). "A Call for Clarity in Reporting BLEU Scores." ACL.
   - **What we used**: BLEU score computation for recipe evaluation
   - **Changes made**: Integrated into custom evaluation function

8. **Gradio**
   - **Source**: Gradio
   - **URL**: https://github.com/gradio-app/gradio
   - **Citation**: Abid, A., et al. (2019). "Gradio: Hassle-Free Sharing and Testing of ML Models in the Wild." ICML.
   - **What we used**: Web interface for recipe generation
   - **Changes made**: Created custom interface for image upload and recipe display

### Dataset

9. **Food Ingredients and Recipe Dataset with Image Name Mapping**
   - **Source**: Public dataset (CSV file provided)
   - **What we used**: 13,501 recipes with ingredients, instructions, and image mappings
   - **Changes made**: 
     - Processed and cleaned the data
     - Matched recipes to image files
     - Formatted recipes into structured text format for training
     - Split into train/test sets (80/20)

### Key Code Modifications

**Custom Dataset Class** (`QwenRecipeDataset`):
- Extends PyTorch `Dataset`
- Loads images and formats them for Qwen
- Creates chat-style prompts for recipe generation
- Handles image preprocessing

**Custom Data Collator** (`QwenDataCollator`):
- Handles batching of vision-language inputs
- Properly formats `image_grid_thw` metadata for Qwen
- Manages padding for variable-length sequences
- Ensures all tensors are on the correct device

**Evaluation Functions**:
- `extract_recipe_sections`: Parses generated recipes to extract ingredients and instructions
- `compute_f1_score`: Computes F1 scores for ingredient and instruction matching
- `compute_recipe_metrics`: Combines ROUGE-L, BLEU, and F1 scores

**Training Configuration**:
- Custom `TrainingArguments` with memory optimizations
- DeepSpeed configuration for multi-GPU training
- Gradient accumulation to simulate larger batch sizes
- Mixed precision training (FP16/BF16)

### Summary of Changes

All repositories and libraries were used as-is for their core functionality. Our contributions were:

1. **Application-specific fine-tuning**: Adapting Qwen2.5-VL for recipe generation
2. **Custom data processing**: Creating dataset and collator classes for recipe data
3. **Evaluation framework**: Implementing recipe-specific metrics
4. **Integration**: Combining multiple libraries into a working system
5. **Problem-solving**: Fixing device management, memory, and processing issues

We did not modify the core model architectures or library code - we used them through their APIs and fine-tuned the models on our specific task.










