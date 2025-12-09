# Method Section: Food Image to Recipe Generation

## 1. Problem Formulation

We address the task of generating complete cooking recipes from food images, formulated as a vision-to-language generation problem. Given an input food image $I$, the goal is to generate a structured recipe $R = \{T, G, S\}$ consisting of a title $T$, list of ingredients $G$, and cooking instructions $S$. This is a challenging multimodal task requiring the model to: (1) recognize food items and their visual characteristics, (2) infer ingredients that may not be directly visible, and (3) generate coherent, executable cooking instructions.

## 2. Approach Overview

We implemented and compared two distinct architectures for this task:

1. **CLIP+GPT-2**: A custom encoder-decoder architecture combining CLIP's visual understanding with GPT-2's language generation capabilities
2. **Qwen2.5-VL-3B**: A state-of-the-art vision-language model fine-tuned using Low-Rank Adaptation (LoRA)

Both approaches were benchmarked against the FIRE (Food Image to REcipe) model from WACV 2024, which represents the current state-of-the-art on the Recipe1M dataset.

## 3. Model Architectures

### 3.1 CLIP+GPT-2 Architecture

Our first approach combines two pre-trained models through a learned projection layer:

**Vision Encoder**: We employ CLIP ViT-B/32 as the image encoder. CLIP was chosen for its strong zero-shot visual recognition capabilities, trained on 400 million image-text pairs. The model produces 512-dimensional image embeddings that capture rich semantic information about food items.

**Projection Layer**: A linear projection layer maps CLIP's 512-dimensional visual features to GPT-2's 768-dimensional embedding space:
$$h_v = W_p \cdot f_{CLIP}(I) + b_p$$
where $W_p \in \mathbb{R}^{768 \times 512}$ and $f_{CLIP}(I)$ is the CLIP image embedding.

**Language Decoder**: GPT-2 (124M parameters) serves as the autoregressive text decoder. The projected visual features are prepended to the text embedding sequence as a "visual token," enabling cross-modal attention:
$$E_{combined} = [h_v; E_{text}]$$

**Training Objective**: Standard causal language modeling loss with the visual token position masked:
$$\mathcal{L} = -\sum_{t=1}^{T} \log P(w_t | w_{<t}, h_v)$$

### 3.2 Qwen2.5-VL-3B with LoRA

Our second approach leverages a modern vision-language model with parameter-efficient fine-tuning:

**Base Model**: Qwen2.5-VL-3B-Instruct, a 3-billion parameter vision-language model with native image understanding capabilities. Unlike our CLIP+GPT-2 approach, Qwen2.5-VL was pre-trained end-to-end on vision-language tasks, enabling more sophisticated visual reasoning.

**LoRA Configuration**: To enable efficient fine-tuning within GPU memory constraints, we applied Low-Rank Adaptation:
- Rank $r = 16$ (increased from initial $r = 8$ for better adaptation capacity)
- Scaling factor $\alpha = 32$
- Target modules: Query, Key, Value, and Output projections in attention layers (`q_proj`, `k_proj`, `v_proj`, `o_proj`)
- Dropout: 0.1 for regularization

The LoRA update for a weight matrix $W_0$ is:
$$W = W_0 + \frac{\alpha}{r} BA$$
where $B \in \mathbb{R}^{d \times r}$ and $A \in \mathbb{R}^{r \times k}$ are learned low-rank matrices.

## 4. Dataset and Preprocessing

**Source Data**: We utilized a subset of the Recipe1M dataset, which contains food images paired with recipes including titles, ingredients, and instructions.

**Data Processing Pipeline**:
1. Image validation: Verified image file existence and format compatibility
2. Image resizing: Limited maximum dimension to 448 pixels for memory efficiency (Qwen) or 224 pixels (CLIP)
3. Recipe formatting: Structured recipes into consistent format with clear section headers
4. Text truncation: Limited recipe length to 1500 characters to manage sequence length

**Dataset Splits**:
- CLIP+GPT-2: 1,000 samples (800 train / 200 test)
- Qwen2.5-VL: 1,000 samples (900 train / 100 test, 90/10 split for more training data)

## 5. Training Configuration

### 5.1 CLIP+GPT-2 Training
| Parameter | Value |
|-----------|-------|
| Batch size | 4 |
| Learning rate | 5e-5 |
| Epochs | 5 |
| Optimizer | AdamW |
| Scheduler | Linear warmup (100 steps) |
| CLIP layers | Frozen |

### 5.2 Qwen2.5-VL Training
| Parameter | Value |
|-----------|-------|
| Effective batch size | 8 (1 × 8 gradient accumulation) |
| Learning rate | 1e-4 |
| Epochs | 5 |
| Optimizer | AdamW |
| Scheduler | Cosine with warmup (10%) |
| Precision | FP16 |
| Gradient checkpointing | Enabled |

## 6. Evaluation Metrics

We adopted metrics consistent with the FIRE paper for fair comparison:

1. **ROUGE-L**: Measures longest common subsequence between generated and reference recipes, capturing fluency and content overlap
2. **SacreBLEU**: Computes n-gram precision with brevity penalty, assessing generation quality
3. **Ingredient F1**: Set-based F1 score comparing predicted and ground-truth ingredients at the word level

## 7. Rationale and Hypotheses

**Why CLIP+GPT-2?** We hypothesized that CLIP's strong visual-semantic alignment, trained on massive image-text data, would provide rich food representations. GPT-2's language modeling capabilities could then leverage these representations for coherent recipe generation. This modular approach also allowed independent analysis of visual understanding vs. text generation.

**Why Qwen2.5-VL?** We anticipated that a natively multimodal model would outperform our modular approach due to:
- End-to-end pre-training on vision-language tasks
- More sophisticated cross-modal attention mechanisms
- Larger model capacity (3B vs ~130M parameters)

**Why LoRA?** Full fine-tuning of a 3B parameter model was infeasible given GPU memory constraints. LoRA enables efficient adaptation by training only ~0.1% of parameters while maintaining most of the pre-trained knowledge.

## 8. Anticipated and Encountered Problems

### 8.1 Anticipated Problems

1. **Domain Gap**: Pre-trained models may not generalize well to food-specific visual features
2. **Ingredient Inference**: Many ingredients are not visually apparent (spices, sauces, etc.)
3. **Recipe Structure**: Generating properly formatted, coherent multi-section recipes is challenging
4. **Limited Data**: Our subset (1,000 samples) is significantly smaller than FIRE's training set (259K+ samples)

### 8.2 Encountered Problems

**GPU Memory Exhaustion (Critical)**: The Qwen2.5-VL model initially caused CUDA Out-of-Memory errors due to:
- Multi-GPU gradient synchronization overhead
- Large model size (3B parameters)

*Solution*: We implemented several mitigations:
- Forced single-GPU training via `CUDA_VISIBLE_DEVICES=0`
- Applied LoRA to reduce trainable parameters
- Enabled gradient checkpointing
- Reduced batch size to 1 with gradient accumulation

**Silent Metric Failures**: Initial evaluation showed ROUGE-L and BLEU scores of 0.0, while Ingredient F1 showed 100%.

*Root Cause*: The `rouge-score` and `sacrebleu` libraries were used but never imported. Try-except blocks silently caught the `NameError` and returned 0.0. The 100% Ingredient F1 occurred because both predicted and ground-truth ingredient lists were empty (returning 1.0 for matching empty sets).

*Solution*: Added explicit library installation and imports; added error logging to except blocks.

**Syntax Errors in Configuration**: Duplicate parameter names in TrainingArguments caused runtime failures.

*Solution*: Code review and removal of duplicate parameters.

## 9. Iterative Development Process

**Initial Attempt (CLIP+GPT-2)**: The first implementation worked but produced low-quality recipes with poor ingredient coverage. ROUGE-L and BLEU scores were significantly below FIRE baseline.

**Qwen2.5-VL First Attempt**: Initial configuration with LoRA rank=8 and 600 training samples produced:
- ROUGE-L: 18.44% (vs FIRE's 21.29%)
- SacreBLEU: 4.82% (vs FIRE's 6.02%)
- Ingredient F1: 14.34% (vs FIRE's 49.27%)

**Improvements Applied**:
1. Increased LoRA rank: 8 → 16 (more adaptation capacity)
2. Added target modules: `q_proj, v_proj` → `q_proj, k_proj, v_proj, o_proj`
3. Increased training samples: 600 → 1,000
4. Increased epochs: 3 → 5
5. Adjusted learning rate: 2e-4 → 1e-4 (more stable training)

**Key Insight**: The low Ingredient F1 score indicates the model struggles with ingredient extraction, likely because:
1. Many ingredients are not visually apparent
2. The model may not be learning the structured ingredient format
3. Limited training data compared to FIRE's 259K samples

## 10. Implementation Details

**Codebase Structure**:
- `Food_to_receipe.ipynb`: CLIP+GPT-2 implementation
- `Food_to_receipe_Qwen.ipynb`: Qwen2.5-VL implementation
- Both notebooks include data loading, model training, evaluation, and visualization

**Libraries Used**:
- PyTorch for model implementation
- Hugging Face Transformers for pre-trained models
- PEFT for LoRA implementation
- rouge-score and sacrebleu for evaluation metrics

**Reproducibility**: Random seeds were fixed (42) for dataset splitting to ensure reproducible results.
