# Metrics Comparison: Our Approach vs. FIRE Paper

## Overview

This document compares our evaluation metrics with those used in the FIRE (Food Image to REcipe generation) paper.

**FIRE Paper**: Chhikara, P., Chaurasia, D., Jiang, Y., Masur, O., & Ilievski, F. (2023). "FIRE: Food Image to REcipe generation." arXiv preprint arXiv:2308.14391.  
**URL**: https://arxiv.org/pdf/2308.14391  
**Code**: https://github.com/prateekchhikara/FIRE

---

## FIRE Paper Methodology

### Architecture
FIRE uses a **three-stage pipeline**:
1. **Title Generation**: BLIP model (Salesforce/blip-image-captioning-base)
2. **Ingredient Extraction**: Vision Transformer (ViT) with attention-based decoder
3. **Recipe Generation**: T5 model (220M parameters, base version)

### Evaluation Metrics Used in FIRE

According to the paper, FIRE evaluates using:

1. **Set Metrics for Ingredient Extraction**
   - Measures how well ingredients are extracted from images
   - Compares predicted ingredient sets with ground truth ingredient sets
   - Likely includes precision, recall, and F1 scores for ingredient sets
   - Standard approach for ingredient extraction tasks

2. **Document-Level Metrics for Cooking Instruction Generation**
   - Measures quality of generated cooking instructions
   - Evaluates the full instruction text as a document
   - Likely includes ROUGE and BLEU scores (standard for text generation)

3. **Title Generation Evaluation**
   - Uses **Longest Common Subsequence (LCS)** for validation during training
   - String similarity metrics for title evaluation
   - LCS was chosen because it achieved superior results compared to loss-based validation

---

## Our Evaluation Metrics

### Metrics We Currently Use

1. **ROUGE-L (Longest Common Subsequence)**
   - **Purpose**: Overall semantic similarity between generated and ground truth recipes
   - **Range**: 0-1 (higher is better)
   - **Our Results (CLIP+GPT-2)**: Mean 0.1337 ± 0.0331
   - **Our Results (Qwen)**: Mean 0.1612 ± 0.0442

2. **BLEU Score**
   - **Purpose**: N-gram overlap between generated and actual recipes
   - **Range**: 0-1 (higher is better)
   - **Our Results (CLIP+GPT-2)**: Mean 0.0489 ± 0.0227
   - **Our Results (Qwen)**: Mean 0.0259 ± 0.0165

3. **Ingredient F1 Score**
   - **Purpose**: Measures how well the model identifies correct ingredients
   - **Range**: 0-1 (higher is better)
   - **Our Results (CLIP+GPT-2)**: Mean 1.0 (perfect - likely extraction issue)
   - **Our Results (Qwen)**: Mean 0.1333 ± 0.0655

4. **Instruction F1 Score**
   - **Purpose**: Measures how well the model generates correct cooking steps
   - **Range**: 0-1 (higher is better)
   - **Our Results (CLIP+GPT-2)**: Mean 0.1159 ± 0.0473
   - **Our Results (Qwen)**: Mean 0.2094 ± 0.0660

---

## Detailed Comparison

### 1. Ingredient Evaluation

**FIRE Approach:**
- Uses **set metrics** for ingredient extraction
- Compares ingredient sets (not individual words)
- Likely reports: Precision, Recall, F1 for ingredient sets
- Evaluates the ViT+Decoder component separately

**Our Approach:**
- Uses **Ingredient F1** based on word overlap
- Compares words in ingredient lists
- **Issue**: Our CLIP+GPT-2 results show perfect F1 (1.0), which suggests the extraction function may not be working correctly
- Our Qwen results show more realistic F1 (0.1333)

**Comparison:**
- FIRE's set-based approach is more standard for ingredient extraction
- Our word-based F1 may be less accurate for ingredient comparison
- **Recommendation**: Switch to set-based metrics like FIRE

### 2. Instruction Evaluation

**FIRE Approach:**
- Uses **document-level metrics** (ROUGE, BLEU)
- Evaluates the full instruction text as a complete document
- Evaluates the T5 generation component separately

**Our Approach:**
- Uses **ROUGE-L** (0.1337-0.1612) and **BLEU** (0.0259-0.0489) for overall recipe
- Uses **Instruction F1** (0.1159-0.2094) for instruction-specific evaluation
- Evaluates end-to-end (not component-specific)

**Comparison:**
- Both use ROUGE and BLEU (good alignment)
- FIRE evaluates instructions separately, we evaluate full recipe
- Our approach is more holistic but less granular

### 3. Title Evaluation

**FIRE Approach:**
- Uses **Longest Common Subsequence (LCS)** for title evaluation
- Explicitly evaluates title generation as a separate component
- Uses LCS for validation during training (not just loss)

**Our Approach:**
- Title is included in overall recipe evaluation
- **Missing**: No separate title-specific metrics
- Title quality is captured in overall ROUGE-L/BLEU but not isolated

**Comparison:**
- **Gap**: We don't evaluate titles separately like FIRE
- **Recommendation**: Add title-specific LCS or F1 metrics

---

## Metric Alignment Table

| Metric Category | FIRE Paper | Our Approach | Alignment |
|----------------|------------|--------------|-----------|
| **Title Evaluation** | LCS (Longest Common Subsequence) | Not separately evaluated | ❌ Missing |
| **Ingredient Evaluation** | Set metrics (Precision, Recall, F1) | Word-based F1 | ⚠️ Different approach |
| **Instruction Evaluation** | Document-level (ROUGE, BLEU) | ROUGE-L, BLEU, Instruction F1 | ✅ Similar |
| **Overall Recipe** | Not explicitly reported | ROUGE-L, BLEU | ✅ We have this |
| **Component Separation** | Yes (title, ingredients, instructions) | No (end-to-end) | ⚠️ Different |

---

## What We Should Add Based on FIRE

### 1. Title-Specific Metrics (High Priority)

FIRE explicitly evaluates titles using LCS. We should add:

```python
def compute_title_lcs(generated_title, ground_truth_title):
    """Compute Longest Common Subsequence for title evaluation (like FIRE)."""
    # Implementation of LCS algorithm
    # Returns LCS score normalized by title length
    pass

def compute_title_f1(generated_title, ground_truth_title):
    """Compute F1 score for title words."""
    gen_words = set(generated_title.lower().split())
    gold_words = set(ground_truth_title.lower().split())
    
    intersection = gen_words & gold_words
    precision = len(intersection) / len(gen_words) if gen_words else 0
    recall = len(intersection) / len(gold_words) if gold_words else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    return {
        'title_precision': precision,
        'title_recall': recall,
        'title_f1': f1,
        'title_lcs': compute_lcs(generated_title, ground_truth_title)
    }
```

### 2. Set-Based Ingredient Metrics (High Priority)

FIRE uses set-based metrics. We should enhance our ingredient evaluation:

```python
def compute_ingredient_set_metrics(pred_ingredients, gold_ingredients):
    """Compute set-based metrics like FIRE (more accurate than word-based)."""
    # Normalize ingredients (lowercase, remove punctuation)
    pred_set = set(normalize_ingredient(ing) for ing in pred_ingredients)
    gold_set = set(normalize_ingredient(ing) for ing in gold_ingredients)
    
    intersection = pred_set & gold_set
    
    precision = len(intersection) / len(pred_set) if pred_set else 0
    recall = len(intersection) / len(gold_set) if gold_set else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    return {
        'ingredient_precision': precision,
        'ingredient_recall': recall,
        'ingredient_f1': f1,
        'ingredient_intersection_size': len(intersection),
        'ingredient_pred_size': len(pred_set),
        'ingredient_gold_size': len(gold_set)
    }
```

### 3. Component-Specific Evaluation (Medium Priority)

FIRE evaluates each component separately. We should add:

```python
def evaluate_by_component(generated_recipe, ground_truth_recipe):
    """Evaluate title, ingredients, and instructions separately (like FIRE)."""
    # Parse recipes into components
    gen_title, gen_ingredients, gen_instructions = parse_recipe(generated_recipe)
    gold_title, gold_ingredients, gold_instructions = parse_recipe(ground_truth_recipe)
    
    # Title metrics
    title_metrics = compute_title_lcs(gen_title, gold_title)
    
    # Ingredient set metrics
    ingredient_metrics = compute_ingredient_set_metrics(gen_ingredients, gold_ingredients)
    
    # Instruction document metrics
    instruction_metrics = {
        'rouge_l': compute_rouge_l(gen_instructions, gold_instructions),
        'bleu': compute_bleu(gen_instructions, gold_instructions)
    }
    
    return {
        'title': title_metrics,
        'ingredients': ingredient_metrics,
        'instructions': instruction_metrics
    }
```

---

## Key Differences in Approach

### Architecture Differences

| Aspect | FIRE | Our Approach |
|--------|------|--------------|
| **Model Type** | 3 separate models (pipeline) | 1 unified model (end-to-end) |
| **Title Generation** | BLIP (separate model) | Generated as part of recipe |
| **Ingredient Extraction** | ViT + Decoder (separate model) | Generated as part of recipe |
| **Recipe Generation** | T5 (separate model) | Generated as part of recipe |
| **Training** | Each component trained separately | Single end-to-end training |

### Evaluation Philosophy

| Aspect | FIRE | Our Approach |
|--------|------|--------------|
| **Evaluation Scope** | Component-specific | End-to-end |
| **Granularity** | High (each component) | Medium (overall + ingredients/instructions) |
| **Title Metrics** | Explicit (LCS) | Implicit (in overall metrics) |
| **Ingredient Metrics** | Set-based | Word-based F1 |
| **Instruction Metrics** | Document-level (ROUGE/BLEU) | Document-level (ROUGE/BLEU) + F1 |

---

## Our Results vs. FIRE (Expected)

**Note**: FIRE doesn't report exact numerical results in the abstract, but we can compare approaches:

### Our Results Summary

**Qwen2.5-VL-3B Approach:**
- ROUGE-L: 0.1612 ± 0.0442
- BLEU: 0.0259 ± 0.0165
- Ingredient F1: 0.1333 ± 0.0655
- Instruction F1: 0.2094 ± 0.0660

**CLIP+GPT-2 Approach:**
- ROUGE-L: 0.1337 ± 0.0331
- BLEU: 0.0489 ± 0.0227
- Ingredient F1: 1.0 (likely extraction issue)
- Instruction F1: 0.1159 ± 0.0473

### What FIRE Likely Reports

Based on the paper's methodology:
- **Ingredient Set Metrics**: Precision, Recall, F1 (separate for each)
- **Instruction Document Metrics**: ROUGE scores, BLEU scores
- **Title Metrics**: LCS scores

---

## Recommendations

### Immediate Actions

1. **Fix Ingredient F1 Calculation**
   - Our CLIP+GPT-2 results show perfect F1 (1.0), which is suspicious
   - Switch to set-based metrics like FIRE
   - Add separate precision and recall for ingredients

2. **Add Title Metrics**
   - Implement LCS-based title evaluation
   - Add title F1, precision, recall
   - Report title metrics separately

3. **Enhance Ingredient Metrics**
   - Switch from word-based to set-based comparison
   - Report precision, recall, and F1 separately
   - Add intersection size for interpretability

### Future Improvements

1. **Component-Specific Evaluation**
   - Parse recipes into title, ingredients, instructions
   - Evaluate each component separately
   - Report metrics for each component (like FIRE)

2. **Standardize with FIRE**
   - Align metric definitions with FIRE's approach
   - Use same evaluation methodology for fair comparison
   - Report metrics in same format

3. **Baseline Comparison**
   - Implement FIRE's three-stage approach as baseline
   - Compare our end-to-end approach vs. FIRE's pipeline
   - Identify which approach works better for our dataset

---

## Code Implementation Recommendations

### Add to Evaluation Code

```python
# Enhanced metrics computation (aligned with FIRE)
def compute_fire_style_metrics(generated_recipe, ground_truth_recipe):
    """Compute metrics in FIRE's style: component-specific evaluation."""
    
    # Parse recipes
    gen_title, gen_ingredients, gen_instructions = parse_recipe(generated_recipe)
    gold_title, gold_ingredients, gold_instructions = parse_recipe(ground_truth_recipe)
    
    metrics = {}
    
    # 1. Title metrics (LCS-based, like FIRE)
    metrics['title'] = {
        'lcs': compute_lcs(gen_title, gold_title),
        'f1': compute_title_f1(gen_title, gold_title)
    }
    
    # 2. Ingredient set metrics (like FIRE)
    metrics['ingredients'] = compute_ingredient_set_metrics(
        gen_ingredients, gold_ingredients
    )
    
    # 3. Instruction document metrics (like FIRE)
    metrics['instructions'] = {
        'rouge_l': compute_rouge_l(gen_instructions, gold_instructions),
        'rouge_1': compute_rouge_1(gen_instructions, gold_instructions),
        'rouge_2': compute_rouge_2(gen_instructions, gold_instructions),
        'bleu': compute_bleu(gen_instructions, gold_instructions)
    }
    
    return metrics
```

---

## Conclusion

### What We're Doing Well

1. ✅ Using ROUGE and BLEU for instruction evaluation (aligned with FIRE)
2. ✅ Evaluating end-to-end recipe generation (more realistic)
3. ✅ Using multiple metrics to capture different aspects

### What We Should Improve

1. ❌ Add title-specific evaluation (FIRE uses LCS)
2. ❌ Switch to set-based ingredient metrics (more accurate)
3. ❌ Add component-specific evaluation (better granularity)
4. ❌ Report precision and recall separately (not just F1)

### Key Takeaway

FIRE's component-specific evaluation provides better insights into which parts of recipe generation are working well. While our end-to-end approach is simpler, adding FIRE's granular metrics would help identify bottlenecks and improve our model.

---

## References

[1] Chhikara, P., Chaurasia, D., Jiang, Y., Masur, O., & Ilievski, F. (2023). "FIRE: Food Image to REcipe generation." arXiv preprint arXiv:2308.14391.  
**URL**: https://arxiv.org/pdf/2308.14391  
**GitHub**: https://github.com/prateekchhikara/FIRE

---

## Appendix: FIRE's Training Details (for Reference)

### Title Generation (BLIP)
- Model: Salesforce/blip-image-captioning-base
- Epochs: 20
- Batch size: 24
- Learning rate: 10^-5
- Validation metric: LCS (Longest Common Subsequence)

### Ingredient Extraction (ViT + Decoder)
- Architecture: Vision Transformer with attention-based decoder
- Epochs: 100
- Batch size: 150
- Learning rate: 10^-4 (with 0.01% decay per epoch)
- Image size: 224×224×3
- Vocabulary size: 1488 ingredients
- Embedding size: 512

### Recipe Generation (T5)
- Model: T5-base (220M parameters)
- Epochs: 30
- Batch size: 12
- Learning rate: 3×10^-4
- Optimizer: AdamW
- Max source length: 50
- Max target length: 512
- Generation: Beam search (beams=4, length_penalty=1, repetition_penalty=2.5)
