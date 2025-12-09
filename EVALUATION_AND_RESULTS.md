# Evaluation and Results: How Did We Measure Success?

## How Did We Measure Success?

### Quantitative Metrics

We employed a multi-faceted evaluation framework to measure success across different dimensions of recipe generation quality:

1. **ROUGE-L (Longest Common Subsequence)**
   - **Purpose**: Measures overall semantic similarity between generated and ground truth recipes
   - **Range**: 0-1 (higher is better)
   - **Why it matters**: Captures whether the model generates recipes that are conceptually similar to the target, even if word-for-word different

2. **BLEU Score**
   - **Purpose**: Measures n-gram overlap between generated and reference text
   - **Range**: 0-1 (higher is better)
   - **Why it matters**: Evaluates exact word/phrase matching, important for recipe accuracy (ingredient names, measurements)

3. **Ingredient F1 Score**
   - **Purpose**: Measures how well the model identifies correct ingredients
   - **Range**: 0-1 (higher is better)
   - **Why it matters**: Critical for recipe usability - wrong ingredients make recipes unusable

4. **Instruction F1 Score**
   - **Purpose**: Measures how well the model generates correct cooking steps
   - **Range**: 0-1 (higher is better)
   - **Why it matters**: Ensures cooking instructions are accurate and complete

5. **Training Loss**
   - **Purpose**: Measures how well the model learns during training
   - **Metric**: Cross-entropy loss (lower is better)
   - **Why it matters**: Indicates whether the model is learning the image-to-recipe mapping

### Qualitative Evaluation

- **Manual inspection**: Generated recipes were examined for:
  - Coherence and readability
  - Logical cooking steps
  - Appropriate ingredient lists
  - Formatting and structure
  - Visual-to-recipe alignment (does the recipe match what's in the image?)

---

## What Experiments Were Used?

### Experiment 1: Fine-Tuning Qwen2.5-VL-3B on Recipe Dataset

**Setup:**
- **Model**: Qwen2.5-VL-3B-Instruct (pre-trained vision-language model)
- **Dataset**: 1,000 image-recipe pairs (subset of 13,471 available)
- **Train/Test Split**: 80/20 (800 training, 200 test)
- **Hyperparameters**:
  - Learning rate: 5e-5
  - Batch size: 1 per GPU (effective batch size: 128 via gradient accumulation)
  - Epochs: 5
  - Optimizer: AdamW
  - Mixed precision: FP16/BF16
  - Gradient checkpointing: Enabled

**Rationale for Hyperparameters:**
- **Small batch size (1)**: Memory constraints with 3.75B parameter model
- **Large gradient accumulation (128)**: Simulates larger batch size for stable training
- **Learning rate (5e-5)**: Standard for fine-tuning large language models (not too high to avoid catastrophic forgetting, not too low to allow learning)
- **5 epochs**: Balance between learning and overfitting risk with limited data

**Training Process:**
- Model was fine-tuned end-to-end (all parameters trainable)
- Training loss monitored per step
- Validation loss computed periodically
- Model checkpoint saved after training

### Experiment 2: Evaluation on Test Set

**Setup:**
- **Test Set**: 50 samples (subset of 200 test samples for computational efficiency)
- **Generation Parameters**:
  - Temperature: 0.7 (balanced creativity/consistency)
  - Top-p: 0.9 (nucleus sampling)
  - Max new tokens: 1024
  - Repetition penalty: 1.1

**Evaluation Process:**
- For each test sample:
  1. Load food image
  2. Generate recipe using fine-tuned model
  3. Compare generated recipe to ground truth
  4. Compute all four metrics (ROUGE-L, BLEU, Ingredient F1, Instruction F1)
- Aggregate statistics computed (mean, std, median, min, max)

### Experiment 3: Baseline Comparison (Implicit)

**What we should have done but didn't explicitly:**
- Compare fine-tuned model to base (pre-trained) Qwen2.5-VL-3B
- Compare to other approaches (CLIP + separate text generator)
- Ablation studies (what if we only fine-tuned certain layers?)

**Why this is a limitation:**
- Without baseline comparison, we cannot claim the fine-tuning actually improved performance
- The metrics we report might be similar to or worse than the base model

---

## Quantitative Results

### Training Performance

**Loss Reduction:**
- **Initial Training Loss**: 5.72
- **Final Training Loss**: 2.74
- **Reduction**: 52.1% decrease
- **Initial Validation Loss**: 4.03
- **Final Validation Loss**: 2.89
- **Reduction**: 28.3% decrease

**Analysis:**
- The model clearly learned during training (loss decreased significantly)
- Training loss lower than validation loss indicates some overfitting, but gap is reasonable (2.74 vs 2.89)
- The 52% reduction in training loss suggests the model learned meaningful patterns
- However, validation loss reduction (28%) is smaller, indicating the model may have memorized some training examples

### Evaluation Metrics (Test Set, n=50)

**ROUGE-L:**
- **Mean**: 0.162 ± 0.050
- **Median**: 0.168
- **Range**: 0.018 - 0.271

**BLEU:**
- **Mean**: 0.026 ± 0.016
- **Median**: 0.026
- **Range**: 0.000 - 0.065

**Ingredient F1:**
- **Mean**: 0.138 ± 0.074
- **Median**: 0.134
- **Range**: 0.000 - 0.265

**Instruction F1:**
- **Mean**: 0.205 ± 0.086
- **Median**: 0.222
- **Range**: 0.000 - 0.409

### Critical Analysis of Quantitative Results

**ROUGE-L (0.162):**
- **Interpretation**: Low to moderate similarity
- **Context**: ROUGE-L of 0.16 means only ~16% of the longest common subsequence matches
- **Comparison**: For reference, good summarization systems achieve ROUGE-L > 0.40
- **Conclusion**: The model generates recipes that are somewhat related to ground truth, but with significant differences

**BLEU (0.026):**
- **Interpretation**: Very low n-gram overlap
- **Context**: BLEU of 0.026 means only 2.6% of n-grams match exactly
- **Comparison**: Machine translation systems typically achieve BLEU > 0.30 for good quality
- **Conclusion**: The model generates recipes with very different wording than ground truth. This could be:
  - **Bad**: Model is hallucinating or not following the image
  - **Good**: Model is generating valid recipes but with different phrasing (there are many ways to write the same recipe)

**Ingredient F1 (0.138):**
- **Interpretation**: Low ingredient accuracy
- **Context**: Only 13.8% of ingredients match on average
- **Conclusion**: **This is concerning** - recipes with wrong ingredients are not useful. The model struggles to identify correct ingredients from images.

**Instruction F1 (0.205):**
- **Interpretation**: Moderate instruction accuracy
- **Context**: 20.5% of cooking steps match
- **Conclusion**: Better than ingredients, but still low. The model can generate some correct steps but misses many.

**Overall Assessment:**
- **Metrics are low** compared to typical NLP benchmarks
- **However**, recipe generation is harder than translation/summarization because:
  - Many valid ways to write the same recipe
  - Visual-to-text is more challenging than text-to-text
  - Recipes require domain knowledge (cooking methods, ingredient relationships)

---

## Qualitative Results

### Example 1: Successful Generation

**Input Image**: Berries and Rum (cocktail)
**Generated Recipe**:
```
RECIPE: Berry Smoothie

INGREDIENTS:
['1/4 cup frozen blackberries', '1/2 cup frozen raspberries', 
 '1/2 cup frozen blueberries', '2 teaspoons lemon juice']

INSTRUCTIONS:
Blend all ingredients in blender until smooth.
Pour into glasses and serve immediately.
```

**Analysis:**
- ✅ **Coherent**: Recipe makes sense
- ✅ **Structured**: Has title, ingredients, instructions
- ⚠️ **Accuracy**: Generated "Berry Smoothie" but ground truth is "Berries and Rum" (cocktail, not smoothie)
- ⚠️ **Missing ingredients**: No alcohol mentioned (rum), missing key ingredient
- ✅ **Formatting**: Clean, readable format

**Conclusion**: Model generates coherent recipes but may misinterpret the dish type or miss key ingredients.

### Example 2: Partially Successful Generation

**Input Image**: Smoke-Roasted Rustic Root Vegetables
**Generated Recipe**: Spicy Roasted Potatoes

**Analysis:**
- ✅ **Related**: Both are roasted vegetable dishes
- ⚠️ **Specificity**: Model identified "roasted" but missed "smoke-roasted" and "root vegetables" (generated just "potatoes")
- ⚠️ **Simplification**: Model simplified the dish to a more generic version

**Conclusion**: Model captures general cooking method but loses specific details.

### Patterns Observed

1. **Model generates coherent recipes**: Outputs are readable and follow recipe structure
2. **Model captures cooking methods**: Often identifies whether food is roasted, grilled, baked, etc.
3. **Model struggles with specificity**: Tends to generate generic recipes rather than specific dishes
4. **Model misses key ingredients**: Frequently omits important ingredients visible in the image
5. **Model generates plausible recipes**: Even when wrong, recipes are usually cookable (not complete nonsense)

---

## Did We Succeed? Did We Fail?

### Partial Success with Significant Limitations

**What Succeeded:**

1. **Technical Implementation**
   - ✅ Successfully fine-tuned a 3.75B parameter vision-language model
   - ✅ Model learned during training (loss decreased 52%)
   - ✅ Model generates coherent, structured recipes
   - ✅ System works end-to-end (image → recipe)

2. **Recipe Quality (Partial)**
   - ✅ Generated recipes are readable and well-formatted
   - ✅ Recipes often capture cooking methods
   - ✅ Recipes are generally plausible (not random text)
   - ⚠️ Recipes often miss key ingredients
   - ⚠️ Recipes may misidentify dish types

3. **Learning Evidence**
   - ✅ Training loss decreased significantly
   - ✅ Model generates recipes in correct format
   - ✅ Model associates images with recipe structure

**What Failed or Is Limited:**

1. **Accuracy**
   - ❌ Low ingredient F1 (0.138) - many wrong ingredients
   - ❌ Low BLEU (0.026) - very different wording from ground truth
   - ❌ Low ROUGE-L (0.162) - limited semantic similarity

2. **Missing Experiments**
   - ❌ No baseline comparison (base model vs. fine-tuned)
   - ❌ No ablation studies
   - ❌ No comparison to other approaches
   - ❌ Limited test set (50 samples, not full 200)

3. **Evaluation Rigor**
   - ❌ No human evaluation (only automatic metrics)
   - ❌ No analysis of failure cases
   - ❌ No analysis of what the model learned vs. what it didn't

---

## Why Did We Get These Results?

### Evidence-Based Analysis

**Why Metrics Are Low:**

1. **Task Difficulty**
   - **Evidence**: Recipe generation from images is inherently difficult
   - **Reasoning**: 
     - Visual features (color, texture) don't directly map to ingredient lists
     - Many ingredients may not be visible (spices, seasonings, preparation steps)
     - Same dish can be written many different ways
   - **Support**: Even state-of-the-art vision-language models struggle with fine-grained visual understanding

2. **Limited Training Data**
   - **Evidence**: Only 1,000 training samples for a 3.75B parameter model
   - **Reasoning**: 
     - Large models typically need 10,000+ examples for good fine-tuning
     - With only 1,000 examples, model may not see enough variety
     - Model may overfit to training examples
   - **Support**: Validation loss (2.89) higher than training loss (2.74) suggests some overfitting

3. **Model Architecture Limitations**
   - **Evidence**: Qwen2.5-VL is a general-purpose vision-language model
   - **Reasoning**:
     - Not specifically designed for recipe generation
     - May not have learned fine-grained visual understanding needed for ingredients
     - Vision encoder may not capture subtle food details
   - **Support**: Ingredient F1 (0.138) is lowest metric, suggesting visual understanding is the bottleneck

4. **Evaluation Metric Limitations**
   - **Evidence**: BLEU and ROUGE penalize different phrasings of same recipe
   - **Reasoning**:
     - "Add salt" vs. "Season with salt" are both correct but get low BLEU
     - Model may generate valid recipes that just use different wording
   - **Support**: Qualitative inspection shows recipes are often reasonable despite low metrics

**Why Some Things Worked:**

1. **Pre-trained Model Quality**
   - **Evidence**: Model generates coherent text and follows structure
   - **Reasoning**: Qwen2.5-VL was pre-trained on large text corpus, so it knows how to write
   - **Support**: Generated recipes are well-formatted and readable

2. **Fine-Tuning Effectiveness**
   - **Evidence**: Training loss decreased 52%
   - **Reasoning**: Model learned to associate images with recipe format
   - **Support**: Model generates recipes in correct structure (title, ingredients, instructions)

3. **Instruction Tuning**
   - **Evidence**: Model follows prompts to generate structured recipes
   - **Reasoning**: Qwen2.5-VL is instruction-tuned, so it understands format requirements
   - **Support**: Output consistently follows the requested recipe format

---

## Detailed Analysis of Decisions

### Decision 1: Using Qwen2.5-VL-3B Instead of Building from Scratch

**Decision**: Fine-tune pre-trained Qwen2.5-VL-3B-Instruct

**Rationale**:
- Pre-trained models have learned general vision-language understanding
- Building from scratch would require massive datasets and compute
- Fine-tuning is standard practice for domain adaptation

**Evidence Supporting Decision**:
- ✅ Model generates coherent recipes (qualitative evidence)
- ✅ Training loss decreased (quantitative evidence)
- ✅ Model follows instruction format (qualitative evidence)

**Evidence Against Decision**:
- ❌ Low accuracy metrics suggest model may not have learned enough
- ❌ Generic recipes suggest model relies more on pre-training than fine-tuning
- ❌ Missing ingredients suggest vision understanding is insufficient

**Conclusion**: Decision was reasonable, but results suggest we needed:
- More training data
- Better vision encoder fine-tuning
- Task-specific architecture modifications

### Decision 2: Training on 1,000 Samples Instead of Full Dataset

**Decision**: Use subset of 1,000 samples for training

**Rationale**:
- Computational constraints (training time, memory)
- Initial testing before full training
- 1,000 samples should be enough for initial fine-tuning

**Evidence Supporting Decision**:
- Training completed successfully
- Model learned (loss decreased)
- Faster iteration for experimentation

**Evidence Against Decision**:
- ❌ Low metrics suggest insufficient data
- ❌ Validation loss higher than training loss (overfitting)
- ❌ Model may have memorized rather than generalized

**Conclusion**: **This was likely a mistake**. With 13,471 samples available, using only 1,000 significantly limited learning. Evidence:
- Large gap between training and validation loss suggests overfitting
- Low test metrics suggest poor generalization
- Model likely needed more diverse examples

**What We Should Have Done**:
- Train on full dataset (or at least 5,000+ samples)
- Use data augmentation
- Monitor validation metrics more closely

### Decision 3: End-to-End Fine-Tuning vs. Frozen Vision Encoder

**Decision**: Fine-tune all parameters (end-to-end)

**Rationale**:
- Vision encoder may need to learn food-specific features
- End-to-end training allows full adaptation
- Standard practice for vision-language models

**Evidence Supporting Decision**:
- Model generates recipes that relate to images (qualitative)
- Training loss decreased (quantitative)

**Evidence Against Decision**:
- ❌ Low ingredient F1 suggests vision understanding is still poor
- ❌ May have caused overfitting (limited data + many parameters)
- ❌ Training was slow and memory-intensive

**Alternative We Should Have Tested**:
- Freeze vision encoder, only fine-tune language model
- Compare results to see if vision encoder fine-tuning helped
- This would be an ablation study (missing from our experiments)

**Conclusion**: Decision was reasonable but unvalidated. We should have compared to frozen encoder baseline.

### Decision 4: Evaluation Metrics Choice

**Decision**: Use ROUGE-L, BLEU, Ingredient F1, Instruction F1

**Rationale**:
- Standard NLP metrics (ROUGE, BLEU)
- Task-specific metrics (Ingredient/Instruction F1)
- Comprehensive evaluation

**Evidence Supporting Decision**:
- Metrics capture different aspects (overall similarity, exact match, ingredient accuracy, instruction accuracy)
- Standard metrics allow comparison to other work

**Evidence Against Decision**:
- ❌ Metrics may not capture recipe quality well (many valid ways to write recipes)
- ❌ No human evaluation (gold standard for generation tasks)
- ❌ Metrics don't measure visual alignment (does recipe match image?)

**What We Should Have Added**:
- Human evaluation (rate recipes 1-5 for quality, accuracy, usefulness)
- Visual alignment metric (does generated recipe match what's in the image?)
- Cookability metric (can someone actually follow this recipe?)

**Conclusion**: Metrics were reasonable but incomplete. Human evaluation would have provided crucial validation.

---

## Comparison to Baselines (What We Should Have Done)

### Missing Baseline: Base Model Performance

**What We Should Have Measured**:
- Generate recipes with pre-trained Qwen2.5-VL-3B (no fine-tuning)
- Compare metrics to fine-tuned model
- This would show if fine-tuning actually helped

**Why This Matters**:
- Without this comparison, we cannot claim fine-tuning improved performance
- The low metrics might be similar to base model (meaning fine-tuning didn't help)
- Or base model might be worse (meaning fine-tuning did help, but not enough)

**Estimated Baseline Performance** (based on general vision-language models):
- Base model would likely generate generic recipes not specific to images
- Base model might have similar or worse metrics
- But we cannot know without testing

### Missing Baseline: Alternative Approaches

**What We Should Have Compared**:
1. **CLIP + GPT-2**: Use CLIP to identify food, then GPT-2 to generate recipe
2. **CLIP + Recipe Retrieval**: Use CLIP to identify food, then retrieve similar recipes
3. **Fine-tuned CLIP + Fine-tuned GPT-2**: Separate models fine-tuned for food

**Why This Matters**:
- Would show if end-to-end approach is better than two-stage
- Would identify which component is the bottleneck
- Would provide context for our results

---

## Conclusions and Justification

### Did We Succeed?

**Partial Success with Significant Limitations**

**Success Criteria Met:**
1. ✅ **Technical Feasibility**: System works end-to-end
2. ✅ **Learning Evidence**: Model learned during training
3. ✅ **Output Quality (Partial)**: Generates coherent, structured recipes

**Success Criteria Not Met:**
1. ❌ **Accuracy**: Low metrics across all dimensions
2. ❌ **Reliability**: Missing key ingredients, misidentifying dishes
3. ❌ **Rigor**: Missing baseline comparisons and ablation studies

### Why These Results?

**Primary Reasons for Limited Success:**

1. **Insufficient Training Data**
   - **Evidence**: Only 1,000 samples, validation loss higher than training loss
   - **Impact**: Model overfitted, poor generalization
   - **Fix**: Train on full 13,471 sample dataset

2. **Task Difficulty**
   - **Evidence**: Low metrics even with fine-tuning
   - **Impact**: Recipe generation from images is inherently hard
   - **Fix**: May need task-specific architecture or more sophisticated approach

3. **Evaluation Limitations**
   - **Evidence**: Only automatic metrics, no human evaluation
   - **Impact**: May underestimate model quality (recipes might be valid but different wording)
   - **Fix**: Add human evaluation, visual alignment metrics

4. **Missing Experiments**
   - **Evidence**: No baseline comparison, no ablation studies
   - **Impact**: Cannot claim fine-tuning improved performance
   - **Fix**: Compare to base model, test frozen encoder, test different architectures

### Is This a Strong Project?

**Honest Assessment: Partially Strong**

**Strengths:**
- ✅ Successfully fine-tuned large vision-language model
- ✅ Comprehensive evaluation metrics
- ✅ Clear evidence of learning (loss reduction)
- ✅ Working end-to-end system

**Weaknesses:**
- ❌ Low accuracy metrics
- ❌ Missing baseline comparisons
- ❌ Limited training data usage
- ❌ No ablation studies
- ❌ No human evaluation

**What Would Make It Stronger:**
1. **Baseline Comparison**: Show fine-tuned model outperforms base model
2. **Full Dataset Training**: Use all 13,471 samples
3. **Ablation Studies**: Test frozen encoder, different architectures
4. **Human Evaluation**: Validate that recipes are actually useful
5. **Failure Analysis**: Understand when and why model fails
6. **Comparison to Alternatives**: Compare to CLIP+GPT-2 approach

### Final Verdict

**The project demonstrates technical competence and a working system, but the results are limited by insufficient data usage, missing baselines, and low accuracy metrics. The approach is sound, but the execution needs improvement to be considered a strong project.**

**Key Evidence:**
- Training loss decreased 52% (learning occurred)
- But test metrics are low (poor generalization)
- Missing baseline comparison (cannot claim improvement)
- Limited data usage (only 7% of available data)

**Recommendation for Improvement:**
1. Train on full dataset
2. Compare to base model baseline
3. Add human evaluation
4. Analyze failure cases
5. Test alternative architectures

---

## Appendix: Detailed Metric Analysis

### Metric Distributions

**ROUGE-L Distribution:**
- Mean: 0.162, Std: 0.050
- **Interpretation**: Moderate variance, most samples score 0.10-0.25
- **Outliers**: One sample scored 0.018 (very poor), one scored 0.271 (relatively good)
- **Conclusion**: Inconsistent performance across samples

**BLEU Distribution:**
- Mean: 0.026, Std: 0.016
- **Interpretation**: Very low scores with low variance
- **Conclusion**: Consistently poor n-gram matching

**Ingredient F1 Distribution:**
- Mean: 0.138, Std: 0.074
- **Interpretation**: Low scores with high variance
- **Outliers**: Several samples scored 0.0 (completely wrong ingredients)
- **Conclusion**: Model sometimes gets ingredients right, often completely wrong

**Instruction F1 Distribution:**
- Mean: 0.205, Std: 0.086
- **Interpretation**: Moderate scores with high variance
- **Best case**: One sample scored 0.409 (40% match)
- **Conclusion**: Instructions are more accurate than ingredients, but still inconsistent

### Correlation Analysis

**Expected Correlations:**
- ROUGE-L and BLEU should correlate (both measure similarity)
- Ingredient F1 and Instruction F1 should correlate (good recipes have both)

**What We Should Measure:**
- Correlation matrix between metrics
- This would show if metrics agree or disagree
- High correlation = metrics measure similar things
- Low correlation = metrics capture different aspects

**Note**: We computed correlation in visualization but didn't analyze it in detail. This is a missed opportunity for deeper understanding.










