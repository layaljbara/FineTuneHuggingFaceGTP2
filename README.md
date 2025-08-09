# GPT-2 Fine-Tuning for Recipe Generator

A compact workflow to fine-tune GPT-2 on the **Epicurious** recipes corpus and generate ingredient-aware recipe text.  
Includes data loading, tokenization, block-wise language modeling, TensorFlow/Keras training, and text generation with the fine-tuned model.

**Tools and Frameworks:**  
- Python  
- TensorFlow / Keras (TF models)  
- Hugging Face Transformers + Datasets  

---

## Dataset & Preprocessing

- **Raw data**: plain-text recipe splits (`train/test/validation`) produced from Epicurious.  
- **Tokenizer**: `AutoTokenizer` for GPT-2.  
- **Tokenization details**:
  - Append a `\n` to each example to preserve recipe line breaks.
  - Truncate to a max length (e.g., 512) for batching.
- **Block grouping**:
  - Concatenate tokenized text then split into fixed-length **blocks** (`block_size`).
  - Set `labels = input_ids` for **causal language modeling** (next-token prediction).


---

## Model & Training

- **Base model**: `TFAutoModelForCausalLM.from_pretrained("gpt2")` (or load a prior checkpoint).  
- **Embedding resize**: `model.resize_token_embeddings(len(tokenizer))`.  
- **Data pipelines**: `DefaultDataCollator` → `to_tf_dataset(...)` for efficient `tf.data` input.  
- **Objective**: standard causal LM loss (internally computed by the model).  
- **Training loop**: `model.compile(optimizer="adam")` → `model.fit(...)` with validation.  
- **Checkpoints**: `model.save_pretrained("output")` for reuse in inference.
---

## Inference (Generation)

Use the fine-tuned checkpoint to generate recipes from an ingredient prompt:

1. **Load** model from `output` and GPT-2 tokenizer.  
2. **Encode** a prompt containing a title + ingredient list (and optionally partial steps).  
3. **Generate** with `model.generate(...)` (tune `max_length`, decoding strategy).  
4. **Decode** tokens back to text and display.

---

## Results:

- Ingredients in the list are often **referenced in directions** and roughly in order of appearance.  
- **Titles** align with ingredients.  
- **Grammar** and fluency are noticeably good.

**Limitations / Observations:**
- Signs of **memorization** (copied titles/ingredient lists) → classic **overfitting**.  
- Dataset is **small for 124M** parameters; scaling data helps.  
- Generic or repetitive phrasing can still appear.

---

## Future Improvements

- **Data scale & quality**: more (and more diverse) recipes; de-duplication; augment with structured fields (cuisine, methods).  
- **Regularization**: dropout, early stopping, learning-rate scheduling.  
- **Decoding**: experiment with **top-k**, **nucleus (top-p)**, and **beam search**.  
- **Structure cues**: explicit tokens for sections (e.g., `<INGR>`, `<DIR>`).  
- **Modeling**: consider larger or instruction-tuned LMs with parameter-efficient finetuning.

---

## Parameter-Efficient & Resource-Aware Finetuning

Modern models can exceed single-GPU memory. Two practical techniques:

- **Quantization**: compress weights (e.g., 8-bit/4-bit) to fit on fewer GPUs.  
- **LoRA (Low-Rank Adaptation)**: train small, low-rank adapters instead of all weights.

**QLoRA** combines both: quantized base weights + LoRA adapters → **faster**, **cheaper** finetuning on modest hardware.

---

## Repository Layout (Scripts)

- **`Train_Recipes.py`** — Fine-tunes GPT-2 on Epicurious using TF/Keras + Hugging Face 
  - Loads tokenizer/config, tokenizes and groups text, builds `tf.data` pipelines, trains, and saves to `output/`.  
- **`Generate_Recipes.py`** — Loads the checkpoint from `output/` and generates recipe text from a prompt.

---


