# src/inference.py

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
from prepare_data import LABEL_NAMES, CATEGORIES_STR

BASE_MODEL  = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
ADAPTER_DIR = "../models/lora-arxiv"

# ── Load base model + adapter ──────────────────────────────────────────────────
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)

print("Loading base model...")
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    quantization_config=bnb_config,
    device_map="auto",
    dtype=torch.float16,
)

print("Loading LoRA adapter...")
model = PeftModel.from_pretrained(model, ADAPTER_DIR)
model.eval()  # disable dropout for inference
print("Ready!\n")

# ── Inference function ─────────────────────────────────────────────────────────
def classify_paper(abstract: str) -> str:
    prompt = f"""### Classify this arXiv paper into one of:
{CATEGORIES_STR}

### Abstract:
{abstract[:800].strip()}

### Category:
"""
    # Note: prompt ends WITHOUT a label — model must generate it
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=10,    # category name is short, don't need more
            temperature=0.1,      # low temperature = more confident/deterministic
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )

    # Decode only the newly generated tokens (not the input prompt)
    generated = outputs[0][inputs['input_ids'].shape[1]:]
    result = tokenizer.decode(generated, skip_special_tokens=True).strip()

    # Extract just the first word/token (e.g. "cs.DS" from "cs.DS\n\n###...")
    predicted = result.split()[0] if result.split() else "unknown"
    return predicted


# ── Test on a few samples ──────────────────────────────────────────────────────
if __name__ == "__main__":
    from datasets import load_dataset

    ds = load_dataset('ccdv/arxiv-classification')
    test_samples = ds['test'].select(range(10))  # try first 10 test samples

    correct = 0
    for i, sample in enumerate(test_samples):
        true_label = LABEL_NAMES[sample['label']]
        predicted   = classify_paper(sample['text'])
        match = "✓" if predicted == true_label else "✗"
        print(f"[{match}] True: {true_label:<10} Predicted: {predicted}")
        if predicted == true_label:
            correct += 1

    print(f"\nAccuracy on 10 samples: {correct}/10")
