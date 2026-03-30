# src/evaluate.py
import warnings
warnings.filterwarnings("ignore")

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
from datasets import load_dataset
from prepare_data import LABEL_NAMES, CATEGORIES_STR
import json

BASE_MODEL  = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
ADAPTER_DIR = "../models/lora-arxiv"

# ── Load model ─────────────────────────────────────────────────────────────────
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)

print("Loading model...")
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    quantization_config=bnb_config,
    device_map="auto",
    dtype=torch.float16,
)
model = PeftModel.from_pretrained(model, ADAPTER_DIR)
model.eval()
print("Ready!\n")

# ── Inference function ─────────────────────────────────────────────────────────
def classify_paper(abstract: str) -> str:
    prompt = f"""### Classify this arXiv paper into one of:
{CATEGORIES_STR}

### Abstract:
{abstract[:800].strip()}

### Category:
"""
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=10,
            temperature=0.1,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )
    generated = outputs[0][inputs['input_ids'].shape[1]:]
    result = tokenizer.decode(generated, skip_special_tokens=True).strip()
    return result.split()[0] if result.split() else "unknown"

# ── Evaluate on full test set ──────────────────────────────────────────────────
print("Loading test set...")
ds = load_dataset('ccdv/arxiv-classification')
test = ds['test']

correct = 0
total   = len(test)

# Confusion matrix: confusion[true][pred] = count
confusion = {l: {l2: 0 for l2 in LABEL_NAMES} for l in LABEL_NAMES}

print(f"Running inference on {total} samples...\n")

for i, sample in enumerate(test):
    true_label = LABEL_NAMES[sample['label']]
    predicted  = classify_paper(sample['text'])

    if predicted in LABEL_NAMES:
        confusion[true_label][predicted] += 1
    if predicted == true_label:
        correct += 1

    # Progress update every 100 samples
    if (i + 1) % 100 == 0:
        running_acc = correct / (i + 1) * 100
        print(f"  [{i+1}/{total}]  Running accuracy: {running_acc:.1f}%")

# ── Results ────────────────────────────────────────────────────────────────────
final_acc = correct / total * 100
print(f"\n{'='*50}")
print(f"Final Accuracy: {correct}/{total} = {final_acc:.1f}%")
print(f"{'='*50}\n")

# Per-class accuracy
print("Per-class accuracy:")
print(f"{'Category':<12} {'Correct':>8} {'Total':>8} {'Acc':>8}")
print("-" * 40)
for label in LABEL_NAMES:
    total_for_label   = sum(confusion[label].values())
    correct_for_label = confusion[label][label]
    acc = correct_for_label / total_for_label * 100 if total_for_label > 0 else 0
    print(f"{label:<12} {correct_for_label:>8} {total_for_label:>8} {acc:>7.1f}%")

# Save results to file
results = {
    "accuracy": final_acc,
    "correct": correct,
    "total": total,
    "confusion_matrix": confusion
}
with open("../models/lora-arxiv/eval_results.json", "w") as f:
    json.dump(results, f, indent=2)

print("\nResults saved to models/lora-arxiv/eval_results.json")