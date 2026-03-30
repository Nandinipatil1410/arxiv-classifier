# src/train.py

import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, TaskType
from trl import SFTConfig, SFTTrainer
from prepare_data import get_formatted_dataset, LABEL_NAMES

# ── 0. Sanity check GPU ────────────────────────────────────────────────────────
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

# ── 1. Config — change these to experiment ────────────────────────────────────
MODEL_NAME   = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
OUTPUT_DIR   = "../models/lora-arxiv"
MAX_SEQ_LEN  = 512       # max tokens per sample (keep low for 6GB VRAM)
BATCH_SIZE   = 4         # samples per gradient step
GRAD_ACCUM   = 4         # effective batch = BATCH_SIZE × GRAD_ACCUM = 16
EPOCHS       = 2
LR           = 2e-4

# ── 2. Load dataset ────────────────────────────────────────────────────────────
ds = get_formatted_dataset()

# ── 3. 4-bit Quantization config ──────────────────────────────────────────────
# This shrinks the model from ~4.4GB → ~1.1GB in VRAM
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,                        # load weights in 4-bit
    bnb_4bit_quant_type="nf4",               # NormalFloat4 — best quality for LLMs
    bnb_4bit_compute_dtype=torch.bfloat16,   # compute in bf16 — more stable, RTX 4050 supports it
    bnb_4bit_use_double_quant=True,          # quantize the quantization constants too (saves ~0.4GB)
)

# ── 4. Load tokenizer ──────────────────────────────────────────────────────────
print("\nLoading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token      # TinyLlama has no pad token by default
tokenizer.padding_side = "right"               # pad on right for causal LM training
tokenizer.model_max_length = MAX_SEQ_LEN       # truncate sequences to this length

# ── 5. Load model in 4-bit ────────────────────────────────────────────────────
print("Loading model in 4-bit...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=bnb_config,
    device_map="auto",                         # automatically place on GPU
    torch_dtype=torch.bfloat16,
)
model.config.use_cache = False                 # must disable for gradient checkpointing

# ── 6. LoRA config ────────────────────────────────────────────────────────────
lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=8,                                       # rank — controls adapter size
    lora_alpha=16,                             # scaling = lora_alpha / r = 2.0
    lora_dropout=0.05,
    target_modules=["q_proj", "v_proj"],       # apply LoRA to attention projections
    bias="none",
)

# Wrap the model with LoRA adapters
model = get_peft_model(model, lora_config)

# Show how many parameters are actually trainable
model.print_trainable_parameters()
# Expected output: ~0.5% of total params — that's the magic of LoRA!

# ── 7. Training config ────────────────────────────────────────────────────────
sft_config = SFTConfig(
    output_dir=OUTPUT_DIR,
    num_train_epochs=EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=GRAD_ACCUM,
    learning_rate=LR,
    bf16=True,
    logging_steps=50,
    save_strategy="epoch",
    eval_strategy="epoch",
    load_best_model_at_end=True,
    report_to="none",
    dataset_text_field="text",
)

# ── 8. Trainer ────────────────────────────────────────────────────────────────
trainer = SFTTrainer(
    model=model,
    args=sft_config,
    train_dataset=ds['train'],
    eval_dataset=ds['validation'],
    processing_class=tokenizer,
)

# ── 9. Train! ─────────────────────────────────────────────────────────────────
print("\nStarting training...")
trainer.train(resume_from_checkpoint=True)

# ── 10. Save ──────────────────────────────────────────────────────────────────
print(f"\nSaving adapter to {OUTPUT_DIR}")
trainer.model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
print("Done! ✓")