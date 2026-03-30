# src/prepare_data.py

from datasets import load_dataset

# ── Label map ──────────────────────────────────────────────────────────────────
LABEL_NAMES = [
    'math.AC', 'cs.CV', 'cs.AI', 'cs.SY', 'math.GR',
    'cs.CE',   'cs.PL', 'cs.IT', 'cs.DS', 'cs.NE', 'math.ST'
]

CATEGORIES_STR = ", ".join(LABEL_NAMES)

# ── Prompt template ────────────────────────────────────────────────────────────
def format_prompt(sample):
    """
    Converts a raw dataset sample into a prompt string.
    The model will learn to 'complete' the prompt with the correct label.
    """
    label_name = LABEL_NAMES[sample['label']]

    # We truncate text to ~800 chars so it fits in context window
    abstract = sample['text'][:800].strip()

    prompt = f"""### Classify this arXiv paper into one of:
{CATEGORIES_STR}

### Abstract:
{abstract}

### Category:
{label_name}"""

    return {"text": prompt}


# ── Main ───────────────────────────────────────────────────────────────────────
def get_formatted_dataset():
    print("Loading dataset...")
    ds = load_dataset('ccdv/arxiv-classification')

    print("Formatting prompts...")
    ds = ds.map(format_prompt, remove_columns=['label'])
    # Note: we removed the 'label' column — the label is now baked into the prompt text

    print("Done!")
    print("Sample formatted prompt:\n")
    print(ds['train'][0]['text'])
    print("\n" + "─"*60)
    print(f"Train: {len(ds['train'])} | Val: {len(ds['validation'])} | Test: {len(ds['test'])}")

    return ds


if __name__ == "__main__":
    get_formatted_dataset()