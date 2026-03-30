# app.py
import torch
import gradio as gr
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

# ── Config ─────────────────────────────────────────────────────────────────────
BASE_MODEL  = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
ADAPTER_DIR = "models/lora-arxiv"

LABEL_NAMES = [
    'math.AC', 'cs.CV', 'cs.AI', 'cs.SY', 'math.GR',
    'cs.CE',   'cs.PL', 'cs.IT', 'cs.DS', 'cs.NE', 'math.ST'
]

LABEL_DESCRIPTIONS = {
    'math.AC':  'Commutative Algebra',
    'cs.CV':    'Computer Vision',
    'cs.AI':    'Artificial Intelligence',
    'cs.SY':    'Systems & Control',
    'math.GR':  'Group Theory',
    'cs.CE':    'Computational Engineering',
    'cs.PL':    'Programming Languages',
    'cs.IT':    'Information Theory',
    'cs.DS':    'Data Structures & Algorithms',
    'cs.NE':    'Neural & Evolutionary Computing',
    'math.ST':  'Statistics Theory',
}

CATEGORIES_STR = ", ".join(LABEL_NAMES)

# ── Load model once at startup ─────────────────────────────────────────────────
print("Loading model... (this takes ~20 seconds)")

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)

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
print("Model ready!")

# ── Prediction function ────────────────────────────────────────────────────────
def classify_abstract(abstract: str):
    # Input validation
    if not abstract or len(abstract.strip()) < 50:
        return "⚠️ Please enter a longer abstract (at least 50 characters).", ""

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
    predicted = result.split()[0] if result.split() else "unknown"

    # Format output
    if predicted in LABEL_DESCRIPTIONS:
        category    = f"**{predicted}**"
        description = f"{LABEL_DESCRIPTIONS[predicted]}"
    else:
        category    = f"**{predicted}**"
        description = "_(unrecognized label — try again)_"

    return category, description


# ── Gradio UI ──────────────────────────────────────────────────────────────────
with gr.Blocks(
    theme=gr.themes.Default(
        font=gr.themes.GoogleFont("Inter"),
        primary_hue="slate",
    ),
    css="""
    .gradio-container { max-width: 780px !important; margin: auto; }
    #title { text-align: center; padding: 24px 0 8px 0; }
    #subtitle { text-align: center; color: #6b7280; margin-bottom: 24px; }
    #predict-btn { background: #1e293b !important; color: white !important; }
    #result-box { font-size: 1.4em; font-weight: 600; }
    """
) as demo:

    gr.HTML("<h1 id='title'>arXiv Paper Classifier</h1>")
    gr.HTML("<p id='subtitle'>Fine-tuned TinyLlama-1.1B with LoRA · Trained on 28,388 arXiv abstracts</p>")

    with gr.Row():
        abstract_input = gr.Textbox(
            label="Paper Abstract",
            placeholder="Paste your arXiv abstract here...",
            lines=8,
        )

    with gr.Row():
        predict_btn = gr.Button("Classify →", variant="primary", elem_id="predict-btn")

    with gr.Row():
        with gr.Column():
            category_out    = gr.Markdown(label="Predicted Category", elem_id="result-box")
        with gr.Column():
            description_out = gr.Markdown(label="Field")

    # Example abstracts — one per category
    gr.Examples(
        examples=[
            ["We propose a novel convolutional neural network architecture for real-time object detection in high-resolution images. Our method achieves state-of-the-art performance on the COCO benchmark while running at 60 FPS on a standard GPU."],
            ["We present a reinforcement learning agent that masters the game of Go without human knowledge, learning solely through self-play using a deep neural network to evaluate positions and select moves."],
            ["This paper studies the chromatic polynomial of planar graphs and its relationship to the four color theorem, providing new bounds on the number of proper colorings for sparse graphs."],
        ],
        inputs=abstract_input,
        label="Try an example",
    )

    predict_btn.click(
        fn=classify_abstract,
        inputs=abstract_input,
        outputs=[category_out, description_out],
    )

# ── Launch ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    demo.launch(
        share=True,       # set True to get a public link (useful for resume demos)
        server_port=7860,
    )