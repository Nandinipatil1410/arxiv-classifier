# arXiv Paper Classifier

Fine-tuned TinyLlama-1.1B with LoRA/PEFT to classify research papers into 11 arXiv categories.

## Results
- **Accuracy:** 84.3% on 2,500 test samples
- **Trainable params:** 0.1% (LoRA)
- **Training time:** ~80 mins on RTX 4050 (6GB VRAM)

## Links
- [Demo Video](https://youtu.be/-nHIcbDLyp0)
- [Model on HuggingFace](https://huggingface.co/nandini1410/arxiv-classifier-lora)

## Tech Stack
Python, PyTorch, HuggingFace Transformers, PEFT, TRL, Gradio

## Project Structure
- src/prepare_data.py   # dataset formatting
- src/train.py          # LoRA fine-tuning
- src/inference.py      # single prediction
- src/evaluate.py       # full test evaluation
- app.py                # Gradio web interface
