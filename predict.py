"""Type a sentence and see what the selected model (RoBERTa, seed 44) predicts.

Interactive:  .\\.venv\\Scripts\\python.exe predict.py
One-off:      .\\.venv\\Scripts\\python.exe predict.py "These shoes fell apart in a week"

For trying the model out only. It was trained on product reviews, so other kinds of
text may do worse. Its confidence is often high even when it is wrong: 31 of its 38
validation errors had over 90% confidence.
"""
import argparse

import torch
from transformers import AutoTokenizer
from transformers.utils import logging as hf_logging

import common
import transformer

CHECKPOINT = common.ROOT / "models" / "roberta.pt"
LABELS = {0: "NEGATIVE", 1: "POSITIVE"}


def load():
    saved = torch.load(CHECKPOINT, map_location="cpu")
    cfg = transformer.Config(**saved["config"])
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
    model = transformer.TransformerClassifier(cfg.model_name, cfg.dropout)
    model.load_state_dict(saved["state"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # the GPU is optional here
    return model.to(device).eval(), tokenizer, cfg, device


@torch.no_grad()
def classify(text, model, tokenizer, cfg, device):
    """Returns the predicted class (0 or 1) and the model's probability for that class."""
    encoded = tokenizer(text, truncation=True, max_length=cfg.max_len, return_tensors="pt").to(device)
    probabilities = torch.softmax(model(encoded["input_ids"], encoded["attention_mask"]), dim=1)[0]
    label = int(probabilities.argmax())
    return label, float(probabilities[label])


def show(text, *loaded):
    label, confidence = classify(text, *loaded)
    print(f"{LABELS[label]} ({label}), confidence {confidence:.1%}")


def main():
    parser = argparse.ArgumentParser(description="Classify the sentiment of a review.")
    parser.add_argument("text", nargs="*", help="the review; leave out to type several in turn")
    words = parser.parse_args().text
    hf_logging.set_verbosity_error()
    hf_logging.disable_progress_bar()
    loaded = load()

    if words:
        show(" ".join(words), *loaded)
        return
    print("Type a review and press Enter. Press Enter on an empty line to quit.")
    while True:
        try:
            text = input("\nReview: ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not text:
            break
        show(text, *loaded)


if __name__ == "__main__":
    main()
