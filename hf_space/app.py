"""Hugging Face Space: try the selected sentiment model (RoBERTa, seed 44) on CPU.

Standalone, so the Space needs only this file and roberta.pt. Typed text is not saved.
Set a DEMO_PASSWORD secret on the Space to require a password (username: group).
"""
import os

import gradio as gr
import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModel, AutoTokenizer

MODEL_PATH = os.environ.get("MODEL_PATH", "roberta.pt")
MAX_CHARACTERS = 5000
EXAMPLES = [
    "These shoes are amazing, I wear them every day",
    "Terrible quality, it fell apart after one wash",
    "Not bad at all for the price",
    "Great, another pair that broke in a week. What a bargain.",
    "Love the colour but it's way too small and I had to return it",
]
NOTES = """
**What to expect**

- Trained on product reviews, so other kinds of text may do worse.
- It can only answer positive or negative, so a neutral sentence still gets one of those.
- Sarcasm and mixed reviews are its main weaknesses.
- High confidence doesn't mean it's right. Most of its validation errors had over 90% confidence.

Validation macro-F1: 0.922 ± 0.006 over 3 seeds. Typed text is not saved.
"""


class Classifier(nn.Module):
    """Same layers as transformer.py, built without downloading the base weights."""

    def __init__(self, name, dropout):
        super().__init__()
        self.encoder = AutoModel.from_config(AutoConfig.from_pretrained(name), add_pooling_layer=False)
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(self.encoder.config.hidden_size, 2)

    def forward(self, input_ids, attention_mask):
        states = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        return self.head(self.dropout(states[:, 0]))


saved = torch.load(MODEL_PATH, map_location="cpu")
config = saved["config"]
tokenizer = AutoTokenizer.from_pretrained(config["model_name"])
model = Classifier(config["model_name"], config["dropout"])
model.load_state_dict(saved["state"])
model.eval()


@torch.no_grad()
def analyse(text):
    text = (text or "").strip()
    if not text:
        raise gr.Error("Type a review first.")
    if len(text) > MAX_CHARACTERS:
        raise gr.Error(f"Please keep it under {MAX_CHARACTERS:,} characters.")
    encoded = tokenizer(text, truncation=True, max_length=config["max_len"], return_tensors="pt")
    negative, positive = torch.softmax(model(encoded["input_ids"], encoded["attention_mask"]), dim=1)[0].tolist()
    return {"Negative": negative, "Positive": positive}


with gr.Blocks(title="Review Sentiment Tester", analytics_enabled=False) as app:
    gr.Markdown("# Review sentiment tester\nType a product review to see what our RoBERTa model predicts.")
    review = gr.Textbox(label="Review", lines=4, max_length=MAX_CHARACTERS,
                        placeholder="The shoes are comfortable but the colour faded quickly")
    button = gr.Button("Check sentiment", variant="primary")
    result = gr.Label(label="Prediction", num_top_classes=2)
    gr.Examples(EXAMPLES, inputs=review, label="Try an example")
    gr.Markdown(NOTES)
    button.click(analyse, review, result, api_name="analyse")
    review.submit(analyse, review, result, api_name=False)

password = os.environ.get("DEMO_PASSWORD")
app.queue(max_size=20).launch(auth=("group", password) if password else None)
