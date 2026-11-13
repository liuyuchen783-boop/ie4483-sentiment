"""A password-protected web page for trying the selected model (RoBERTa, seed 44).

On this computer only:   .\\.venv\\Scripts\\python.exe demo.py
With a shareable link:   .\\.venv\\Scripts\\python.exe demo.py --share

The model runs on this computer, so the link only works while this script is running.
Typed text is not saved. A new password is printed at each start unless one is given
with --password. Extra packages are listed in requirements-demo.txt.
"""
import argparse
import secrets

import gradio as gr
from transformers.utils import logging as hf_logging

import predict

MAX_CHARACTERS = 5000  # far beyond a normal review; stops very long pastes tying up the GPU
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


def build(loaded):
    def analyse(text):
        text = (text or "").strip()
        if not text:
            raise gr.Error("Type a review first.")
        if len(text) > MAX_CHARACTERS:
            raise gr.Error(f"Please keep it under {MAX_CHARACTERS:,} characters.")
        label, confidence = predict.classify(text, *loaded)
        names = {0: "Negative", 1: "Positive"}
        return {names[label]: confidence, names[1 - label]: 1 - confidence}

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
    return app


def main():
    parser = argparse.ArgumentParser(description="Web page for trying the sentiment model.")
    parser.add_argument("--share", action="store_true", help="create a temporary public link")
    parser.add_argument("--password", help="password for the page; by default a new one is generated")
    args = parser.parse_args()

    hf_logging.set_verbosity_error()
    hf_logging.disable_progress_bar()
    password = args.password or secrets.token_urlsafe(6)
    app = build(predict.load())
    print(f"Username: group\nPassword: {password}", flush=True)
    app.queue(max_size=20).launch(share=args.share, auth=("group", password))


if __name__ == "__main__":
    main()
