"""Upload the demo to a Hugging Face Space, so it runs without this computer.

First sign in:  .\\.venv\\Scripts\\hf.exe auth login
Then run:       .\\.venv\\Scripts\\python.exe deploy_space.py <your-username>/review-sentiment-tester

Add --private to limit the Space to your account and invited members.
Add --password to be asked for a password that visitors must enter.
"""
import argparse
import getpass
from pathlib import Path

from huggingface_hub import HfApi

ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description="Upload the demo to a Hugging Face Space.")
    parser.add_argument("repo", help="your-username/space-name")
    parser.add_argument("--private", action="store_true", help="only you and invited members can open it")
    parser.add_argument("--password", action="store_true", help="ask for a password for the page")
    args = parser.parse_args()

    api = HfApi()
    api.create_repo(args.repo, repo_type="space", space_sdk="gradio", private=args.private, exist_ok=True)
    if args.password:
        api.add_space_secret(args.repo, "DEMO_PASSWORD", getpass.getpass("Page password: "))
    print("Uploading the app files...")
    api.upload_folder(repo_id=args.repo, repo_type="space", folder_path=ROOT / "hf_space")
    print("Uploading the model (496 MB), which can take several minutes...")
    api.upload_file(repo_id=args.repo, repo_type="space", path_or_fileobj=ROOT / "models" / "roberta.pt",
                    path_in_repo="roberta.pt")
    print(f"Done. The Space builds for a few minutes, then opens at: https://huggingface.co/spaces/{args.repo}")


if __name__ == "__main__":
    main()
