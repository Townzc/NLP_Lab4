import os
from pathlib import Path

from mindformers.pipeline import pipeline


DEFAULT_REPO_DIR = Path(__file__).resolve().parent.parent.parent / "NLP_Lab4"
REPO_DIR = Path(os.environ.get("NLP_LAB4_REPO_DIR", str(DEFAULT_REPO_DIR)))
PROMPT_FILE = REPO_DIR / "prompts" / "base_prompts.txt"


def load_prompts():
    if PROMPT_FILE.is_file():
        return [line.strip() for line in PROMPT_FILE.read_text(encoding="utf-8").splitlines() if line.strip()]
    return [
        "I love China, because",
        "Briefly explain what LoRA fine-tuning is.",
        "How can a student recharge a campus card?",
    ]


def main():
    generator = pipeline(task="text_generation", model="llama_7b", max_length=128)
    for index, prompt in enumerate(load_prompts(), start=1):
        print("\n===== Base Prompt {} =====".format(index))
        print("Q: {}".format(prompt))
        result = generator(prompt, top_k=3)
        print("A: {}".format(result))


if __name__ == "__main__":
    main()
