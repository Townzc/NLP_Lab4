import os
from pathlib import Path

import mindspore as ms
from mindformers import Trainer


DEFAULT_REPO_DIR = Path(__file__).resolve().parent.parent.parent / "NLP_Lab4"
REPO_DIR = Path(os.environ.get("NLP_LAB4_REPO_DIR", str(DEFAULT_REPO_DIR)))
PROMPT_FILE = REPO_DIR / "prompts" / "campus_prompts.txt"
DEFAULT_CKPT = (
    "/home/ma-user/work/llama_lab/mindformers/scripts/mf_standalone/"
    "output/checkpoint/rank_0/llama_7b_lora_rank_0-20_5.ckpt"
)


def load_questions():
    if PROMPT_FILE.is_file():
        return [line.strip() for line in PROMPT_FILE.read_text(encoding="utf-8").splitlines() if line.strip()]
    return [
        "如何充值校园卡？",
        "校园卡丢失后应该怎么办？",
        "如何预约图书馆座位？",
    ]


def build_prompt(question):
    return (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{}\n\n### Response:".format(question)
    )


def main():
    lora_ckpt = os.environ.get("LORA_CKPT", DEFAULT_CKPT)
    print("Using LoRA checkpoint: {}".format(lora_ckpt))

    ms.set_context(mode=ms.GRAPH_MODE, device_target="Ascend", device_id=0)
    trainer = Trainer(task="text_generation", model="llama_7b", pet_method="lora")

    for index, question in enumerate(load_questions(), start=1):
        print("\n===== LoRA Prompt {} =====".format(index))
        print("Q: {}".format(question))
        result = trainer.predict(
            input_data=build_prompt(question),
            predict_checkpoint=lora_ckpt,
        )
        print("A: {}".format(result[0]["text_generation_text"][0]))


if __name__ == "__main__":
    main()
