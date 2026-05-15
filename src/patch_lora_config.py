#!/usr/bin/env python3
"""Patch MindFormers Llama 7B LoRA YAML without adding extra dependencies."""

import argparse
import re
import shutil
from pathlib import Path


KEY_RE = re.compile(r"^(\s*)([A-Za-z_][A-Za-z0-9_]*):(.*)$")


def parse_bool(value):
    if isinstance(value, bool):
        return value
    value = str(value).strip().lower()
    if value in {"1", "true", "yes", "y"}:
        return True
    if value in {"0", "false", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError("expected a boolean value")


def split_comment(text):
    quote = None
    for index, char in enumerate(text):
        if char in {"'", '"'}:
            if quote == char:
                quote = None
            elif quote is None:
                quote = char
        elif char == "#" and quote is None:
            return text[:index].rstrip(), text[index:].rstrip()
    return text.rstrip(), ""


def render_value(value):
    if isinstance(value, bool):
        return "True" if value else "False"
    if isinstance(value, (int, float)):
        return str(value)
    value = str(value)
    if value in {"True", "False"}:
        return value
    if re.fullmatch(r"-?(\d+(\.\d*)?|\.\d+)(e-?\d+)?", value, flags=re.IGNORECASE):
        return value
    escaped = value.replace("'", "''")
    return "'{}'".format(escaped)


def patch_yaml(source, output, replacements):
    lines = source.read_text(encoding="utf-8").splitlines()
    patched = []
    stack = []
    replaced = set()

    for line in lines:
        match = KEY_RE.match(line)
        if not match or line.lstrip().startswith("#"):
            patched.append(line)
            continue

        spaces, key, rest = match.groups()
        indent = len(spaces)
        while stack and stack[-1][0] >= indent:
            stack.pop()

        path = tuple([item[1] for item in stack] + [key])
        if path in replacements:
            _, comment = split_comment(rest)
            new_line = "{}{}: {}".format(spaces, key, render_value(replacements[path]))
            if comment:
                new_line = "{}  {}".format(new_line, comment)
            patched.append(new_line)
            replaced.add(path)
        else:
            patched.append(line)

        stack.append((indent, key))

    missing = sorted(".".join(path) for path in set(replacements) - replaced)
    if missing:
        raise KeyError("Could not find YAML keys: {}".format(", ".join(missing)))

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(patched) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--epochs", default="20")
    parser.add_argument("--batch-size", default="2")
    parser.add_argument("--sink-size", default="5")
    parser.add_argument("--learning-rate", default="1.e-4")
    parser.add_argument("--optimizer-type", default="FP32StateAdamWeightDecay")
    parser.add_argument("--lr-schedule-type", default="CosineWithWarmUpLR")
    parser.add_argument("--warmup-ratio", default="0.03")
    parser.add_argument("--lora-rank", default="16")
    parser.add_argument("--lora-alpha", default="16")
    parser.add_argument("--lora-dropout", default="0.05")
    parser.add_argument("--use-parallel", type=parse_bool, default=False)
    args = parser.parse_args()

    source = Path(args.source)
    output = Path(args.output)
    if not source.is_file():
        raise FileNotFoundError(source)

    output.parent.mkdir(parents=True, exist_ok=True)
    if source.resolve() != output.resolve():
        shutil.copyfile(str(source), str(output))
        source = output

    replacements = {
        ("load_checkpoint",): args.checkpoint,
        ("use_parallel",): args.use_parallel,
        ("runner_config", "epochs"): args.epochs,
        ("runner_config", "batch_size"): args.batch_size,
        ("runner_config", "sink_size"): args.sink_size,
        ("optimizer", "type"): args.optimizer_type,
        ("optimizer", "learning_rate"): args.learning_rate,
        ("lr_schedule", "type"): args.lr_schedule_type,
        ("lr_schedule", "learning_rate"): args.learning_rate,
        ("lr_schedule", "warmup_ratio"): args.warmup_ratio,
        ("train_dataset", "data_loader", "dataset_dir"): args.dataset,
        ("train_dataset", "batch_size"): args.batch_size,
        ("model", "model_config", "pet_config", "lora_rank"): args.lora_rank,
        ("model", "model_config", "pet_config", "lora_alpha"): args.lora_alpha,
        ("model", "model_config", "pet_config", "lora_dropout"): args.lora_dropout,
    }
    patch_yaml(source, output, replacements)
    print("Wrote patched config: {}".format(output))


if __name__ == "__main__":
    main()
