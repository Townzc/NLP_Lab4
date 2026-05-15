#!/usr/bin/env python3
"""Clean and validate Stanford Alpaca style JSON data."""

import argparse
import html
import json
import re
from pathlib import Path


CONTROL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
TAG_RE = re.compile(r"<[^>]+>")
SPACE_RE = re.compile(r"\s+")


def normalize_text(value):
    value = "" if value is None else str(value)
    value = html.unescape(value)
    value = TAG_RE.sub("", value)
    value = CONTROL_RE.sub("", value)
    value = SPACE_RE.sub(" ", value)
    return value.strip()


def clean_records(records):
    cleaned = []
    seen = set()
    dropped_empty = 0
    dropped_duplicate = 0

    for item in records:
        if not isinstance(item, dict):
            dropped_empty += 1
            continue

        instruction = normalize_text(item.get("instruction", ""))
        input_text = normalize_text(item.get("input", ""))
        output = normalize_text(item.get("output", ""))

        if not instruction or not output:
            dropped_empty += 1
            continue

        key = (instruction, input_text)
        if key in seen:
            dropped_duplicate += 1
            continue

        seen.add(key)
        cleaned.append(
            {
                "instruction": instruction,
                "input": input_text,
                "output": output,
            }
        )

    stats = {
        "input": len(records),
        "output": len(cleaned),
        "dropped_empty": dropped_empty,
        "dropped_duplicate": dropped_duplicate,
    }
    return cleaned, stats


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Input Alpaca JSON file.")
    parser.add_argument("--output", required=True, help="Output cleaned JSON file.")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)

    with input_path.open("r", encoding="utf-8") as f:
        records = json.load(f)

    if not isinstance(records, list):
        raise ValueError("Alpaca data must be a JSON list.")

    cleaned, stats = clean_records(records)
    if not cleaned:
        raise ValueError("No valid Alpaca records remained after cleaning.")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(cleaned, f, ensure_ascii=False, indent=2)
        f.write("\n")

    print(
        "Cleaned Alpaca data: input={input}, output={output}, "
        "dropped_empty={dropped_empty}, dropped_duplicate={dropped_duplicate}".format(
            **stats
        )
    )
    print("Wrote {}".format(output_path))


if __name__ == "__main__":
    main()
