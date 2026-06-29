#!/usr/bin/env python3
"""Build LLaMA-Factory multimodal SFT JSON files from jewelry color annotations."""

from __future__ import annotations

import argparse
import csv
import json
import random
import re
import unicodedata
from pathlib import Path


LABEL_SPLIT_RE = re.compile(r"[,，、;\n\r\t]+")
COLOR_SPLIT_RE = re.compile(r"[,，、\s]+")
LABEL_CHAR_TRANSLATION = str.maketrans(
    {
        "⾊": "色",
        "⻩": "黄",
        "⾦": "金",
        "⽩": "白",
        "⿊": "黑",
        "⽔": "水",
        "⻘": "青",
        "⾐": "衣",
    }
)


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[1]
    data_dir = repo_root / "data"

    parser = argparse.ArgumentParser(description="Build jewelry color SFT train/val data.")
    parser.add_argument("--data-dir", type=Path, default=data_dir, help="LLaMA-Factory data directory.")
    parser.add_argument("--palette", type=Path, default=None, help="Fixed jewelry color palette JSON.")
    parser.add_argument("--input", type=Path, default=None, help="Reviewed annotation CSV.")
    parser.add_argument("--train-output", type=Path, default=None, help="Output train JSON path.")
    parser.add_argument("--val-output", type=Path, default=None, help="Output validation JSON path.")
    parser.add_argument("--train-ratio", type=float, default=0.8, help="Train split ratio.")
    parser.add_argument("--seed", type=int, default=42, help="Deterministic shuffle seed.")
    parser.add_argument("--no-shuffle", action="store_true", help="Preserve CSV order before splitting.")
    return parser.parse_args()


def load_allowed_labels(path: Path) -> list[str]:
    with path.open("r", encoding="utf-8") as f:
        palette = json.load(f)

    return list(palette.keys())


def parse_labels(raw_labels: str) -> list[str]:
    normalized = raw_labels.replace("颜色标签：", "").replace("颜色标签:", "")
    labels = [
        unicodedata.normalize("NFKC", label.strip()).translate(LABEL_CHAR_TRANSLATION)
        for label in LABEL_SPLIT_RE.split(normalized)
        if label.strip()
    ]

    deduped: list[str] = []
    seen: set[str] = set()
    for label in labels:
        if label not in seen:
            deduped.append(label)
            seen.add(label)

    return deduped


def parse_colors(raw_colors: str) -> list[str]:
    colors = []
    for color in COLOR_SPLIT_RE.split(raw_colors):
        color = color.strip().upper()
        if color:
            colors.append(color)

    return colors


def build_user_prompt(colors: list[str], allowed_labels: list[str]) -> str:
    color_text = "、".join(colors) if colors else "未填写"
    label_text = "、".join(allowed_labels)
    return (
        f"<image>\n已根据固定 Web Safe 色卡提取到以下颜色：{color_text}。\n"
        f"请根据固定色卡输出这个饰品的颜色标签。只能从以下16个固定中文颜色标签中选择：{label_text}。"
        "最少输出1个，最多输出4个。按视觉重要性排序，第一个必须是主色系。"
        "不要输出英文、字母编号、色号或解释。"
    )


def build_sample(row: dict[str, str], labels: list[str], allowed_labels: list[str]) -> dict[str, object]:
    colors = parse_colors(row.get("extracted_websafe_colors", ""))
    assistant_content = "颜色标签：\n" + "\n".join(labels)
    return {
        "messages": [
            {
                "role": "user",
                "content": build_user_prompt(colors, allowed_labels),
            },
            {
                "role": "assistant",
                "content": assistant_content,
            },
        ],
        "images": [row["image"]],
    }


def validate_labels(row_number: int, labels: list[str], allowed_set: set[str]) -> None:
    if not 1 <= len(labels) <= 4:
        raise ValueError(f"Row {row_number}: final_color_labels must contain 1 to 4 labels, got {len(labels)}.")

    invalid_labels = [label for label in labels if label not in allowed_set]
    if invalid_labels:
        raise ValueError(f"Row {row_number}: invalid labels: {', '.join(invalid_labels)}")


def load_samples(csv_path: Path, data_dir: Path, allowed_labels: list[str]) -> list[dict[str, object]]:
    allowed_set = set(allowed_labels)
    samples: list[dict[str, object]] = []

    with csv_path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        required_columns = {"image", "extracted_websafe_colors", "final_color_labels"}
        missing_columns = required_columns - set(reader.fieldnames or [])
        if missing_columns:
            raise ValueError(f"Missing columns in {csv_path}: {', '.join(sorted(missing_columns))}")

        for row_number, row in enumerate(reader, start=2):
            raw_labels = (row.get("final_color_labels") or "").strip()
            if not raw_labels:
                continue

            image = (row.get("image") or "").strip()
            if not image:
                raise ValueError(f"Row {row_number}: image is empty.")

            image_path = data_dir / image
            if not image_path.exists():
                raise ValueError(f"Row {row_number}: image file does not exist: {image_path}")

            labels = parse_labels(raw_labels)
            validate_labels(row_number, labels, allowed_set)
            samples.append(build_sample({**row, "image": image}, labels, allowed_labels))

    return samples


def split_samples(
    samples: list[dict[str, object]], train_ratio: float, seed: int, no_shuffle: bool
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    if not 0 < train_ratio < 1:
        raise ValueError("--train-ratio must be between 0 and 1.")

    samples = list(samples)
    if not no_shuffle:
        rng = random.Random(seed)
        rng.shuffle(samples)

    if len(samples) <= 1:
        return samples, []

    val_count = max(1, int(round(len(samples) * (1.0 - train_ratio))))
    val_count = min(val_count, len(samples) - 1)
    train_count = len(samples) - val_count
    return samples[:train_count], samples[train_count:]


def write_json(path: Path, samples: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(samples, f, ensure_ascii=False, indent=2)
        f.write("\n")


def main() -> None:
    args = parse_args()
    data_dir = args.data_dir.resolve()
    palette_path = (args.palette or data_dir / "jewelry_color_palette.json").resolve()
    input_path = (args.input or data_dir / "jewelry_color_annotation_template.csv").resolve()
    train_output = (args.train_output or data_dir / "jewelry_color_train.json").resolve()
    val_output = (args.val_output or data_dir / "jewelry_color_val.json").resolve()

    allowed_labels = load_allowed_labels(palette_path)
    samples = load_samples(input_path, data_dir, allowed_labels)
    if not samples:
        raise SystemExit(f"No rows with final_color_labels were found in {input_path}.")

    train_samples, val_samples = split_samples(samples, args.train_ratio, args.seed, args.no_shuffle)
    write_json(train_output, train_samples)
    write_json(val_output, val_samples)
    print(f"Wrote {len(train_samples)} train samples to {train_output}")
    print(f"Wrote {len(val_samples)} validation samples to {val_output}")


if __name__ == "__main__":
    main()
