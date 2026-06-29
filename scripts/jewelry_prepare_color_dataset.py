#!/usr/bin/env python3
"""Prepare a jewelry color annotation CSV from local product images."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
import unicodedata
from collections import Counter, deque
from datetime import datetime
from pathlib import Path

import numpy as np

try:
    from PIL import Image, UnidentifiedImageError
except ImportError as exc:  # pragma: no cover - dependency guard for standalone use
    raise SystemExit("This script requires Pillow. Install LLaMA-Factory dependencies first.") from exc


CSV_COLUMNS = ["image", "extracted_websafe_colors", "suggested_color_labels", "final_color_labels"]
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}
WEBSAFE_STEP = 51
GROUNDTRUTH_LABEL_COLUMNS = ["color1", "color2", "color3", "color4"]
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

    parser = argparse.ArgumentParser(description="Extract Web Safe colors and build a jewelry annotation CSV.")
    parser.add_argument("--data-dir", type=Path, default=data_dir, help="LLaMA-Factory data directory.")
    parser.add_argument("--image-dir", type=Path, default=None, help="Directory containing jewelry images.")
    parser.add_argument("--palette", type=Path, default=None, help="Fixed jewelry color palette JSON.")
    parser.add_argument("--output", type=Path, default=None, help="Annotation CSV to create or extend.")
    parser.add_argument("--groundtruth", type=Path, default=None, help="CSV with image,color1,color2,color3,color4 labels.")
    parser.add_argument("--refresh", action="store_true", help="Rebuild the whole annotation CSV even if it exists.")
    parser.add_argument("--max-dim", type=int, default=512, help="Resize max side before extraction.")
    parser.add_argument("--max-colors", type=int, default=12, help="Maximum Web Safe colors to write per image.")
    return parser.parse_args()


def load_palette(path: Path) -> dict[str, list[str]]:
    with path.open("r", encoding="utf-8") as f:
        palette = json.load(f)

    return {label: [color.upper() for color in colors] for label, colors in palette.items()}


def normalize_label(label: str) -> str:
    return unicodedata.normalize("NFKC", label.strip()).translate(LABEL_CHAR_TRANSLATION)


def dedupe_labels(labels: list[str]) -> list[str]:
    deduped: list[str] = []
    seen: set[str] = set()
    for label in labels:
        if label and label not in seen:
            deduped.append(label)
            seen.add(label)

    return deduped


def find_images(image_dir: Path) -> list[Path]:
    def sort_key(path: Path) -> tuple[int, int | str, str]:
        stem = path.stem
        if stem.isdigit():
            return (0, int(stem), path.suffix.lower())

        return (1, stem, path.suffix.lower())

    return sorted(
        (path for path in image_dir.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS),
        key=sort_key,
    )


def load_groundtruth(path: Path | None, allowed_labels: set[str]) -> dict[str, list[str]]:
    if path is None:
        return {}

    label_by_image: dict[str, list[str]] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        required_columns = {"image", *GROUNDTRUTH_LABEL_COLUMNS}
        missing_columns = required_columns - set(reader.fieldnames or [])
        if missing_columns:
            raise ValueError(f"Missing columns in {path}: {', '.join(sorted(missing_columns))}")

        for row_number, row in enumerate(reader, start=2):
            image_key = (row.get("image") or "").strip()
            labels = [normalize_label(row.get(column) or "") for column in GROUNDTRUTH_LABEL_COLUMNS]
            labels = dedupe_labels(labels)
            if not image_key and labels:
                raise ValueError(f"Row {row_number}: image is empty.")
            if not labels:
                continue
            if not 1 <= len(labels) <= 4:
                raise ValueError(f"Row {row_number}: ground truth must contain 1 to 4 labels.")

            invalid_labels = [label for label in labels if label not in allowed_labels]
            if invalid_labels:
                raise ValueError(f"Row {row_number}: invalid labels: {', '.join(invalid_labels)}")

            label_by_image[image_key] = labels

    return label_by_image


def get_groundtruth_labels(image_path: Path, image_rel: str, label_by_image: dict[str, list[str]]) -> list[str]:
    return (
        label_by_image.get(image_rel)
        or label_by_image.get(image_path.name)
        or label_by_image.get(image_path.stem)
        or []
    )


def image_to_rgba(path: Path, max_dim: int) -> np.ndarray:
    resample = getattr(Image, "Resampling", Image).LANCZOS
    with Image.open(path) as image:
        image = image.convert("RGBA")
        image.thumbnail((max_dim, max_dim), resample)
        return np.asarray(image)


def border_connected_mask(candidates: np.ndarray) -> np.ndarray:
    height, width = candidates.shape
    visited = np.zeros_like(candidates, dtype=bool)
    queue: deque[tuple[int, int]] = deque()

    def add(y: int, x: int) -> None:
        if candidates[y, x] and not visited[y, x]:
            visited[y, x] = True
            queue.append((y, x))

    for x in range(width):
        add(0, x)
        add(height - 1, x)
    for y in range(height):
        add(y, 0)
        add(y, width - 1)

    while queue:
        y, x = queue.popleft()
        if y > 0:
            add(y - 1, x)
        if y + 1 < height:
            add(y + 1, x)
        if x > 0:
            add(y, x - 1)
        if x + 1 < width:
            add(y, x + 1)

    return visited


def foreground_mask(rgba: np.ndarray) -> np.ndarray:
    alpha = rgba[:, :, 3]
    valid = alpha > 16
    if not valid.any():
        return valid

    rgb = rgba[:, :, :3].astype(np.int16)
    height, width = valid.shape
    border = np.zeros_like(valid, dtype=bool)
    border[0, :] = True
    border[-1, :] = True
    border[:, 0] = True
    border[:, -1] = True

    border_valid = border & valid
    if not border_valid.any():
        return valid

    border_rgb = rgb[border_valid]
    bg_rgb = np.median(border_rgb, axis=0)
    border_distance = np.linalg.norm(border_rgb - bg_rgb, axis=1)
    tolerance = float(np.clip(np.percentile(border_distance, 75) + 20.0, 30.0, 64.0))

    # Only remove a connected border background when the border has a dominant, consistent color.
    if float(np.mean(border_distance <= tolerance)) < 0.45:
        return valid

    distance = np.linalg.norm(rgb - bg_rgb, axis=2)
    background_candidates = (distance <= tolerance) & valid
    connected_background = border_connected_mask(background_candidates)
    foreground = valid & ~connected_background

    min_pixels = max(50, int(valid.sum() * 0.005))
    if int(foreground.sum()) < min_pixels:
        return valid

    return foreground


def quantize_websafe(rgb: np.ndarray) -> np.ndarray:
    quantized = ((rgb.astype(np.int16) + (WEBSAFE_STEP // 2)) // WEBSAFE_STEP) * WEBSAFE_STEP
    return np.clip(quantized, 0, 255).astype(np.uint8)


def rgb_to_hex(rgb: np.ndarray | tuple[int, int, int]) -> str:
    r, g, b = [int(channel) for channel in rgb]
    return f"#{r:02X}{g:02X}{b:02X}"


def hex_to_rgb(color: str) -> np.ndarray:
    color = color.lstrip("#")
    return np.array([int(color[i : i + 2], 16) for i in (0, 2, 4)], dtype=np.int16)


def extract_websafe_colors(path: Path, max_dim: int, max_colors: int) -> list[tuple[str, float]]:
    rgba = image_to_rgba(path, max_dim)
    mask = foreground_mask(rgba)
    rgb = rgba[:, :, :3][mask]
    if rgb.size == 0:
        return []

    quantized = quantize_websafe(rgb).reshape(-1, 3)
    colors, counts = np.unique(quantized, axis=0, return_counts=True)
    order = np.argsort(-counts)
    total = int(counts.sum())
    min_count = max(3, int(total * 0.005))

    selected: list[tuple[str, float]] = []
    seen: set[str] = set()
    for idx in order:
        if int(counts[idx]) < min_count and len(selected) >= 4:
            continue

        color = rgb_to_hex(colors[idx])
        if color in seen:
            continue

        selected.append((color, float(counts[idx]) / total))
        seen.add(color)
        if len(selected) >= max_colors:
            break

    return selected


def score_labels(
    websafe_colors: list[tuple[str, float]], palette: dict[str, list[str]]
) -> list[str]:
    labels = list(palette.keys())
    color_to_labels: dict[str, list[str]] = {}
    palette_rgb = {label: np.array([hex_to_rgb(color) for color in colors]) for label, colors in palette.items()}

    for label, colors in palette.items():
        for color in colors:
            color_to_labels.setdefault(color.upper(), []).append(label)

    scores: Counter[str] = Counter()
    hit_counts: Counter[str] = Counter()
    first_rank: dict[str, int] = {}

    for rank, (color, share) in enumerate(websafe_colors):
        exact_labels = color_to_labels.get(color.upper(), [])
        if exact_labels:
            for label in exact_labels:
                scores[label] += share * (1.0 + 0.15 / (rank + 1))
                hit_counts[label] += 1
                first_rank[label] = min(first_rank.get(label, rank), rank)
            continue

        rgb = hex_to_rgb(color)
        distances = []
        for label in labels:
            min_distance = float(np.min(np.linalg.norm(palette_rgb[label] - rgb, axis=1)))
            distances.append((label, min_distance))

        distances.sort(key=lambda item: item[1])
        best_distance = distances[0][1]
        if best_distance > 96:
            continue

        for label, distance in distances[:2]:
            if distance > best_distance + 18:
                continue

            scores[label] += share * 0.35 * max(0.0, 1.0 - distance / 180.0)
            first_rank[label] = min(first_rank.get(label, rank), rank)

    if not scores:
        if not websafe_colors:
            return []

        rgb = hex_to_rgb(websafe_colors[0][0])
        nearest_label = min(
            labels,
            key=lambda label: float(np.min(np.linalg.norm(palette_rgb[label] - rgb, axis=1))),
        )
        return [nearest_label]

    palette_order = {label: idx for idx, label in enumerate(labels)}
    ranked = sorted(
        scores,
        key=lambda label: (
            -scores[label],
            -hit_counts[label],
            first_rank.get(label, 999),
            palette_order[label],
        ),
    )
    top_score = scores[ranked[0]]
    threshold = max(0.01, top_score * 0.18)
    return [label for label in ranked if scores[label] >= threshold][:4]


def read_existing_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []

    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        return [{column: row.get(column, "") for column in CSV_COLUMNS} for row in reader]


def write_csv(path: Path, rows: list[dict[str, str]], backup_existing: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if backup_existing and path.exists():
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        backup_path = path.with_name(f"{path.name}.bak.{timestamp}")
        shutil.copy2(path, backup_path)
        print(f"Backed up existing CSV to {backup_path}")

    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    data_dir = args.data_dir.resolve()
    image_dir = (args.image_dir or data_dir / "image").resolve()
    palette_path = (args.palette or data_dir / "jewelry_color_palette.json").resolve()
    output_path = (args.output or data_dir / "jewelry_color_annotation_template.csv").resolve()

    palette = load_palette(palette_path)
    label_by_image = load_groundtruth(args.groundtruth.resolve() if args.groundtruth else None, set(palette))
    existing_rows = [] if args.refresh else read_existing_csv(output_path)
    existing_images = {row["image"] for row in existing_rows if row.get("image")}
    new_rows: list[dict[str, str]] = []

    for image_path in find_images(image_dir):
        image_rel = image_path.relative_to(data_dir).as_posix()
        if image_rel in existing_images:
            continue

        try:
            colors = extract_websafe_colors(image_path, args.max_dim, args.max_colors)
        except (OSError, UnidentifiedImageError) as exc:
            print(f"Skipped unreadable image {image_path}: {exc}", file=sys.stderr)
            continue

        suggested_labels = score_labels(colors, palette)
        final_labels = get_groundtruth_labels(image_path, image_rel, label_by_image)
        new_rows.append(
            {
                "image": image_rel,
                "extracted_websafe_colors": "、".join(color for color, _share in colors),
                "suggested_color_labels": "、".join(suggested_labels),
                "final_color_labels": "、".join(final_labels),
            }
        )

    if not new_rows:
        print(f"No new images to add. Existing CSV is unchanged: {output_path}")
        return

    write_csv(output_path, existing_rows + new_rows, backup_existing=output_path.exists())
    print(f"Wrote {len(new_rows)} new rows to {output_path}")
    print(f"Total rows: {len(existing_rows) + len(new_rows)}")


if __name__ == "__main__":
    main()
