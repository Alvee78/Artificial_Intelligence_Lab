from __future__ import annotations

import argparse
import random
import shutil
from pathlib import Path


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp"}


def collect_images(class_dir: Path) -> list[Path]:
    return sorted(
        path for path in class_dir.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def prepare_directory(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def copy_split(files: list[Path], destination: Path) -> int:
    destination.mkdir(parents=True, exist_ok=True)
    for file_path in files:
        shutil.copy2(file_path, destination / file_path.name)
    return len(files)


def split_class_images(
    class_name: str,
    class_dir: Path,
    output_root: Path,
    train_ratio: float,
    test_ratio: float,
    validation_ratio: float,
    seed: int,
) -> dict[str, int]:
    files = collect_images(class_dir)
    if not files:
        return {"train": 0, "test": 0, "validation": 0}

    rng = random.Random(seed)
    rng.shuffle(files)

    total = len(files)
    test_count = round(total * test_ratio)
    validation_count = round(total * validation_ratio)
    train_count = total - test_count - validation_count

    train_files = files[:train_count]
    test_files = files[train_count:train_count + test_count]
    validation_files = files[train_count + test_count:]

    counts = {
        "train": copy_split(train_files, output_root / "train" / class_name),
        "test": copy_split(test_files, output_root / "test" / class_name),
        "validation": copy_split(validation_files, output_root / "validation" / class_name),
    }

    return counts


def main() -> None:
    parser = argparse.ArgumentParser(description="Split a leaf image dataset into train, test, and validation folders.")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path(__file__).parent / "Dataset",
        help="Path to the source dataset folder containing class subfolders.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).parent / "split_dataset",
        help="Path where train/test/validation folders will be created.",
    )
    parser.add_argument("--train-ratio", type=float, default=0.7, help="Training split ratio.")
    parser.add_argument("--test-ratio", type=float, default=0.2, help="Test split ratio.")
    parser.add_argument("--validation-ratio", type=float, default=0.1, help="Validation split ratio.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducible splits.")
    args = parser.parse_args()

    if not args.input.exists():
        raise FileNotFoundError(f"Input dataset folder not found: {args.input}")

    split_total = args.train_ratio + args.test_ratio + args.validation_ratio
    if abs(split_total - 1.0) > 1e-9:
        raise ValueError("train-ratio, test-ratio, and validation-ratio must sum to 1.0")

    prepare_directory(args.output)

    class_dirs = [path for path in sorted(args.input.iterdir()) if path.is_dir()]
    if not class_dirs:
        raise ValueError(f"No class folders found in: {args.input}")

    summary: dict[str, dict[str, int]] = {}
    for class_dir in class_dirs:
        summary[class_dir.name] = split_class_images(
            class_name=class_dir.name,
            class_dir=class_dir,
            output_root=args.output,
            train_ratio=args.train_ratio,
            test_ratio=args.test_ratio,
            validation_ratio=args.validation_ratio,
            seed=args.seed,
        )

    print(f"Dataset split created at: {args.output}")
    for class_name, counts in summary.items():
        print(
            f"{class_name}: train={counts['train']}, test={counts['test']}, validation={counts['validation']}"
        )


if __name__ == "__main__":
    main()