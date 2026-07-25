from __future__ import annotations

import argparse
from pathlib import Path

import tensorflow as tf


def load_test_dataset(dataset_root: Path, image_size: tuple[int, int], batch_size: int):
    test_dir = dataset_root / "test"
    if not test_dir.exists():
        raise FileNotFoundError(f"Test folder not found: {test_dir}")

    test_ds = tf.keras.utils.image_dataset_from_directory(
        test_dir,
        label_mode="int",
        image_size=image_size,
        batch_size=batch_size,
        shuffle=False,
    )

    class_names = test_ds.class_names
    test_ds = test_ds.cache().prefetch(buffer_size=tf.data.AUTOTUNE)
    return test_ds, class_names


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the saved leaf recognition model on the test split.")
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path(__file__).parent / "split_dataset",
        help="Folder containing the test subfolder.",
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=Path(__file__).parent / "leaf_recognition_model.keras",
        help="Path to the trained Keras model.",
    )
    parser.add_argument("--image-size", type=int, default=180, help="Square image size used during training.")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size for loading the test set.")
    args = parser.parse_args()

    if not args.model_path.exists():
        raise FileNotFoundError(f"Model file not found: {args.model_path}")

    image_size = (args.image_size, args.image_size)
    test_ds, class_names = load_test_dataset(args.dataset_root, image_size, args.batch_size)

    print("Classes:", ", ".join(class_names))

    model = tf.keras.models.load_model(args.model_path)
    test_loss, test_accuracy = model.evaluate(test_ds)

    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_accuracy:.4f}")


if __name__ == "__main__":
    main()