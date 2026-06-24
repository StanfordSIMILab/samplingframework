# data_partitioner.py - Utility for splitting processed dataset into train/val/test and saving as .npy files.
import os
import json
import numpy as np
from pathlib import Path
import re
from PIL import Image
import shutil


def train_val_test_split(
        data_folder: str | Path,
        test_size: float = 0.1,
        val_size: float = 0.0, # Set val_size to non-zero to enable full (train/val/test split), default (train/test)
        shuffle: bool = True,
        seed: int = 42,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:

    data_folder = Path(data_folder)
    frame_metadata = np.load(processed_dir / "frame_metadata.npy", allow_pickle=True) if (processed_dir / "frame_metadata.npy").exists() else None
    processed_dir = data_folder / "processed"

    if not processed_dir.exists():
        raise FileNotFoundError(
            f"No processed directory found at {processed_dir} — run data_manager first."
        )

    frames = np.load(processed_dir / "frames.npy")
    masks  = np.load(processed_dir / "masks.npy") if (processed_dir / "masks.npy").exists() else None
    labels = np.load(processed_dir / "labels.npy", allow_pickle=True) if (processed_dir / "labels.npy").exists() else None

    n = len(frames)
    indices = np.arange(n)

    if shuffle:
        rng = np.random.default_rng(seed)
        rng.shuffle(indices)

    n_test  = max(1, int(np.floor(n * test_size)))
    n_val   = max(1, int(np.floor(n * val_size))) if val_size > 0 else 0
    n_train = n - n_test - n_val

    if n_train <= 0:
        raise ValueError(
            f"test_size={test_size} + val_size={val_size} leaves no training samples "
            f"for a dataset of {n} frames."
        )

    train_idx = indices[:n_train]
    val_idx   = indices[n_train : n_train + n_val] if n_val > 0 else np.array([], dtype=int)
    test_idx  = indices[n_train + n_val :]

    split_data_dir = data_folder / "split_data"
    splits = {"train": train_idx, "test": test_idx}
    if n_val > 0:
        splits["val"] = val_idx

    for split_name, idx in splits.items():
        split_dir = split_data_dir / split_name
        split_dir.mkdir(parents=True, exist_ok=True)
        np.save(split_dir / "frames.npy", frames[idx])
        np.save(split_dir / "indices.npy", idx)
        if masks is not None:
            np.save(split_dir / "masks.npy", masks[idx])
        if labels is not None:
            np.save(split_dir / "labels.npy", labels[idx])
        if frame_metadata is not None:
            np.save(split_dir / "frame_metadata.npy", frame_metadata[idx])
            with open(processed_dir / "frame_metadata.json", "w") as f:
                json.dump({str(i): str(m) for i, m in enumerate(frame_metadata)}, f, indent=2)

    if (processed_dir / "color_map.json").exists():
        shutil.copy(processed_dir / "color_map.json", split_data_dir / "color_map.json")

    print(f"\nSplit summary: train={len(train_idx)}  val={len(val_idx)}  test={len(test_idx)}")
    print(f"Saved splits to {split_data_dir}")

    return train_idx, val_idx, test_idx


def split_by_directory(
        data_folder: str | Path,
        target_size: tuple[int, int] | None = None,
    ) -> None:

    data_folder = Path(data_folder)
    frame_metadata = np.load(processed_dir / "frame_metadata.npy", allow_pickle=True) if (processed_dir / "frame_metadata.npy").exists() else None
    split_data_dir = data_folder / "split_data"
    split_data_dir.mkdir(parents=True, exist_ok=True)
    processed_dir = data_folder / "processed"

    if (processed_dir / "frames.npy").exists():
        print(f"Loading from existing processed directory: {processed_dir}")
        frames = np.load(processed_dir / "frames.npy")
        masks = np.load(processed_dir / "masks.npy") if (processed_dir / "masks.npy").exists() else None
        labels = np.load(processed_dir / "labels.npy", allow_pickle=True) if (processed_dir / "labels.npy").exists() else None

        splits: dict[str, dict[str, np.ndarray]] = {}
        for f in sorted(data_folder.rglob("*")):
            if not f.is_file():
                continue
            if "processed" in f.parts or "split_data" in f.parts:
                continue
            if f.suffix.lower() not in {".jpg", ".jpeg", ".png", ".npy"}:
                continue
            split_name = infer_split_from_path(f.parts)
            splits.setdefault(split_name, {})

        if not splits:
            raise FileNotFoundError(
                f"No split-structured directories found under {data_folder} — "
                "cannot infer train/val/test from directory names."
            )

        n = len(frames)
        all_split_names = list(splits.keys())
        n_per_split = n // len(all_split_names)

        indices = np.arange(n)
        offset = 0
        for split_name in all_split_names:
            split_dir = split_data_dir / split_name
            split_dir.mkdir(parents=True, exist_ok=True)
            idx = indices[offset : offset + n_per_split]
            np.save(split_dir / "frames.npy", frames[idx])
            if masks is not None:
                np.save(split_dir / "masks.npy", masks[idx])
            if labels is not None:
                np.save(split_dir / "labels.npy", labels[idx])
            offset += n_per_split

        if (processed_dir / "color_map.json").exists():
            shutil.copy(processed_dir / "color_map.json", split_data_dir / "color_map.json")

        print(f"Saved directory-inferred splits to {split_data_dir}")
        for split_name in all_split_names:
            print(f"  {split_name}: {n_per_split} frames")
        return

    # Fall back to reading from raw directory structure if no processed/ exists
    splits: dict[str, dict[str, list]] = {}

    for f in sorted(data_folder.rglob("*")):
        if not f.is_file():
            continue
        if "processed" in f.parts or "split_data" in f.parts:
            continue
        if f.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
            continue

        split_name = infer_split_from_path(f.parts)
        if split_name not in splits:
            splits[split_name] = {"frames": [], "masks": []}

        is_mask = any(m in f.name.lower() for m in {"mask", "annotation"})
        img = Image.open(f).convert("RGB" if not is_mask else "L")
        if target_size is not None:
            h, w = target_size
            resample = Image.BILINEAR if not is_mask else Image.NEAREST
            img = img.resize((w, h), resample)

        arr = np.array(img, dtype=np.uint8)
        if is_mask:
            splits[split_name]["masks"].append(arr)
        else:
            splits[split_name]["frames"].append(arr)

    for split_name, arrays in splits.items():
        split_dir = split_data_dir / split_name
        split_dir.mkdir(parents=True, exist_ok=True)
        if arrays["frames"]:
            np.save(split_dir / "frames.npy", np.stack(arrays["frames"]))
        if arrays["masks"]:
            np.save(split_dir / "masks.npy", np.stack(arrays["masks"]))

    print(f"Saved directory-inferred splits to {split_data_dir}")
    for split_name, arrays in splits.items():
        print(f"  {split_name}: {len(arrays['frames'])} frames, {len(arrays['masks'])} masks")


def infer_split_from_path(parts: tuple[str, ...]) -> str:
    for part in parts:
        p = part.lower()
        if re.search(r'(^|[_\-\s])test($|[_\-\s\.])', p):
            return "test"
        if re.search(r'(^|[_\-\s])(val|validation)($|[_\-\s\.])', p):
            return "val"
        if re.search(r'(^|[_\-\s])train($|[_\-\s\.])', p):
            return "train"
    return "train"