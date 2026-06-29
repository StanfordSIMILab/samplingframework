# data_partitioner.py - utilities for splitting processed dataset into train/val/test and saving as .npy files.
import os
import json
import numpy as np
from pathlib import Path
import re
from PIL import Image
import shutil

# Split dataset into train/val/test or train/test based on provided proportions
def train_val_test_split(
        data_folder: str | Path,
        test_size: float = 0.1,
        val_size: float = 0.0, # Set val_size to non-zero to enable full (train/val/test split), default (train/test)
        shuffle: bool = True,
        seed: int = 42,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:

    data_folder = Path(data_folder)
    processed_dir = data_folder / "processed"

    if not processed_dir.exists():
        raise FileNotFoundError(
            f"No processed directory found at {processed_dir} — run data_manager first."
        )
    
    frame_metadata = np.load(processed_dir / "frame_metadata.npy", allow_pickle=True) if (processed_dir / "frame_metadata.npy").exists() else None

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
            # Output index -> frame mappings in numerical order
            with open(split_dir / "frame_metadata.json", "w") as f:
                    json.dump({
                        str(pos): {
                            "index": int(i),
                            "frame_name": str(frame_metadata[i])
                        }
                        for pos, i in enumerate(sorted(idx))
                    }, f, indent=2)

    if (processed_dir / "color_map.json").exists():
        shutil.copy(processed_dir / "color_map.json", split_data_dir / "color_map.json")

    print(f"\nSplit summary: train={len(train_idx)}  val={len(val_idx)}  test={len(test_idx)}")
    print(f"Saved splits to {split_data_dir}")

    return train_idx, val_idx, test_idx

# If any folder / frame names contain words like train/val(idation)/test, split on these naming conventions
def split_by_directory(
        data_folder: str | Path,
        target_size: tuple[int, int] | None = None,
    ) -> None:

    data_folder = Path(data_folder)
    split_data_dir = data_folder / "split_data"
    split_data_dir.mkdir(parents=True, exist_ok=True)
    processed_dir = data_folder / "processed"

    if not (processed_dir / "frames.npy").exists():
        raise FileNotFoundError(
            f"No processed data found at {processed_dir} — run data_manager first."
        )

    frames = np.load(processed_dir / "frames.npy")
    masks = np.load(processed_dir / "masks.npy") if (processed_dir / "masks.npy").exists() else None
    labels = np.load(processed_dir / "labels.npy", allow_pickle=True) if (processed_dir / "labels.npy").exists() else None
    frame_metadata = np.load(processed_dir / "frame_metadata.npy", allow_pickle=True) if (processed_dir / "frame_metadata.npy").exists() else None

    if frame_metadata is None:
        raise FileNotFoundError(
            f"No frame_metadata.npy found at {processed_dir} — cannot infer splits without frame provenance."
        )

    split_indices: dict[str, list[int]] = {}
    for i, meta in enumerate(frame_metadata):
        split_name = infer_split_from_path((str(meta),))
        split_indices.setdefault(split_name, []).append(i)

    if not split_indices:
        raise FileNotFoundError(
            f"No split-structured directories found in frame metadata — "
            "cannot infer train/val/test from frame names."
        )

    for split_name, idx_list in split_indices.items():
        idx = np.array(idx_list)
        split_dir = split_data_dir / split_name
        split_dir.mkdir(parents=True, exist_ok=True)

        np.save(split_dir / "frames.npy", frames[idx])
        np.save(split_dir / "indices.npy", idx)

        if masks is not None:
            np.save(split_dir / "masks.npy", masks[idx])
        if labels is not None:
            np.save(split_dir / "labels.npy", labels[idx])

        np.save(split_dir / "frame_metadata.npy", frame_metadata[idx])
        # Output index -> frame mappings in numerical order
        with open(split_dir / "frame_metadata.json", "w") as f:
                json.dump({
                    str(pos): {
                        "index": int(i),
                        "frame_name": str(frame_metadata[i])
                    }
                    for pos, i in enumerate(sorted(idx))
                }, f, indent=2)

    if (processed_dir / "color_map.json").exists():
        shutil.copy(processed_dir / "color_map.json", split_data_dir / "color_map.json")

    print(f"Saved directory-inferred splits to {split_data_dir}")
    for split_name, idx_list in split_indices.items():
        print(f"  {split_name}: {len(idx_list)} frames")

# Helper function to figure out if it is train/val/test (train by default)
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