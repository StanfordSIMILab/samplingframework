# data_partitioner.py - Utility for splitting dataset into train/val/test and saving as .npy files.
import os
import json
import numpy as np
from pathlib import Path

def train_val_test_split(
    data_arr: np.ndarray,
    test_size: float = 0.1,
    val_size: float = 0.1,
    out_folder: str | Path = ".",
    mask_arr: np.ndarray | None = None,
    shuffle: bool = True,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = len(data_arr)
    indices = np.arange(n)
    if shuffle:
        rng = np.random.default_rng(seed)
        rng.shuffle(indices)
    n_test = max(1, int(np.floor(n * test_size)))
    n_val  = max(1, int(np.floor(n * val_size)))
    n_train = n - n_test - n_val
    if n_train <= 0:
        raise ValueError(
            f"test_size={test_size} + val_size={val_size} leaves no training samples "
            f"for a dataset of {n} frames."
        )
    train_idx = indices[:n_train]
    val_idx   = indices[n_train : n_train + n_val]
    test_idx  = indices[n_train + n_val :]
    splits = {"train": train_idx, "val": val_idx, "test": test_idx}
    for split_name, idx in splits.items():
        split_dir = os.path.join(out_folder, split_name)
        os.makedirs(split_dir, exist_ok=True)
        np.save(os.path.join(split_dir, "frames.npy"), data_arr[idx])
        np.save(os.path.join(split_dir, "indices.npy"), idx)
        if mask_arr is not None:
            np.save(os.path.join(split_dir, "masks.npy"), mask_arr[idx])
    print(f"\nSplit summary: train={len(train_idx)}  val={len(val_idx)}  test={len(test_idx)}")
    return train_idx, val_idx, test_idx


def split_by_directory_name(
    frames_by_split: dict[str, np.ndarray],
    masks_by_split: dict[str, np.ndarray] | None = None,
    color_map: dict | None = None,
    out_folder: str | Path = "split_data",
) -> None:
    out_folder = Path(out_folder)
    out_folder.mkdir(parents=True, exist_ok=True)
    for split_name, data_arr in frames_by_split.items():
        if masks_by_split is not None and split_name in masks_by_split:
            if len(data_arr) != len(masks_by_split[split_name]):
                raise ValueError(
                    f"Frame count ({len(data_arr)}) and mask count "
                    f"({len(masks_by_split[split_name])}) don't match for split '{split_name}'."
                )
    for split_name, data_arr in frames_by_split.items():
        split_dir = out_folder / split_name
        split_dir.mkdir(parents=True, exist_ok=True)
        np.save(split_dir / "frames.npy", data_arr)
        if masks_by_split is not None and split_name in masks_by_split:
            np.save(split_dir / "masks.npy", masks_by_split[split_name])
    if color_map is not None:
        with open(out_folder / "color_map.json", "w") as f:
            json.dump({str(k): v for k, v in color_map.items()}, f, indent=2)
    print("Finished saving all splits.")


def infer_split_from_path(parts: tuple[str, ...]) -> str:
    parts_lower = [p.lower() for p in parts]
    if any("test" in p for p in parts_lower):
        return "test"
    if any("val" in p for p in parts_lower):
        return "val"
    return "train"