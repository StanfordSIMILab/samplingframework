# data_manager.py: Functions for loading, parsing, and splitting datasets into train/val/test sets.
# Also includes DatasetManager class which wraps these functions and provides additional utilities.
import os
import json

import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image

from frame_extractor import load_video, process_video, process_videos, VIDEO_EXTENSIONS
from auxiliary import pitvis_extractor

# Different ways to load frames and masks, together or separately
def load_frames_and_masks(
        dataset_root: str | os.PathLike = ".",
        mask_type: str = "color_mask",
        target_size: tuple[int, int] | None = (224, 224),
        videos: list[str] | None = None,
        dataset_style: str = "cholec",
        force_color_processing: bool = False,
        global_color_map: dict | None = None,
    ) -> tuple[np.ndarray, np.ndarray, dict]:

    dataset_root = Path(dataset_root)
    if not dataset_root.exists():
        raise FileNotFoundError(f"Dataset root not found: {dataset_root}")

    load_as_rgb = mask_type == "color_mask" or force_color_processing
    mask_mode = "RGB" if load_as_rgb else "L"
    excluded = {"masks", "annotations", "mask"}

    frame_paths = []

    if dataset_style == "cholec":
        video_dirs = sorted(
            d for d in dataset_root.iterdir()
            if d.is_dir() and (videos is None or d.name in videos)
        )
        if not video_dirs:
            raise FileNotFoundError(f"No video folders found under {dataset_root}")
        for video_dir in video_dirs:
            print(f"Scanning {video_dir.name}...")
            for sample_dir in sorted(video_dir.iterdir()):
                if sample_dir.is_dir():
                    for f in sorted(sample_dir.iterdir()):
                        if any(exc in part for part in f.parts for exc in excluded):
                            continue
                        if f.suffix.lower() in {".jpg", ".jpeg", ".png"}:
                            frame_paths.append(f)

    elif dataset_style == "flat":
        for f in sorted(dataset_root.iterdir()):
            if any(exc in part for part in f.parts for exc in excluded):
                continue
            if f.suffix.lower() in {".jpg", ".jpeg", ".png"}:
                frame_paths.append(f)

    elif dataset_style == "nested":
        subdirs = sorted(
            d for d in dataset_root.iterdir()
            if d.is_dir() and (videos is None or d.name in videos)
            and not any(exc in d.name.lower() for exc in excluded)
        )
        if not subdirs:
            raise FileNotFoundError(f"No subfolders found under {dataset_root}")
        for subdir in subdirs:
            print(f"Scanning {subdir.name}...")
            for f in sorted(subdir.iterdir()):
                if any(exc in part for part in f.parts for exc in excluded):
                    continue
                if f.suffix.lower() in {".jpg", ".jpeg", ".png"}:
                    frame_paths.append(f)

    elif dataset_style == "pitvis":
        video_ids = (
            [int(v) for v in videos]
            if videos is not None
            else sorted(
                int(d.name)
                for d in dataset_root.iterdir()
                if d.is_dir() and d.name.isdigit()
            )
        )
        if not video_ids:
            raise FileNotFoundError(f"No numbered video folders found under {dataset_root}")

        # Load all annotations up front using the pitvis_extractor helper
        annot_index = pitvis_extractor.load_pitvis_annotations(dataset_root, video_ids)

        # Collect frame paths and track (video_id, frame_index) metadata in parallel
        pitvis_meta = []  # list of (int_video, int_time) tuples, one per frame
        for vid_id in video_ids:
            vid_dir = dataset_root / f"{vid_id:02d}"
            print(f"Scanning {vid_dir.name}...")
            for f in sorted(vid_dir.rglob("*")):
                if any(exc in part for part in f.parts for exc in excluded):
                    continue
                if f.suffix.lower() in {".jpg", ".jpeg", ".png"}:
                    frame_paths.append(f)
                    # Frame index is encoded in the filename stem as an integer timestamp
                    pitvis_meta.append((vid_id, int(f.stem)))

    else:
        raise ValueError(
            f"Unknown dataset_style '{dataset_style}'. Choose 'cholec', 'flat', 'nested', or 'pitvis'."
        )

    if not frame_paths:
        raise FileNotFoundError(
            f"No image frames found under {dataset_root} with style='{dataset_style}'"
        )

    print(f"Found {len(frame_paths)} frames.")

    if target_size is not None:
        h, w = target_size
    else:
        h, w = Image.open(frame_paths[0]).size[::-1]

    n = len(frame_paths)
    frames = np.empty((n, h, w, 3), dtype=np.uint8)
    masks  = np.empty((n, h, w, 3 if load_as_rgb else 1), dtype=np.uint8)

    missing_masks = []

    for i, frame_path in enumerate(frame_paths):
        print(f"Loading frame {i+1}/{n}", end="\r", flush=True)

        mask_name = f"{frame_path.stem}_{mask_type}.png"
        mask_path = frame_path.parent / mask_name

        if not mask_path.exists():
            missing_masks.append(str(mask_path))
            continue

        frame_img = Image.open(frame_path).convert("RGB")
        mask_img  = Image.open(mask_path).convert(mask_mode)

        if target_size is not None:
            frame_img = frame_img.resize((w, h), Image.BILINEAR)
            mask_img  = mask_img.resize((w, h), Image.NEAREST)

        frames[i] = np.array(frame_img, dtype=np.uint8)
        mask_np   = np.array(mask_img, dtype=np.uint8)
        masks[i]  = mask_np if mask_np.ndim == 3 else mask_np[..., np.newaxis]

    if missing_masks:
        raise FileNotFoundError(
            f"{len(missing_masks)} mask(s) not found. First missing:\n  {missing_masks[0]}"
        )

    print(f"\nFinished loading {n} frames.")

    color_map = {}
    if load_as_rgb:
        if global_color_map is None:
            flat = masks.reshape(-1, 3)
            unique_colors = np.unique(flat, axis=0)
            unique_colors = sorted(unique_colors, key=luminance)
            global_color_map = {
                idx: list(color)
                for idx, color in enumerate(unique_colors)
            }

        rgb_to_cls = {tuple(color): idx for idx, color in global_color_map.items()}
        N, H, W, _ = masks.shape
        flat = masks.reshape(-1, 3)
        flat_out = np.zeros(flat.shape[0], dtype=np.uint8)
        for rgb_tuple, cls in rgb_to_cls.items():
            flat_out[np.all(flat == np.array(rgb_tuple), axis=1)] = cls
        masks = flat_out.reshape(N, H, W)
        color_map = global_color_map
        print(f"Finished processing masks. Unique classes found: {len(color_map)}")
    else:
        masks = masks.squeeze(-1)
        unique_vals = np.unique(masks)
        print(f"Finished processing masks. Unique values found: {unique_vals}")

    # For pitvis datasets, resolve annotations into a labels array and attach it
    # to the return value as an extra element so callers can unpack it.
    if dataset_style == "pitvis":
        labels = pitvis_extractor.get_pitvis_labels(annot_index, pitvis_meta)
        print(f"Finished loading pitvis labels. Shape: {labels.shape}")
        return frames, masks, color_map, labels

    return frames, masks, color_map


def load_frames_from_dir(
        data_dir: str | Path,
        target_size: tuple[int, int] | None = None,
        exclude: set[str] | None = None,
    ) -> np.ndarray:

    data_dir = Path(data_dir)
    excluded = exclude or {"masks", "annotations", "mask"}
    frames_list = []

    for frame in sorted(data_dir.rglob("*")):
        if any(exc in part for part in frame.parts for exc in excluded):
            continue
        if frame.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
            continue
        img = Image.open(frame).convert("RGB")
        if target_size is not None:
            h, w = target_size
            img = img.resize((w, h), Image.BILINEAR)
        frames_list.append(np.array(img, dtype=np.uint8))

    if not frames_list:
        raise FileNotFoundError(f"No image files found under {data_dir}")

    return np.stack(frames_list, axis=0)


def load_masks_from_dir(
        mask_dir: str | Path,
        target_size: tuple[int, int] | None = None,
        mask_mode: str = "L",
        include: set[str] | None = None,
    ) -> np.ndarray:

    mask_dir = Path(mask_dir)
    included = include or {"masks", "annotations", "mask"}
    masks_list = []

    for mask_path in sorted(mask_dir.rglob("*")):
        if not any(inc in part for part in mask_path.parts for inc in included):
            continue
        if mask_path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
            continue
        mask_img = Image.open(mask_path).convert(mask_mode)
        if target_size is not None:
            h, w = target_size
            mask_img = mask_img.resize((w, h), Image.NEAREST)
        masks_list.append(np.array(mask_img, dtype=np.uint8))

    if not masks_list:
        raise FileNotFoundError(f"No mask files found under {mask_dir}")

    return np.stack(masks_list, axis=0)


def export_frames(
        frames: np.ndarray,
        out_folder: str | Path,
        is_mask: bool = False,
    ) -> None:

    out_folder = Path(out_folder)
    out_folder.mkdir(parents=True, exist_ok=True)

    for i, img in enumerate(frames):
        if img.ndim == 2:
            pil_img = Image.fromarray(img, mode="L")
        elif img.ndim == 3 and img.shape[-1] == 1:
            pil_img = Image.fromarray(img.squeeze(-1), mode="L")
        elif img.ndim == 3 and img.shape[-1] == 3:
            pil_img = Image.fromarray(img, mode="RGB")
        else:
            raise ValueError(f"Unsupported image shape: {img.shape}")

        filename = f"{i:06d}_mask.png" if is_mask else f"{i:06d}.png"
        pil_img.save(out_folder / filename)

# Build color map if color mask identities are not provided
def luminance(rgb):
    r, g, b = rgb
    return 0.2126 * r + 0.7152 * g + 0.0722 * b

def build_color_map(masks: np.ndarray, out_folder: Path | None) -> dict:
    flat = masks.reshape(-1, 3)
    unique_colors = np.unique(flat, axis=0)
    unique_colors = sorted(unique_colors, key=luminance)

    if out_folder is not None:
        os.makedirs(out_folder, exist_ok=True)
        with open(out_folder / "color_map.json", "w") as f:
            json.dump({str(idx): color.tolist() for idx, color in enumerate(unique_colors)}, f, indent=2)
    return {idx: list(color) for idx, color in enumerate(unique_colors)}


def apply_color_map(masks: np.ndarray, color_map: dict) -> np.ndarray:
    rgb_to_cls = {tuple(color): idx for idx, color in color_map.items()}
    N, H, W, _ = masks.shape
    flat = masks.reshape(-1, 3)
    flat_out = np.zeros(flat.shape[0], dtype=np.uint8)
    for rgb_tuple, cls in rgb_to_cls.items():
        flat_out[np.all(flat == np.array(rgb_tuple), axis=1)] = cls
    return flat_out.reshape(N, H, W)
