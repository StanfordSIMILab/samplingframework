import os
import json
from pathlib import Path
from typing import Union
import argparse
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(os.path.abspath('..'))

from div_sampling import parse_dataset_with_masks

#Helper function to used to sort colors by brightness
def luminance(rgb):
    r, g, b = rgb
    return 0.2126 * r + 0.7152 * g + 0.0722 * b

def parse_dataset_with_masks(
    dataset_root: Union[str, os.PathLike] = ".",
    mask_type: str = "color_mask",  # "color_mask" | "mask" | "watershed_mask"
    target_size: tuple[int, int] | None = (224, 224),  # (H, W); None keeps original
    videos: list[str] | None = None,  # e.g. ["video01", "video09"]; None = all
) -> tuple[np.ndarray, np.ndarray]:
    # <dataset_root>/videoXX/videoXX_NNNNN/frame_NNN_endo.png
    # <dataset_root>/videoXX/videoXX_NNNNN/frame_NNN_endo_color_mask.png
    dataset_root = Path(dataset_root)
    if not dataset_root.exists():
        raise FileNotFoundError(f"Dataset root not found: {dataset_root}")

    video_dirs = sorted(
        d for d in dataset_root.iterdir()
        if d.is_dir() and (videos is None or d.name in videos)
    )
    if not video_dirs:
        raise FileNotFoundError(f"No video folders found under {dataset_root}")

    frame_paths = []
    for video_dir in video_dirs:
        print("Doing ", video_dir)
        for sample_dir in sorted(video_dir.iterdir()):
            if sample_dir.is_dir():
                frame_paths.extend(sorted(sample_dir.glob("*_endo.png")))

    if not frame_paths:
        raise FileNotFoundError(f"No *_endo.png frames found under {dataset_root}")

    load_as_rgb = mask_type == "color_mask"
    mask_mode = "RGB" if load_as_rgb else "L"

    h, w = target_size if target_size is not None else Image.open(frame_paths[0]).size[::-1]
    frames = np.empty((len(frame_paths), h, w, 3), dtype=np.uint8)
    masks = np.empty((len(frame_paths), h, w, 3 if load_as_rgb else 1), dtype=np.uint8)

    for i, frame_path in enumerate(frame_paths):
        print(f"frame {i} of {len(frame_paths)} total", end="\r", flush=True)
        mask_path = frame_path.with_name(f"{frame_path.stem}_{mask_type}.png")
        if not mask_path.exists():
            raise FileNotFoundError(f"Mask not found: {mask_path}")
        frame_img = Image.open(frame_path).convert("RGB")
        mask_img  = Image.open(mask_path).convert(mask_mode)
        if target_size is not None:
            frame_img = frame_img.resize((w, h), Image.BILINEAR)
            mask_img  = mask_img.resize((w, h), Image.NEAREST)
        frames[i] = np.array(frame_img, dtype=np.uint8)
        masks[i]  = np.array(mask_img, dtype=np.uint8)

    print("\nFinished loading images")
    if load_as_rgb:
        N, H, W, _ = masks.shape

        # Flatten RGB pixels
        flat = masks.reshape(-1, 3)

        # Get unique colors
        unique_colors = np.unique(flat, axis=0)
        unique_colors = sorted(unique_colors, key=luminance)

        # Vectorized conversion (faster than per-color loop)
        flat_out = np.zeros((flat.shape[0],), dtype=np.uint8)

        for cls, rgb in enumerate(unique_colors):
            mask = np.all(flat == rgb, axis=1)
            flat_out[mask] = cls

        masks = flat_out.reshape(N, H, W)

        # Optional: keep mapping for visualization/debugging
        color_map = {
            idx: color.tolist()
            for idx, color in enumerate(unique_colors)
        }

    else:
        color_map = {}

    print("Finished processing masks. Unique classes found:", len(color_map))

    return frames, masks, color_map

def which_cholec_videos_are_representative():
    data_dir = "/mnt/sda1/nishanr/Diversity_Sampling/Cholec8K/data/raw"
    video_class_map = {}
    all_classes = set()

    for video_dir in sorted(Path(data_dir).iterdir()):
        if not video_dir.is_dir():
            continue
        _, masks, _ = parse_dataset_with_masks(
            dataset_root=data_dir,
            videos=[video_dir.name],
            mask_type="color_mask",
            dataset_style="cholec",
        )
        classes = set(np.unique(masks).tolist())
        video_class_map[video_dir.name] = classes
        all_classes.update(classes)
        print(f"{video_dir.name}: {len(classes)} classes — {sorted(classes)}")

    print(f"\nAll classes: {sorted(all_classes)} ({len(all_classes)} total)")

    # Show which videos cover the most classes
    for video, classes in sorted(video_class_map.items(), key=lambda x: len(x[1]), reverse=True):
        missing = all_classes - classes
        print(f"{video}: {len(classes)}/{len(all_classes)} classes, missing: {sorted(missing)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert raw Cholec8K frames and masks into numpy arrays for training Mask2Former"
    )
    parser.add_argument(
        "--base_dir", default="/mnt/sda1/nishanr/Diversity_Sampling/Cholec8K/data", type=str,
        help="Root directory of the full dataset where processed frames will be saved"
    )
    parser.add_argument(
        "--raw_data_dir", default="raw", type=str,
        help="Root directory of raw files dataset"
    )
    parser.add_argument(
        "--out_dir", default="processed_data", type=str,
        help="Out directory for the processed data"
    )
    parser.add_argument(
        "--video-representation", action="store_true",
        help="Determine each video's class coverage only"
    )
    args = parser.parse_args()

    if args.video_representation:
        which_cholec_videos_are_representative()
    else:
        data_dir = Path(args.base_dir) / args.raw_data_dir
        frames, masks, color_map = parse_cholec8k(videos=["video01", "video09", "video12", "video17", "video18", 
                                               "video20", "video24", "video25", "video26", "video27", 
                                               "video28", "video35", "video37", "video43", "video48", 
                                           "video52", "video55"], dataset_root=data_dir)
    
        output_dir = Path(args.base_dir) / args.out_dir
        os.makedirs(output_dir, exist_ok=True)

        with open(output_dir / "class_color_map.json", "w") as f:
            json.dump(color_map, f, indent=2)

        print(f"Frames: {frames.shape}  dtype={frames.dtype}")
        print(f"Masks:  {masks.shape}  dtype={masks.dtype}")
        print(f"Unique class IDs: {np.unique(masks).tolist()}")

        np.save(output_dir/"allcholec_frames.npy", frames)
        np.save(output_dir/"allcholec_masks.npy", masks)