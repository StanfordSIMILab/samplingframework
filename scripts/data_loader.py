# data_loader.py - Functions for loading and parsing datasets.
import os
import json
import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image
from tqdm import tqdm

from auxiliary.coco_converter import convert_coco_to_png_masks
from auxiliary.frame_extractor import load_video, process_video, process_videos, VIDEO_EXTENSIONS
from auxiliary import pitvis_extractor

# For loading the frames and masks for segmentation
def load_frames_and_masks(
        data_folder: str | os.PathLike,
        output_folder: str | Path | None = None,
        mask_type: str = "color_mask",
        target_size: tuple[int, int] | None = (224, 224),
        videos: list[str] | None = None,
        dataset_style: str = "cholec",
        force_color_processing: bool = False,
        global_color_map: dict | None = None,
    ) -> tuple[np.ndarray, np.ndarray, dict]:

    data_folder = Path(data_folder)
    if not data_folder.exists():
        raise FileNotFoundError(f"Data folder not found: {data_folder}")

    processed_dir = Path(output_folder) / "processed" if output_folder is not None else data_folder / "processed"

    if processed_dir.exists():
        print(f"Loading from existing processed directory: {processed_dir}")
        frames = np.load(processed_dir / "frames.npy")
        masks  = np.load(processed_dir / "masks.npy")
        frame_metadata = np.load(processed_dir / "frame_metadata.npy", allow_pickle=True) if (processed_dir / "frame_metadata.npy").exists() else None
        with open(processed_dir / "color_map.json") as f:
            color_map = {int(k): v for k, v in json.load(f).items()}
        if (processed_dir / "labels.npy").exists():
            labels = np.load(processed_dir / "labels.npy", allow_pickle=True)
            return frames, masks, color_map, labels, frame_metadata
        return frames, masks, color_map, frame_metadata

    load_as_rgb = mask_type == "color_mask" or force_color_processing
    mask_mode = "RGB" if load_as_rgb else "L"
    excluded = {"masks", "annotations", "mask", "processed", "split_data", "outputs"}

    frame_paths = []
    frame_metadata = []

    if dataset_style == "cholec":
        video_dirs = sorted(
            d for d in data_folder.iterdir()
            if d.is_dir() and (videos is None or d.name in videos)
            and not any(exc in d.name.lower() for exc in excluded)
        )
        if not video_dirs:
            raise FileNotFoundError(f"No video folders found under {data_folder}")
        for video_dir in video_dirs:
            print(f"Scanning {video_dir.name}...")
            for sample_dir in sorted(video_dir.iterdir()):
                if sample_dir.is_dir():
                    for f in sorted(sample_dir.iterdir()):
                        if any(exc in part for part in f.parts for exc in excluded):
                            continue
                        if f.suffix.lower() in {".jpg", ".jpeg", ".png"}:
                            frame_paths.append(f)
                            frame_metadata.append(f"{video_dir.name}_{f.stem}")

    elif dataset_style == "flat":
        for f in sorted(data_folder.iterdir()):
            if any(exc in part for part in f.parts for exc in excluded):
                continue
            if f.suffix.lower() in {".jpg", ".jpeg", ".png"}:
                frame_paths.append(f)
                frame_metadata.append(f.stem)

    elif dataset_style == "nested":
        subdirs = sorted(
            d for d in data_folder.iterdir()
            if d.is_dir() and (videos is None or d.name in videos)
            and not any(exc in d.name.lower() for exc in excluded)
        )
        if not subdirs:
            raise FileNotFoundError(f"No subfolders found under {data_folder}")
        for subdir in subdirs:
            print(f"Scanning {subdir.name}...")
            for f in sorted(subdir.iterdir()):
                if any(exc in part for part in f.parts for exc in excluded):
                    continue
                if f.suffix.lower() in {".jpg", ".jpeg", ".png"}:
                    frame_paths.append(f)
                    frame_metadata.append(f"{subdir.name}_{f.stem}")

    elif dataset_style == "pitvis":
        video_ids = (
            [int(v) for v in videos]
            if videos is not None
            else sorted(
                int(d.name)
                for d in data_folder.iterdir()
                if d.is_dir() and d.name.isdigit()
            )
        )
        if not video_ids:
            raise FileNotFoundError(f"No numbered video folders found under {data_folder}")

        annot_index = pitvis_extractor.load_pitvis_annotations(data_folder, video_ids)

        pitvis_meta = []
        for vid_id in video_ids:
            vid_dir = data_folder / f"{vid_id:02d}"
            print(f"Scanning {vid_dir.name}...")
            for f in sorted(vid_dir.rglob("*")):
                if any(exc in part for part in f.parts for exc in excluded):
                    continue
                if f.suffix.lower() in {".jpg", ".jpeg", ".png"}:
                    frame_paths.append(f)
                    pitvis_meta.append((vid_id, int(f.stem)))
                    frame_metadata.append(f"video_{vid_id:02d}_{f.stem}")

    else:
        raise ValueError(
            f"Unknown dataset_style '{dataset_style}'. Choose 'cholec', 'flat', 'nested', or 'pitvis'."
        )

    if not frame_paths:
        raise FileNotFoundError(
            f"No image frames found under {data_folder} with style='{dataset_style}'"
        )

    print(f"Found {len(frame_paths)} frames.")

    dirs_needing_conversion: set[Path] = set()
    for frame_path in frame_paths:
        mask_path = frame_path.parent / f"{frame_path.stem}_{mask_type}.png"
        if not mask_path.exists():
            dirs_needing_conversion.add(frame_path.parent)

    for d in dirs_needing_conversion:
        coco_path = d / "coco.json"
        if coco_path.exists():
            print(f"No mask PNGs found in {d.name}, converting from {coco_path.name}...")
            with open(coco_path) as f:
                coco_data = json.load(f)
            convert_coco_to_png_masks(coco_data, str(d))
        else:
            raise FileNotFoundError(
                f"No mask PNGs and no coco.json found in {d} — cannot load masks."
            )

    if target_size is not None:
        h, w = target_size
    else:
        h, w = Image.open(frame_paths[0]).size[::-1]

    n = len(frame_paths)
    frames = np.empty((n, h, w, 3), dtype=np.uint8)
    masks  = np.empty((n, h, w, 3 if load_as_rgb else 1), dtype=np.uint16)

    missing_masks = []

    for i, frame_path in enumerate(frame_paths):
        print(f"Loading frame {i+1}/{n}", end="\r", flush=True)

        mask_path = frame_path.parent / f"{frame_path.stem}_{mask_type}.png"

        if not mask_path.exists():
            missing_masks.append(str(mask_path))
            continue

        frame_img = Image.open(frame_path).convert("RGB")
        mask_img  = Image.open(mask_path).convert(mask_mode)

        if target_size is not None:
            frame_img = frame_img.resize((w, h), Image.BILINEAR)
            mask_img  = mask_img.resize((w, h), Image.NEAREST)

        frames[i] = np.array(frame_img, dtype=np.uint8)
        mask_np   = np.array(mask_img, dtype=np.uint16)
        masks[i]  = mask_np if mask_np.ndim == 3 else mask_np[..., np.newaxis]

    if missing_masks:
        raise FileNotFoundError(
            f"{len(missing_masks)} mask(s) not found. First missing:\n  {missing_masks[0]}"
        )

    print(f"\nFinished loading {n} frames.")

    color_map = {}
    if load_as_rgb:
        print("Building color map...", flush=True)
        if global_color_map is None:
            flat = masks.reshape(-1, 3)
            print(f"Finding unique colors (sampling from {flat.shape[0]} pixels)...", flush=True)
            if len(flat) > 1_000_000:
                rng = np.random.default_rng(42)
                sample_idx = rng.choice(len(flat), size=1_000_000, replace=False)
                sample = flat[sample_idx]
            else:
                sample = flat
            unique_colors = np.unique(sample, axis=0)
            print(f"Found {len(unique_colors)} unique colors, sorting...", flush=True)
            unique_colors = sorted(unique_colors, key=luminance)
            global_color_map = {
                idx: [int(c) for c in color]
                for idx, color in enumerate(unique_colors)
            }
            print(f"Color map built with {len(global_color_map)} entries", flush=True)

        rgb_to_cls = {tuple(color): idx for idx, color in global_color_map.items()}
        N, H, W, _ = masks.shape
        flat = masks.reshape(-1, 3)
        flat_out = np.zeros(flat.shape[0], dtype=np.uint8)
        print(f"Mapping {len(rgb_to_cls)} colors to class indices across {flat.shape[0]} pixels...", flush=True)
        flat_view = flat[:, 0].astype(np.uint32) * 65536 + flat[:, 1].astype(np.uint32) * 256 + flat[:, 2].astype(np.uint32)
        for rgb_tuple, cls in tqdm(rgb_to_cls.items(), desc="Mapping colors to classes", total=len(rgb_to_cls)):
            key = int(rgb_tuple[0]) * 65536 + int(rgb_tuple[1]) * 256 + int(rgb_tuple[2])
            flat_out[flat_view == key] = cls
        masks = flat_out.reshape(N, H, W)
        color_map = global_color_map
        print(f"Finished processing masks. Unique classes found: {len(color_map)}", flush=True)
    else:
        print("Processing grayscale masks...", flush=True)
        masks = masks.squeeze(-1)
        print(f"Remapping {len(np.unique(masks))} unique values to contiguous indices...", flush=True)
        unique_vals = np.unique(masks)
        val_to_idx = {v: i for i, v in enumerate(unique_vals)}
        remapped = np.zeros_like(masks, dtype=np.uint8)
        for val, idx in tqdm(val_to_idx.items(), desc="Remapping mask values", total=len(val_to_idx)):
            remapped[masks == val] = idx
        masks = remapped
        color_map = {i: int(v) for i, v in enumerate(unique_vals)}
        print(f"Finished processing masks. Unique values remapped: {len(color_map)}", flush=True)

    frame_metadata = np.array(frame_metadata)

    processed_dir.mkdir(parents=True, exist_ok=True)
    np.save(processed_dir / "frames.npy", frames)
    np.save(processed_dir / "masks.npy", masks)
    np.save(processed_dir / "frame_metadata.npy", frame_metadata)
    with open(processed_dir / "frame_metadata.json", "w") as f:
        json.dump({str(i): str(m) for i, m in enumerate(frame_metadata)}, f, indent=2)
    with open(processed_dir / "color_map.json", "w") as f:
        json.dump({str(k): v for k, v in color_map.items()}, f, indent=2)
    print(f"Saved processed data to {processed_dir}")

    if dataset_style == "pitvis":
        labels = pitvis_extractor.get_pitvis_labels(annot_index, pitvis_meta)
        np.save(processed_dir / "labels.npy", labels)
        print(f"Finished loading pitvis labels. Shape: {labels.shape}")
        return frames, masks, color_map, labels, frame_metadata

    return frames, masks, color_map, frame_metadata, 

# For if using phase classification / unannotated data, simply loads just the frames
def load_frames_from_dir(
        data_folder: str | Path,
        output_folder: str | Path | None = None,
        target_size: tuple[int, int] | None = None,
        exclude: set[str] | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:

    data_folder = Path(data_folder)
    excluded = exclude or {"masks", "annotations", "mask", "processed", "split_data"}
    
    # Set output folders
    if output_folder is not None:
        processed_dir = Path(output_folder) / "processed"
    else:
        processed_dir = data_folder / "processed"

    if (processed_dir / "frames.npy").exists():
        print(f"Loading from existing processed directory: {processed_dir}")
        frames = np.load(processed_dir / "frames.npy")
        frame_metadata = np.load(processed_dir / "frame_metadata.npy", allow_pickle=True) if (processed_dir / "frame_metadata.npy").exists() else None
        return frames, frame_metadata

    frames_list = []
    frame_metadata = []

    for frame in sorted(data_folder.rglob("*")):
        if any(exc in part for part in frame.parts for exc in excluded):
            continue
        if frame.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
            continue
        img = Image.open(frame).convert("RGB")
        if target_size is not None:
            h, w = target_size
            img = img.resize((w, h), Image.BILINEAR)
        frames_list.append(np.array(img, dtype=np.uint8))
        frame_metadata.append(frame.stem)

    if not frames_list:
        raise FileNotFoundError(f"No image files found under {data_folder}")

    frames = np.stack(frames_list, axis=0)
    frame_metadata = np.array(frame_metadata)

    processed_dir.mkdir(parents=True, exist_ok=True)
    np.save(processed_dir / "frames.npy", frames)
    np.save(processed_dir / "frame_metadata.npy", frame_metadata)
    with open(processed_dir / "frame_metadata.json", "w") as f:
        json.dump({str(i): str(m) for i, m in enumerate(frame_metadata)}, f, indent=2)
    print(f"Saved processed frames to {processed_dir}")

    return frames, frame_metadata

# export the frames in proper format to out_folder
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

# Helper function for differentiating colors in a color mask with no keys
def luminance(rgb):
    r, g, b = rgb
    return 0.2126 * r + 0.7152 * g + 0.0722 * b

# Helper function that uses the luminance functions to build a color map of class -> color
def build_color_map(masks: np.ndarray, out_folder: Path | None) -> dict:
    flat = masks.reshape(-1, 3)
    unique_colors = np.unique(flat, axis=0)
    unique_colors = sorted(unique_colors, key=luminance)

    if out_folder is not None:
        os.makedirs(out_folder, exist_ok=True)
        with open(out_folder / "color_map.json", "w") as f:
            json.dump({str(idx): color.tolist() for idx, color in enumerate(unique_colors)}, f, indent=2)
    return {idx: list(color) for idx, color in enumerate(unique_colors)}

# convert the color into a class identity
def apply_color_map(masks: np.ndarray, color_map: dict) -> np.ndarray:
    rgb_to_cls = {tuple(color): idx for idx, color in color_map.items()}
    N, H, W, _ = masks.shape
    flat = masks.reshape(-1, 3)
    flat_out = np.zeros(flat.shape[0], dtype=np.uint16)
    for rgb_tuple, cls in rgb_to_cls.items():
        flat_out[np.all(flat == np.array(rgb_tuple), axis=1)] = cls
    return flat_out.reshape(N, H, W)
