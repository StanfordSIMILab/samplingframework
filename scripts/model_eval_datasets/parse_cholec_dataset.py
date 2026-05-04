import os
import numpy as np
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt

def parse_cholec8k(
    dataset_root: str | os.PathLike = "Cholec8k_dataset/raw_data",
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

    frames_list: list[np.ndarray] = []
    masks_list: list[np.ndarray] = []

    for i, frame_path in enumerate(frame_paths):
        print(i, " ", len(frame_paths))
        mask_path = frame_path.with_name(f"{frame_path.stem}_{mask_type}.png")
        if not mask_path.exists():
            raise FileNotFoundError(f"Mask not found: {mask_path}")

        frame_img = Image.open(frame_path).convert("RGB")
        mask_img  = Image.open(mask_path).convert(mask_mode)

        if target_size is not None:
            h, w = target_size
            frame_img = frame_img.resize((w, h), Image.BILINEAR)
            mask_img  = mask_img.resize((w, h), Image.NEAREST)

        frames_list.append(np.array(frame_img, dtype=np.uint8))
        masks_list.append(np.array(mask_img,   dtype=np.uint8))

    frames = np.stack(frames_list, axis=0)  # (N, H, W, 3)

    if load_as_rgb:
        masks_rgb = np.stack(masks_list, axis=0)          # (N, H, W, 3)
        N, H, W, _ = masks_rgb.shape
        flat = masks_rgb.reshape(-1, 3)                    # (N*H*W, 3)
        _, inverse = np.unique(flat, axis=0, return_inverse=True)
        masks = inverse.reshape(N, H, W).astype(np.uint8)  # (N, H, W) class IDs
    else:
        masks = np.stack(masks_list, axis=0)               # (N, H, W) raw grayscale

    return frames, masks


# if __name__ == "__main__":
#     frames, masks = parse_cholec8k(videos=None)
#     print(f"Frames: {frames.shape}  dtype={frames.dtype}")
#     print(f"Masks:  {masks.shape}  dtype={masks.dtype}")
#     print(f"Unique class IDs: {np.unique(masks).tolist()}")

#     np.save("cholec8080_frames.npy", frames)
#     np.save("cholec8080_masks.npy", masks)

# check = np.load("cholec8080_masks.npy")
# plt.imshow(check[0])
# plt.show()
