# frame_extractor.py
# Process raw data (videos / frames / images) into a format suitable for diversity sampling
import sys
import os
import glob
import threading

import numpy as np
from PIL import Image
import cv2

import matplotlib.pyplot as plt

from . import fvi_computation as fvi_utils

# Valid video file extensions, can be extended as needed
VIDEO_EXTENSIONS = {'.mp4', '.avi', '.mov', '.mkv'}

# Loading Utilities for Processing Videos
def load_video(video_path=None, to_rgb=True, timeout=5.0):
    # Simply load all frames from the video w/ a timout for corrupted files
    if video_path is None:
        raise ValueError("video_path must be provided.")
    print(f"\nLoading video {video_path}...")
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")
    frames = []
    try:
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        while True:
            result = {}
            def _read(cap=cap, result=result):
                result["ret"], result["frame"] = cap.read()
            t = threading.Thread(target=_read)
            t.start()
            t.join(timeout=timeout)
            if t.is_alive():
                print(f"\nTimeout at frame {len(frames)}/{total_frames}, stopping.")
                break
            if not result.get("ret", False):
                break
            frame = result["frame"]
            if to_rgb:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
            print(f"{len(frames)}/{total_frames}", end="\r", flush=True)
    finally:
        cap.release()
    frames = np.asarray(frames)
    print(f"\nLoaded data (shape={frames.shape})")
    return frames

def process_video(video_path, output_dir=None, fvi_filtering=True, thresh=None, use_elbow=True, show_hist=True, save=True):
    frames = load_video(video_path)

    if len(frames) == 0:
        raise ValueError(
            f"No frames found in video: {video_path}"
        )

    if not fvi_filtering:
        filtered = frames
        kept = np.arange(len(frames))
        scores = np.array([])
        threshold = None
    else:
        filtered, kept, scores, threshold = fvi_utils.fvi_filter(
            frames,
            thresh=thresh,
            use_elbow=use_elbow,
        )

        if show_hist and len(scores) > 0:
            fvi_utils.show_fvi_histogram(scores)

        print(
            f"Threshold={threshold:.3f}, "
            f"kept {len(filtered)}/{len(frames)} frames"
        )

    if save:
        save_dir = (
            output_dir
            if output_dir is not None
            else os.path.dirname(os.path.abspath(video_path))
        )

        os.makedirs(save_dir, exist_ok=True)

        stem = os.path.splitext(
            os.path.basename(video_path)
        )[0]

        np.savez_compressed(
            os.path.join(save_dir, f"{stem}.npz"),
            frames=filtered,
            kept_indices=kept,
            fvi_scores=scores,
            threshold=threshold,
        )

    return filtered, kept, scores, threshold

def process_videos(input_dir, output_dir=None, fvi_filtering=True, thresh=None, use_elbow=True, show_hist=True, save=True):
    """
    Process all video files in the input directory and its subdirectories,
    applying FVI filtering and saving the results. The output directory
    mirrors the input directory structure, with one .npz per video:
    output_dir/
    ├── video_01.npz
    ├── subdir/
    │   ├── video_02.npz
    │   └── ...
    └── ...
    """
    # Convert input and output directories to absolute paths
    input_dir = os.path.abspath(input_dir)

    if output_dir is not None:
        output_dir = os.path.abspath(output_dir)
        os.makedirs(output_dir, exist_ok=True)

    print(f"Processing videos in {input_dir}...")

    all_frames = []

    # Walk recursively so subdirectories are included
    for root, dirs, files in os.walk(input_dir):
        print(f"Processing folder: {root}")

        for file in sorted(files):
            if not any(file.lower().endswith(ext) for ext in VIDEO_EXTENSIONS):
                continue

            video_path = os.path.join(root, file)
            print(f"Found video: {video_path}")

            # Mirror the input folder structure in the output directory
            if output_dir is not None:
                relative_path = os.path.relpath(root, input_dir)
                video_output_dir = os.path.join(output_dir, relative_path)
            else:
                video_output_dir = None

            filtered, _, _, _ = process_video(
                video_path=video_path,
                output_dir=video_output_dir,
                fvi_filtering=fvi_filtering,
                thresh=thresh,
                use_elbow=use_elbow,
                show_hist=show_hist,
                save=save,
            )

            all_frames.append(filtered)

    if not all_frames:
        raise ValueError(f"No video files found in {input_dir}")

    return all_frames