#pitvis_extractor.py - PitVis-specific functions for loading annotations and extracting labels based on video/frame metadata
import numpy as np
import pandas as pd
import os
import cv2
from pathlib import Path
from PIL import Image

VIDEO_EXTENSIONS = {'.mp4', '.avi', '.mov', '.mkv'}

def extract_pitvis_frames(data_dir: Path, video_ids: list) -> None:
    for vid_id in video_ids:
        vid_dir = data_dir / f"{vid_id:02d}"
        if vid_dir.exists() and any(vid_dir.iterdir()):
            print(f"Frames already extracted for video {vid_id:02d}, skipping...")
            continue

        video_path = None
        for ext in VIDEO_EXTENSIONS:
            for candidate in [
                data_dir / f"video_{vid_id:02d}{ext}",
                data_dir / f"video{vid_id:02d}{ext}",
                data_dir / f"{vid_id:02d}{ext}",
            ]:
                if candidate.exists():
                    video_path = candidate
                    break
            if video_path:
                break

        if video_path is None:
            raise FileNotFoundError(
                f"No video file found for video_id={vid_id} in {data_dir}"
            )

        vid_dir.mkdir(exist_ok=True)
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration_secs = int(total_frames / fps)

        print(f"Extracting {video_path.name} at 1fps ({duration_secs} frames) -> {vid_dir}")

        for sec in range(duration_secs):
            frame_pos = int(sec * fps)
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_pos)
            ret, frame = cap.read()
            if not ret:
                break
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            Image.fromarray(frame_rgb).save(vid_dir / f"{sec:06d}.jpg")
            print(f"  {sec}/{duration_secs}", end="\r", flush=True)

        cap.release()
        print(f"\n  Done — {duration_secs} frames saved to {vid_dir}")

def load_pitvis_annotations(data_dir: Path, video_ids: list) -> pd.DataFrame:
    dfs = [pd.read_csv(data_dir / f"annotations_{v:02d}.csv") for v in video_ids]
    return pd.concat(dfs, ignore_index=True).set_index(["int_video", "int_time"])

# Load the map_steps.csv file and return a dictionary mapping raw integer labels to string class names, along with the number of classes.
def load_pitvis_phase_map(map_phase_csv: str | Path) -> tuple[dict, int, str]:
    df = pd.read_csv(map_phase_csv)

    real = df[df["int_step"] >= 0].drop_duplicates(subset="int_step").sort_values("int_step")
    class_names_by_raw_label = dict(zip(real["int_step"].astype(int), real["str_step"].str.strip()))
    num_classes = len(class_names_by_raw_label)

    negative = df[df["int_step"] < 0]
    unlabeled_name = ", ".join(negative["str_step"].str.strip().tolist())

    return class_names_by_raw_label, num_classes, unlabeled_name

# Load the map_instruments.csv file and return a dictionary mapping raw integer labels to string class names, along with the number of classes and a dictionary for negative labels.
def load_pitvis_instrument_map(map_instrument_csv: str | Path) -> tuple[dict, int, dict]:
    df = pd.read_csv(map_instrument_csv)
    real = df[df["int_instrument"] >= 0].drop_duplicates(subset="int_instrument").sort_values("int_instrument")
    class_names_by_raw_label = dict(zip(real["int_instrument"].astype(int), real["str_instrument"].str.strip()))
    num_classes = len(class_names_by_raw_label)

    negative = df[df["int_instrument"] < 0].drop_duplicates(subset="int_instrument")
    negative_names_by_raw_label = dict(zip(negative["int_instrument"].astype(int), negative["str_instrument"].str.strip()))

    return class_names_by_raw_label, num_classes, negative_names_by_raw_label

# Load the labels from the given video
def get_pitvis_labels(annot_index: pd.DataFrame, meta_pairs: list) -> np.ndarray:
    rows = []
    for vid, fi in meta_pairs:
        key = (vid, fi)
        if key in annot_index.index:
            row = annot_index.loc[key][["int_step", "int_instrument1", "int_instrument2"]].tolist()
        else:
            row = [-1, -1, -2]
        rows.append(row)
    return np.array(rows, dtype=np.int32)