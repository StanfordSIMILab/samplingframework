# frame_extractor.py
# Process raw data (videos / frames / images) into a format suitable for diversity sampling
import os
import numpy as np
from PIL import Image
import cv2
from . import fvi_computation as fvi_utils

VIDEO_EXTENSIONS = {'.mp4', '.avi', '.mov', '.mkv'}

def extract_video_frames(
    video_path: str,
    output_dir: str,
    target_fps: float = 1.0,
) -> int:
    """
    Extract frames from a video at a configurable target fps, streaming
    frame-by-frame to disk. Saves as zero-padded original-frame-index JPEGs.
    Returns number of frames saved.
    """
    os.makedirs(output_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")

    native_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_interval = max(1, int(round(native_fps / target_fps)))

    print(
        f"Extracting {os.path.basename(video_path)} "
        f"(native_fps={native_fps:.2f}, target_fps={target_fps}, "
        f"interval={frame_interval}) -> {output_dir}"
    )

    frame_idx = 0
    saved = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % frame_interval == 0:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            Image.fromarray(frame_rgb).save(
                os.path.join(output_dir, f"{frame_idx:06d}.jpg")
            )
            saved += 1
        frame_idx += 1
        print(f"  {frame_idx}/{total_frames} (saved {saved})", end="\r", flush=True)

    cap.release()
    print(f"\n  Saved {saved} frames to {output_dir}")
    return saved


def extract_frames_with_filtering(
    video_path: str,
    output_dir: str,
    filtering: str = None,
    thresh: float = None,
    percentile: int = 20,
    use_elbow: bool = True,
    num_samples: int = None,
    keep_interactive: bool = False,
    log=None,
) -> tuple[int, np.ndarray, np.ndarray, float]:
    """
    Extract frames from a video to disk, optionally filtering with FVI.
    filtering=None (default): single pass, save all frames.
    filtering='fvi': two-pass — compute FVI scores, then save only kept frames.
    Returns (num_saved, kept_indices, fvi_scores, threshold).
    """
    os.makedirs(output_dir, exist_ok=True)
    log = log or print

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Extracting {os.path.basename(video_path)} ({total_frames} frames)...")

    if filtering is None:
        saved = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            Image.fromarray(frame_rgb).save(
                os.path.join(output_dir, f"{saved:06d}.jpg")
            )
            saved += 1
            print(f"  Saving frames: {saved}/{total_frames}", end="\r", flush=True)
        cap.release()
        print(f"\n  Saved {saved} frames to {output_dir}")
        return saved, np.arange(saved), np.array([]), None

    if filtering == "fvi":
        # Pass 1 — compute FVI scores without storing frames
        fvi_scores = []
        prev_frame = None
        frame_idx = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB).astype(np.float32)
            fvi_scores.append(
                np.linalg.norm(frame_rgb - prev_frame) if prev_frame is not None else 0.0
            )
            prev_frame = frame_rgb
            frame_idx += 1
            print(f"  Computing FVI: {frame_idx}/{total_frames}", end="\r", flush=True)

        cap.release()
        fvi_scores = np.array(fvi_scores)
        print()

        # Determine threshold interactively or automatically
        if keep_interactive:
            while True:
                raw = input(
                    "FVI filtering method — elbow, percentile, custom (or Enter to skip): "
                ).strip().lower()

                if raw == "":
                    thresh = None
                    kept_indices = np.arange(total_frames)
                    break

                if raw == "elbow":
                    thresh = fvi_utils.elbow_threshold(fvi_scores)
                    kept_indices = np.where(fvi_scores > thresh)[0]
                    if num_samples is not None and len(kept_indices) <= num_samples:
                        log(f"Elbow kept only {len(kept_indices)} frames (need {num_samples}) — try another method.")
                        continue
                    fvi_utils.show_fvi_histogram(scores=fvi_scores, thresh=thresh, save_path=output_dir)
                    log(f"FVI elbow: keeping {len(kept_indices)}/{total_frames} frames")
                    break

                elif raw == "percentile":
                    while True:
                        raw2 = input("Percentile (0-100): ").strip()
                        try:
                            pct = int(raw2)
                            if not 0 <= pct <= 100:
                                log("  Enter an integer between 0 and 100.")
                                continue
                            thresh = np.percentile(fvi_scores, pct)
                            kept_indices = np.where(fvi_scores > thresh)[0]
                            if num_samples is not None and len(kept_indices) <= num_samples:
                                log(f"Percentile={pct} kept only {len(kept_indices)} frames (need {num_samples}) — try lower.")
                                continue
                            fvi_utils.show_fvi_histogram(scores=fvi_scores, thresh=thresh, save_path=output_dir)
                            log(f"FVI percentile={pct}: keeping {len(kept_indices)}/{total_frames} frames")
                            break
                        except ValueError:
                            log("  Enter a valid integer.")
                    break

                elif raw == "custom":
                    while True:
                        raw2 = input("Threshold (float): ").strip()
                        try:
                            thresh = float(raw2)
                            kept_indices = np.where(fvi_scores > thresh)[0]
                            if num_samples is not None and len(kept_indices) <= num_samples:
                                log(f"Threshold={thresh} kept only {len(kept_indices)} frames (need {num_samples}) — try lower.")
                                continue
                            fvi_utils.show_fvi_histogram(scores=fvi_scores, thresh=thresh, save_path=output_dir)
                            log(f"FVI custom threshold={thresh}: keeping {len(kept_indices)}/{total_frames} frames")
                            break
                        except ValueError:
                            log("  Enter a valid float.")
                    break

                else:
                    log("  Enter 'elbow', 'percentile', or 'custom'.")

        else:
            # automatic threshold selection
            if use_elbow and thresh is None:
                thresh = fvi_utils.elbow_threshold(fvi_scores)
                kept_indices = np.where(fvi_scores > thresh)[0]
                if num_samples is not None and len(kept_indices) < total_frames * 0.1:
                    log(f"  Elbow over-filtered ({len(kept_indices)} frames), falling back to percentile={percentile}")
                    thresh = np.percentile(fvi_scores, percentile)
            elif thresh is None:
                if num_samples is not None:
                    buffer = 1.5
                    keep_pct = (num_samples * buffer / total_frames) * 100
                    percentile = int(100 - keep_pct)
                    percentile = max(0, min(percentile, 95))
                thresh = np.percentile(fvi_scores, percentile)

            kept_indices = np.where(fvi_scores > thresh)[0]
            fvi_utils.show_fvi_histogram(scores=fvi_scores, thresh=thresh, save_path=output_dir)
            log(f"  FVI threshold={thresh:.2f}, keeping {len(kept_indices)}/{total_frames} frames")

            if num_samples is not None and len(kept_indices) < num_samples:
                raise ValueError(
                    f"FVI filtering removed too many frames — only {len(kept_indices)} remain "
                    f"but {num_samples} are needed. Disable fvi_filtering or reduce num_samples."
                )
    else:
        raise ValueError(
            f"Unknown filtering method '{filtering}'. "
            f"Choose 'fvi' or None. "
            f"Additional methods can be added to extract_frames_with_filtering()."
        )

    kept_set = set(kept_indices.tolist())

    # Pass 2 — save only kept frames to disk
    cap = cv2.VideoCapture(video_path)
    frame_idx = 0
    saved = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx in kept_set:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            Image.fromarray(frame_rgb).save(
                os.path.join(output_dir, f"{frame_idx:06d}.jpg")
            )
            saved += 1
        frame_idx += 1
        print(f"  Saving frames: {saved}/{len(kept_indices)}", end="\r", flush=True)

    cap.release()
    print(f"\n  Saved {saved} frames to {output_dir}")
    return saved, kept_indices, fvi_scores, thresh


def process_video(
    video_path: str,
    output_dir: str = None,
    filtering: str = None,
    thresh: float = None,
    percentile: int = 20,
    use_elbow: bool = True,
    keep_interactive: bool = False,
    num_samples: int = None,
    log=None,
) -> tuple[int, np.ndarray, np.ndarray, float]:
    """
    Process a single video — extract frames with optional FVI filtering to disk.
    output_dir defaults to a folder named after the video stem next to the video.
    """
    stem = os.path.splitext(os.path.basename(video_path))[0]

    if output_dir is None:
        output_dir = os.path.join(os.path.dirname(os.path.abspath(video_path)), stem)

    saved, kept_indices, fvi_scores, threshold = extract_frames_with_filtering(
        video_path=video_path,
        output_dir=output_dir,
        filtering=filtering,
        thresh=thresh,
        percentile=percentile,
        use_elbow=use_elbow,
        num_samples=num_samples,
        keep_interactive=keep_interactive,
        log=log,
    )

    return saved, kept_indices, fvi_scores, threshold


def process_videos(
    input_dir: str,
    output_dir: str = None,
    fvi_filtering: bool = False,
    thresh: float = None,
    percentile: int = 20,
    use_elbow: bool = True,
    show_hist: bool = True,
    keep_interactive: bool = False,
    num_samples: int = None,
    log=None,
) -> list[str]:
    """
    Process all video files in input_dir, extracting frames into subfolders
    named after each video stem.
    Returns list of output subdirectory paths.
    """
    input_dir = os.path.abspath(input_dir)
    if output_dir is not None:
        output_dir = os.path.abspath(output_dir)
        os.makedirs(output_dir, exist_ok=True)
    else:
        output_dir = input_dir

    print(f"Processing videos in {input_dir}...")
    output_dirs = []

    for root, dirs, files in os.walk(input_dir):
        for file in sorted(files):
            if not any(file.lower().endswith(ext) for ext in VIDEO_EXTENSIONS):
                continue

            video_path = os.path.join(root, file)
            stem = os.path.splitext(file)[0]
            relative_path = os.path.relpath(root, input_dir)
            video_output_dir = os.path.join(output_dir, relative_path, stem)

            print(f"\nProcessing {file} -> {video_output_dir}")

            process_video(
                video_path=video_path,
                output_dir=video_output_dir,
                filtering=filtering,
                thresh=thresh,
                percentile=percentile,
                use_elbow=use_elbow,
                show_hist=show_hist,
                keep_interactive=keep_interactive,
                num_samples=num_samples,
                log=log,
            )
            output_dirs.append(video_output_dir)

    if not output_dirs:
        raise ValueError(f"No video files found in {input_dir}")

    return output_dirs