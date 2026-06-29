# fvi_computation.py: Frame Variation Index (FVI) computation and filtering utilities for pre-processing frames from mp4 videos.
import os

import numpy as np
import pandas as pd
import seaborn as sns

import matplotlib.pyplot as plt
from scipy.stats import norm

# Thresholding utilities for FVI filtering:
def elbow_threshold(values):
    # Use Elbow Method to determine  for FVI:

    y = np.sort(np.asarray(values))
    x = np.arange(len(y))

    if len(y) < 2:
        return y[0] if len(y) else 0

    x = (x - x.min()) / (x.max() - x.min())
    y = (y - y.min()) / (y.max() - y.min() + 1e-8)

    line = np.array([1, 1])
    points = np.column_stack((x, y))

    distances = np.abs(
        line[0] * points[:, 1] - line[1] * points[:, 0]
    ) / np.linalg.norm(line)

    idx = np.argmax(distances)

    return np.sort(values)[idx]

# Frame Variation Index (FVI) utilities for pre-filtering frames from mp4 videos
def compute_fvi(frames) -> np.ndarray:
    """
    Compute Frame Variation Index (FVI) for a given set of frames.
    Args:
        frames (list): List of frames represented as numpy arrays.
    Returns:
        np.ndarray: FVI scores for each frame.
    """

    if len(frames) <= 1:
        return np.array([])

    fvi_scores = []

    for i in range(1, len(frames)):
        variation = np.linalg.norm(frames[i].astype(np.float32) - frames[i - 1].astype(np.float32))
        fvi_scores.append(variation)
    
    return np.array(fvi_scores)


def fvi_filter(frames: np.ndarray, thresh: float = None, percentile: int = 20, use_elbow=True):
    scores = compute_fvi(frames)
    if len(scores) == 0:
        return frames, np.arange(len(frames)), scores, thresh

    if use_elbow and thresh is None:
        thresh = elbow_threshold(scores)
    elif thresh is None:
        # Filter out the percentile% of frames with the lowest FVI scores by default
        thresh = np.percentile(scores, percentile)

    kept = [0]
    kept.extend(
        j + 1
        for j, s in enumerate(scores)
        if s > thresh
    )

    return frames[kept], np.asarray(kept), scores, thresh

def show_fvi_histogram(scores: np.ndarray, thresh: float = None, save_path: str = None) -> None:
    if len(scores) == 0:
        return

    df = pd.DataFrame(scores, columns=["fvi"])

    sns.histplot(df, x="fvi", bins=20)

    if thresh is not None:
        pct_kept = np.mean(scores > thresh) * 100
        plt.axvline(thresh, color="red", linestyle="--", linewidth=2,
                    label=f"Threshold = {thresh:.1f} ({pct_kept:.1f}% kept)")

    p25 = np.percentile(scores, 25)
    p50 = np.percentile(scores, 50)
    p75 = np.percentile(scores, 75)

    plt.axvline(p25, color="green", linestyle="--", label=f"25th percentile = {p25:.1f}")
    plt.axvline(p50, color="blue", linestyle="--", label=f"Median = {p50:.1f}")
    plt.axvline(p75, color="orange", linestyle="--", label=f"75th percentile = {p75:.1f}")

    plt.legend()
    plt.title("FVI distribution")
    plt.xlabel("FVI")
    plt.ylabel("Count")
    plt.tight_layout()

    if save_path:
        os.makedirs(save_path, exist_ok=True)
        plt.savefig(os.path.join(save_path, "fvi_histogram.png"), bbox_inches="tight")
    else:
        plt.show()
    plt.close()
