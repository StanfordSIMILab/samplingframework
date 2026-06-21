# fvi_computation.py: Frame Variation Index (FVI) computation and filtering utilities for pre-processing frames from mp4 videos.
import numpy as np
import pandas as pds
import seaborn as sns

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


def fvi_filter(frames: np.ndarray, thresh: float = None, use_elbow=True):
    
    scores = compute_fvi(frames)
    if len(scores) == 0:
        return frames, np.arange(len(frames)), scores, thresh

    if use_elbow and thresh is None:
        thresh = elbow_threshold(scores)
    elif thresh is None:
        # Filter out the 10% of frames with the lowest FVI scores by default
        thresh = np.percentile(scores, 90)

    kept = [0]
    kept.extend(
        j + 1
        for j, s in enumerate(scores)
        if s > thresh
    )

    return frames[kept], np.asarray(kept), scores, thresh

def show_fvi_histogram(scores: np.ndarray, video_id: int = None, save_path: str = None) -> None:

    if len(scores) == 0:
        return

    df = pd.DataFrame(scores, columns=["fvi"])

    # Histogram
    sns.histplot(df, x="fvi", bins=20)

    # Fit normal distribution
    mu, std = norm.fit(df["fvi"])

    # Bell curve
    min_fvi = df["fvi"].min()
    max_fvi = df["fvi"].max()

    x = np.linspace(min_fvi, max_fvi, 100)
    y = norm.pdf(x, mu, std) * len(df) * (x[1] - x[0])

    plt.plot(
        x,
        y,
        "r--",
        linewidth=2,
        label="Normal fit"
    )

    # Mean and SD lines
    plt.axvline(mu, color="blue", linestyle="--", label="Mean")

    plt.axvline(
        mu + std,
        color="green",
        linestyle="--",
        label="Mean + 1 SD"
    )
    plt.axvline(
        mu - std,
        color="green",
        linestyle="--",
        label="Mean - 1 SD"
    )

    plt.axvline(
        mu + 2 * std,
        color="orange",
        linestyle="--",
        label="Mean + 2 SD"
    )
    plt.axvline(
        mu - 2 * std,
        color="orange",
        linestyle="--",
        label="Mean - 2 SD"
    )

    plt.legend()

    if video_id is not None:
        plt.title(f"FVI distribution — video {video_id:02d}")
    else:
        plt.title("FVI distribution")

    plt.xlabel("FVI")
    plt.ylabel("Count")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
    else:
        plt.show()

    plt.close()
