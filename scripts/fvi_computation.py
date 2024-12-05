import numpy as np

def compute_fvi(frames):
    """
    Compute Frame Variation Index (FVI) for a given set of frames.
    Args:
        frames (list): List of frames represented as numpy arrays.
    Returns:
        list: FVI scores for each frame.
    """
    fvi_scores = []
    for i in range(1, len(frames)):
        variation = np.linalg.norm(frames[i] - frames[i-1])
        fvi_scores.append(variation)
    return fvi_scores

