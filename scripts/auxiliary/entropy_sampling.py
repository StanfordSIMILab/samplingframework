import numpy as np

def compute_entropy(probabilities):
    """
    Compute entropy for a set of probabilities.
    Args:
        probabilities (numpy array): Probability distribution.
    Returns:
        float: Entropy score.
    """
    return -np.sum(probabilities * np.log2(probabilities + 1e-10))

def entropy_sampling(frames, model):
    """
    Sample frames based on entropy scores.
    Args:
        frames (list): List of frames.
        model: Pretrained model to compute predictions.
    Returns:
        list: Frames with the highest entropy scores.
    """
    entropy_scores = []
    for frame in frames:
        predictions = model.predict(frame)
        entropy_scores.append(compute_entropy(predictions))
    return np.argsort(entropy_scores)[-10:]

