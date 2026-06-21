import numpy as np
import pandas as pd
from pathlib import Path

# PitVis-specific functions for loading annotations and extracting labels based on video/frame metadata
def load_pitvis_annotations(data_dir: Path, video_ids: list) -> pd.DataFrame:
    dfs = [pd.read_csv(data_dir / f"annotations_{v:02d}.csv") for v in video_ids]
    return pd.concat(dfs, ignore_index=True).set_index(["int_video", "int_time"])


def get_pitvis_labels(annot_index: pd.DataFrame, meta_pairs: list) -> np.ndarray:
    rows = []
    for vid, fi in meta_pairs:
        key = (vid, fi)
        if key in annot_index.index:
            row = annot_index.loc[key][
                ["int_step", "int_instrument1", "int_instrument2"]
            ].tolist()
        else:
            row = [-1, -1, -2]
        rows.append(row)
    return np.array(rows, dtype=np.int32)