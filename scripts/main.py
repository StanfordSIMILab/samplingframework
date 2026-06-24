# main.py - Main script for running diversity sampling pipeline end to end
# process raw/annotated data, optionally split into train/val/test, and create training set using diversity sampling or random sampling
# evaluate diversity training set on model training

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(os.path.abspath('..'))

import argparse
import yaml
import logging

from pathlib import Path

import numpy as np

from diversity_sampler import DiversitySampler
from auxiliary.data_manager import load_frames_and_masks, load_frames_from_dir
from auxiliary.data_partitioner import train_val_test_split, split_by_directory
import eval_data as eval

# Parse configuration yaml parameters
def load_config(config_path: str) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)

if __name__ == "__main__":
    """
    Full data_folder structure after running script:
    data_folder/
    ├── [raw input data]
    ├── sampling_pipeline_data.txt       # pipeline logger — all steps, timestamps
    ├── processed/
    │   ├── frames.npy
    │   ├── masks.npy                    # if annotated
    │   ├── labels.npy                   # if pitvis
    │   ├── color_map.json               # if annotated
    │   ├── frame_metadata.json          # save the mappings of indexes to video/frame
    │   └── frame_metadata.npy           # frame_metadata for downstream tasks
    ├── split_data/
    │   ├── color_map.json
    │   ├── train/
    │   │   ├── frames.npy
    │   │   ├── masks.npy
    │   │   ├── indices.npy
    │   │   ├── frame_metadata.json      # save the mappings of indexes to video/frame
    │   │   ├── frame_metadata.npy       # frame_metadata for downstream tasks
    │   │   ├── diversity/
    │   │   │   ├── frames.npy
    │   │   │   ├── masks.npy
    │   │   │   ├── frames/              # exported PNGs
    │   │   │   ├── masks/               # exported PNGs
    │   │   │   ├── all_embeddings.npy
    │   │   │   ├── diverse_indices.npy
    │   │   │   ├── index_to_frame.json  # save the mappings of indexes to video/frame
    │   │   │   └── diversity_metadata.json
    │   │   │   └── cluster_eval/
    │   │   │       ├── cluster_vis.png
    │   │   │       ├── cluster_iso_dist.png
    │   │   │       ├── cluster_tightness_graph.png
    │   │   │       ├── cluster_pairwise_ssim.png
    │   │   │       ├── clusters_inner_dist.png
    │   │   │       ├── centroid_pdist.png
    │   │   │       ├── ssim_cluster_rep.png
    │   │   │       ├── cluster_quality_score.png
    │   │   │       └── optimal_k.png
    │   │   └── random/
    │   │       ├── frames.npy
    │   │       ├── masks.npy
    │   │       ├── random_indices.npy
    │   │       └── index_to_frame.json  # save the mappings of indexes to video/frame
    │   ├── val/                         # if val_prop > 0
    │   │   ├── frames.npy
    │   │   ├── masks.npy
    │   │   ├── indices.npy
    │   │   ├── frame_metadata.json      # save the mappings of indexes to video/frame
    │   │   └── frame_metadata.npy       # frame_metadata for downstream tasks
    │   └── test/
    │       ├── frames.npy
    │       ├── masks.npy
    │       ├── indices.npy
    │       ├── frame_metadata.json      # save the mappings of indexes to video/frame
    │       └── frame_metadata.npy       # frame_metadata for downstream tasks
    └── eval_outputs/                    # if div_eval=true
        ├── evaluation_metrics.txt       # from eval logger
        ├── training_comparison/         # compare how well did the model learn
        │   ├── training_curves.png
        │   ├── confusion_matrices.png
        │   ├── per_class_f1.png
        │   └── balanced_accuracy.png
        └── data_coverage/               # compare how good was the sampling
            ├── hausdorff_coverage_redundancy.png
            ├── nn_coverage.png
            ├── pca_coverage_heatmap.png
            └── umap_selected.png
    """
    # parser to allow user to use separate custom configurations
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configuration.yaml", type=str,
                        help="Path to configuration YAML file")

    # Determine and load correct configuration files
    args = parser.parse_args()
    cfg = load_config(args.config)

    # Initialize parameters
    data_folder    = Path(cfg["data"]["data_dir"])
    processed_dir  = data_folder / "processed"
    split_data_dir = data_folder / "split_data"
    annotated      = cfg["data"]["annotated"]
    mask_type      = cfg["data"]["mask_type"]
    dataset_style  = cfg["data"]["dataset_style"]
    num_classes    = cfg["data"]["num_classes"]

    skip_split     = cfg["partitioning"]["skip_split"]
    split_by_dir   = cfg["partitioning"]["split_by_dir"]
    test_prop      = cfg["partitioning"]["test_prop"]
    val_prop       = cfg["partitioning"]["val_prop"]

    num_train_samples = = cfg["sampling"]["num_train_samples"]
    train_prop     = cfg["sampling"]["train_prop"]
    keep_interactive = cfg["sampling"]["keep_interactive"]
    use_filter = cfg["sampling"]["use_filter"]
    filter_thresh = cfg["sampling"]["filter_thresh"]

    div_eval         = cfg["evaluation"]["div_eval"]
    task_evaluation  = cfg["evaluation"]["task_evaluation"]
    model_name       = cfg["evaluation"]["model_name"]
    num_epochs       = cfg["evaluation"]["num_epochs"]
    batch_size       = cfg["evaluation"]["batch_size"]

    # Configure logger:
    log_path = data_folder / "sampling_pipeline_log.txt"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(message)s",
        handlers=[
            logging.FileHandler(log_path, mode="a"),
            logging.StreamHandler(),
        ]
    )
    logger = logging.getLogger("pipeline")

    logger.info("=" * 60)
    logger.info(f"Config: {args.config}")
    logger.info(f"Data folder: {data_folder}")
    logger.info(f"Dataset style: {dataset_style} | Annotated: {annotated}")
    logger.info(f"Task: {task_evaluation} | Model: {model_name}")
    logger.info("=" * 60)

    # Initialize Diversity Sampler
    sampler = DiversitySampler(
        emb_model="openclip",
        method="kmeans_elbow",
        keep_interactive = keep_interactive
        use_filter = use_filter
        filter_thresh = filter_thresh
    )

    # Load and process data
    if processed_dir.exists() and split_data_dir.exists():
        logger.info("Found existing processed/ and split_data/ — skipping straight to sampling...")
        skip_split = True
        frames = np.load(processed_dir / "frames.npy")
        masks  = np.load(processed_dir / "masks.npy") if (processed_dir / "masks.npy").exists() else None
        labels = np.load(processed_dir / "labels.npy", allow_pickle=True) if (processed_dir / "labels.npy").exists() else None
    elif processed_dir.exists():
        logger.info("Found existing processed/ — skipping data loading...")
        frames = np.load(processed_dir / "frames.npy")
        masks  = np.load(processed_dir / "masks.npy") if (processed_dir / "masks.npy").exists() else None
        labels = np.load(processed_dir / "labels.npy", allow_pickle=True) if (processed_dir / "labels.npy").exists() else None
    else:
        if annotated:
            logger.info("Loading annotated dataset...")
            result = load_frames_and_masks(
                data_folder=data_folder,
                mask_type=mask_type,
                dataset_style=dataset_style,
            )
            if len(result) == 4:
                frames, masks, color_map, labels = result
            else:
                frames, masks, color_map = result
                labels = None
        else:
            logger.info("Loading unannotated dataset...")
            frames = load_frames_from_dir(data_folder=data_folder)
            masks  = None
            labels = None

    len_total_frames = len(frames)
    logger.info(f"Loaded {len_total_frames} frames")

    # Split data
    if split_by_dir:
        logger.info("Splitting by directory name...")
        split_by_directory(data_folder=data_folder)
    elif not skip_split:
        logger.info("Performing random train/val/test split...")
        train_val_test_split(
            data_folder=data_folder,
            test_size=test_prop,
            val_size=val_prop,
        )

    # Load train split for sampling
    if split_by_dir and skip_split:
        logger.info("split_by_dir ignored because skip_split=true and existing splits found.")

    if skip_split:
        train_frames = frames
        train_masks  = masks
        train_frame_metadata = frame_metadata
    else:
        logger.info("Loading train split...")
        train_frames = np.load(split_data_dir / "train" / "frames.npy")
        train_masks_path = split_data_dir / "train" / "masks.npy"
        train_masks = np.load(train_masks_path) if train_masks_path.exists() else None
        train_metadata_path = split_data_dir / "train" / "frame_metadata.npy"
        train_frame_metadata = np.load(train_metadata_path, allow_pickle=True) if train_metadata_path.exists() else None

    if num_train_samples is not None:
        num_train_samples = num_train_samples
    elif train_prop is not None:
        num_train_samples = int(train_prop * len(train_frames))
    else:
        raise ValueError("Please indicate either a percent of training data or a total sample size under sampling configuration")

    # Diversity Sampling
    logger.info(f"Running diversity sampling — target n={num_train_samples}...")
    diversity_out = split_data_dir / "train"
    _, div_frames, div_masks, div_indices = sampler.sample(
        data_arr=train_frames,
        mask_arr=train_masks,
        num_samples=num_train_samples,
        run_eval=div_eval,
        save_data=True,
        data_dir=str(diversity_out),
        frame_metadata=train_frame_metadata,
    )
    logger.info(f"Diversity sampling complete — selected {len(div_indices)} frames")

    # Random sampling
    logger.info("Running random sampling...")
    random_indices = np.random.choice(len(train_frames), size=num_train_samples, replace=False)
    logger.info(f"Random sampling complete — selected {len(random_indices)} frames")
    random_out = split_data_dir / "train" / "random"
    random_out.mkdir(parents=True, exist_ok=True)
    np.save(random_out / "frames.npy", train_frames[random_indices])
    np.save(random_out / "random_indices.npy", random_indices)
    if train_masks is not None:
        np.save(random_out / "masks.npy", train_masks[random_indices])
    if train_frame_metadata is not None:
        np.save(random_out / "frame_metadata.npy", train_frame_metadata[random_indices])
        index_map = {int(i): str(train_frame_metadata[i]) for i in random_indices}
        with open(random_out / "index_to_frame.json", "w") as f:
            json.dump(index_map, f, indent=2)

    if div_eval:
        logger.info("Running model training evaluation...")

        train_videos = None
        val_video = None

        if dataset_style == "pitvis":
            train_videos = []
            logger.info("Enter video IDs to use for training (one or more per line, blank line to finish):")
            while True:
                raw = input().strip()
                if raw == "":
                    break
                for item in raw.split():
                    try:
                        train_videos.append(int(item))
                    except ValueError:
                        print(f"Skipping non-integer value: {item!r}")

            while True:
                raw = input("Enter video ID to use for validation: ").strip()
                try:
                    val_video = int(raw)
                    break
                except ValueError:
                    print("Please enter a valid integer.")

            logger.info(f"Training videos: {train_videos}")
            logger.info(f"Validation video: {val_video}")

        eval.main(
            task=task_evaluation,
            dataset_root=str(data_folder),
            dataset_style=dataset_style,
            train_videos=train_videos,
            val_video=val_video,
            output_dir=str(data_folder / "eval_outputs"),
            model_name=model_name,
            num_classes=num_classes,
            num_epochs=num_epochs,
            batch_size=batch_size,
            div_frames=div_frames,
            div_masks=div_masks,
            div_indices=np.array(div_indices),
            rand_frames=train_frames[random_indices],
            rand_masks=train_masks[random_indices] if train_masks is not None else None,
            rand_indices=random_indices,
            all_emb=np.load(str(diversity_out / "diversity" / "all_embeddings.npy")),
        )

    logger.info("Done!")