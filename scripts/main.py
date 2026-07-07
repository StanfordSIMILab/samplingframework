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
import re

import numpy as np
import json
import pandas as pd

from diversity_sampler import DiversitySampler
import data_loader as dm
from data_partitioner import train_val_test_split, split_by_directory, split_by_video_ids
import eval_data as ev

# Parse configuration yaml parameters
def load_config(config_path: str) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)

# Check with directory has valid data to pass to evaluation function
def has_valid_data(path: Path, style: str) -> bool:
    if not path.is_dir():
        return False
    return (path / "frames.npy").exists()

if __name__ == "__main__":
    """
    Full data_folder structure after running script:
    data_folder/
    ├── [raw input data]
    └── sampling_outputs/
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
        └── eval_outputs/                    # if evaluate_data=true
            ├── evaluation_metrics.txt       # from eval logger
            ├── training_comparison/         # compare how well did the model learn
            │   ├── training_curves.png
            │   ├── confusion_matrices.png
            │   ├── per_class_f1.png
            │   └── balanced_accuracy.png
            └── data_coverage/               # compare how good was the sampling (div vs. random)
                ├── hausdorff_coverage_redundancy.png
                ├── nn_coverage.png
                ├── pca_coverage_heatmap.png
                └── umap_selected.png
    
    * If pipeline stopped halfway, script automatically resumes at proper step
    """
    # parser to allow user to use separate custom configurations
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="./configs/configuration.yaml", type=str,
                        help="Path to configuration YAML file")

    # Determine and load correct configuration files
    args = parser.parse_args()
    cfg = load_config(args.config)

    # Initialize directory from cfg
    data_folder    = Path(cfg["data"]["data_dir"])
    output_folder  = data_folder / "sampling_outputs"
    processed_dir  = output_folder / "processed"
    split_data_dir = output_folder / "split_data"
    output_folder.mkdir(parents=True, exist_ok=True)

    # Data loading configurations
    annotated     = cfg["data"]["annotated"]
    filtering     = cfg["data"]["filtering"]
    dataset_style = cfg["data"]["dataset_style"]
    videos_fps    = cfg["data"].get("videos_fps", 1.0) # if raw data are mp4s

    # video/pitvis specific parameters
    if dataset_style in {"pitvis", "video"}:
        choose_videos = cfg["video"].get("choose_videos", False)
        train_videos  = cfg["video"].get("train_videos", []) or []
        val_videos    = cfg["video"].get("val_videos", []) or []
        test_videos   = cfg["video"].get("test_videos", []) or []
    else:
        choose_videos = False
        train_videos  = None
        val_videos    = None
        test_videos   = None

    # annotated data
    mask_type       = cfg["annotation"]["mask_type"]
    num_classes     = cfg["annotation"]["num_classes"]
    task_evaluation = cfg["annotation"]["task_evaluation"]
    labels_path     = cfg["annotation"].get("labels_path", None)

    # Partioning
    skip_split   = cfg["partitioning"]["skip_split"]
    split_by_dir = cfg["partitioning"]["split_by_dir"]
    test_prop    = cfg["partitioning"]["test_prop"]
    val_prop     = cfg["partitioning"]["val_prop"]

    # Sampling
    num_train_samples = cfg["diversity_sampling"]["num_train_samples"]
    train_prop        = cfg["diversity_sampling"]["train_prop"]
    emb_model         = cfg["diversity_sampling"]["emb_model"]
    method            = cfg["diversity_sampling"]["method"]
    reduce_dims       = cfg["diversity_sampling"]["reduce_dims"]
    n_components      = cfg["diversity_sampling"]["n_components"]
    spread_sampling   = cfg["diversity_sampling"].get("spread_sampling", False)
    keep_interactive  = cfg["diversity_sampling"]["keep_interactive"]

    # Evaluation random vs. diversity
    evaluate_data  = cfg["evaluation"]["evaluate_data"]
    use_pretrained = cfg["evaluation"]["use_pretrained"]
    model_name     = cfg["evaluation"]["model_name"]
    num_epochs     = cfg["evaluation"]["num_epochs"]
    batch_size     = cfg["evaluation"]["batch_size"]
    run_efficiency_curve = cfg["evaluation"].get("efficiency_curve", False)
    sample_sizes   = cfg["evaluation"].get("sample_sizes", [100, 200, 300, 500, 750, 1000])

    # Configure logger:
    log_path = output_folder / "sampling_pipeline_log.txt"
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
        emb_model=emb_model,
        method=method,
        keep_interactive = keep_interactive,
        spread_sampling = spread_sampling,
    )

    # Load and process data
    if processed_dir.exists():
        logger.info("Found existing processed data — loading from cache...")
        frames = np.load(processed_dir / "frames.npy")
        masks = np.load(processed_dir / "masks.npy") if (processed_dir / "masks.npy").exists() else None
        labels = np.load(processed_dir / "labels.npy", allow_pickle=True) if (processed_dir / "labels.npy").exists() else None
        frame_metadata = np.load(processed_dir / "frame_metadata.npy", allow_pickle=True) if (processed_dir / "frame_metadata.npy").exists() else None
        
        color_map_json = processed_dir / "color_map.json"
        color_map = {}
        if color_map_json.exists():
            with open(color_map_json) as f:
                color_map = {int(k): v for k, v in json.load(f).items()}
                
        if split_data_dir.exists():
            skip_split = True
            logger.info("Found existing split_data/ — skipping split...")
    else:
        color_map = {}

        if task_evaluation == "phase_classification":
            logger.info("Loading frames for phase classification...")
            if annotated:
                if dataset_style == "pitvis":
                    frames, masks, color_map, labels, frame_metadata = dm.load_frames_and_masks(
                        data_folder=data_folder,
                        dataset_style="pitvis",
                        output_folder=output_folder
                    )
                elif dataset_style == "video" and labels_path is not None:
                    frames, frame_metadata = dm.load_frames_from_dir(
                        data_folder=data_folder,
                        dataset_style=dataset_style,
                        videos=[str(v) for v in train_videos] if train_videos else None,
                        videos_fps=videos_fps,
                        filtering=filtering,
                        output_folder=output_folder,
                    )
                    labels = None
                    masks = None
                else:
                    raise ValueError(
                        "Phase classification requires temporal labels — use 'pitvis' dataset style, "
                        "or 'video' style with a labels_path CSV providing per-frame phase labels."
                    )
            else:
                frames, frame_metadata = dm.load_frames_from_dir(
                    data_folder=data_folder,
                    dataset_style=dataset_style,
                    videos_fps=videos_fps,
                    filtering=filtering,
                    output_folder=output_folder
                )
                labels = None
                masks = None

        elif task_evaluation == "segmentation":
            logger.info("Loading frames and masks for segmentation...")
            if annotated:
                # works for video / non-video datasets
                result = dm.load_frames_and_masks(
                    data_folder=data_folder,
                    output_folder=output_folder,
                    mask_type=mask_type,
                    dataset_style=dataset_style,
                    videos_fps=videos_fps,
                    filtering=filtering,
                )

                if len(result) == 5:
                    frames, masks, color_map, labels, frame_metadata = result
                else:
                    frames, masks, color_map, frame_metadata = result
                    labels = None
            else:
                frames, frame_metadata = dm.load_frames_from_dir(
                    data_folder=data_folder,
                    dataset_style=dataset_style,
                    videos_fps=videos_fps,
                    output_folder=output_folder,
                    filtering=filtering,
                )
                masks = None
                labels = None
            
        else:
            raise ValueError(
                f"Unknown task_evaluation '{task_evaluation}'. "
                f"Choose 'phase_classification' or 'segmentation'."
            )

    if labels_path is not None and labels is None:
        labels_df = pd.read_csv(labels_path)
        labels = labels_df["phase"].values.astype(np.int64)

    len_total_frames = len(frames)
    logger.info(f"Loaded {len_total_frames} frames")

    # Split data
    if skip_split:
        logger.info("Skipping split...")
    elif split_by_dir:
        # Split data by directory / filenames (contains val/test/train in name)
        logger.info("Splitting data by directory...")
        split_by_directory(data_folder=output_folder)
    elif dataset_style in {"pitvis", "video"}:
        if choose_videos or not train_videos:
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

            test_videos = []
            logger.info("Enter video IDs to use for test (one or more per line, blank line to finish):")
            while True:
                raw = input().strip()
                if raw == "":
                    break
                for item in raw.split():
                    try:
                        test_videos.append(int(item))
                    except ValueError:
                        print(f"Skipping non-integer value: {item!r}")

            all_videos = sorted({
                int(m.group(1))
                for d in os.listdir(data_folder)
                if os.path.isdir(os.path.join(data_folder, d))
                for m in [re.search(r'(\d+)', d)]
                if m
            })
            val_videos = [v for v in all_videos if v not in train_videos and v not in test_videos]

            logger.info(f"Training videos: {train_videos}")
            logger.info(f"Test videos: {test_videos}")
            logger.info(f"Validation videos: {val_videos}")

        split_by_video_ids(
            output_folder=output_folder,
            train_videos=train_videos,
            val_videos=val_videos,
            test_videos=test_videos,
            frames=frames,
            masks=masks,
            labels=labels,
            frame_metadata=frame_metadata,
        )

    else:
        # Split data into train/val/test using default random train_val_test_split function
        logger.info("Splitting data into train/val/test...")
        train_val_test_split(
            data_folder=output_folder,
            test_size=test_prop,
            val_size=val_prop,
        )

    # Load train split for sampling
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

    # Determine number/proportion of training samples to select for diversity sampling
    if num_train_samples is None and train_prop is None:
        raise ValueError("Please indicate either a percent of training data or a total sample size under sampling configuration")
    elif num_train_samples is None and train_prop is not None:
        if train_prop <= 0 or train_prop > 1.0:
            raise ValueError("Please provide a proper train_prop (0.0, 1.0]")
        num_train_samples = int(train_prop * len(train_frames))
        if num_train_samples <= 0:
            raise ValueError(f"train_prop={train_prop} results in zero training samples for {len(train_frames)} frames")

    # Check for existing diversity sampling outputs
    diversity_out = split_data_dir / "train"
    existing_diversity = diversity_out / "diversity"

    # find the most recent diversity dir
    if not existing_diversity.exists():
        for i in range(1, 20):
            candidate = diversity_out / f"diversity_{i}"
            if not candidate.exists():
                break
            existing_diversity = candidate

    div_out_path = None
    div_frames = None
    div_masks = None
    div_indices = None

    # Run Diversity Sampling if no previous examples exist
    if existing_diversity.exists() and (existing_diversity / "frames.npy").exists():
        logger.info(f"Found existing diversity sampling outputs at {existing_diversity} — skipping sampling...")
        div_frames  = np.load(existing_diversity / "frames.npy")
        div_masks   = np.load(existing_diversity / "masks.npy") if (existing_diversity / "masks.npy").exists() else None
        div_indices = np.load(existing_diversity / "diverse_indices.npy")
        div_out_path = str(existing_diversity)
        logger.info(f"Loaded {len(div_indices)} diverse frames from existing outputs")
    else:
        logger.info(f"Running diversity sampling — target n={num_train_samples}...")
        _, div_frames, div_masks, div_indices, div_out_path = sampler.sample(
            data_arr=train_frames,
            mask_arr=train_masks,
            num_samples=num_train_samples,
            reduce_dims=reduce_dims,
            n_components=n_components,
            run_eval=evaluate_data,
            save_data=True,
            data_dir=str(diversity_out),
            frame_metadata=train_frame_metadata,
        )
        logger.info(f"Diversity sampling complete — selected {len(div_indices)} frames")

    # Check for existing random sampling outputs
    random_out = split_data_dir / "train" / "random"

    if random_out.exists() and (random_out / "random_indices.npy").exists():
        logger.info("Found existing random sampling outputs — skipping random sampling...")
        random_indices = np.load(random_out / "random_indices.npy")
        logger.info(f"Loaded {len(random_indices)} random indices from existing outputs")
    else:
        # Randomly sample the same number of frames as diversity sampling for comparison
        if num_train_samples > len(train_frames):
            raise ValueError(f"Requested {num_train_samples} random samples, but only {len(train_frames)} frames available.")

        # Random Sample
        logger.info("Running random sampling...")
        random_indices = np.random.choice(len(train_frames), size=num_train_samples, replace=False)
        logger.info(f"Random sampling complete — selected {len(random_indices)} frames")
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

    if evaluate_data:
        logger.info("Running model training evaluation...")

        # load val/test from split npy files
        val_root_path  = split_data_dir / "val"
        test_root_path = split_data_dir / "test"

        val_frames_for_eval = np.load(val_root_path  / "frames.npy") if has_valid_data(val_root_path, dataset_style) else None
        val_masks_for_eval = np.load(val_root_path  / "masks.npy")  if val_frames_for_eval  is not None and (val_root_path  / "masks.npy").exists() else None
        test_frames_for_eval = np.load(test_root_path / "frames.npy") if has_valid_data(test_root_path, dataset_style) else None
        test_masks_for_eval = np.load(test_root_path / "masks.npy")  if test_frames_for_eval is not None and (test_root_path / "masks.npy").exists() else None

        # derive labels for val/test
        val_labels_for_eval  = None
        test_labels_for_eval = None
        if task_evaluation == "segmentation":
            val_labels_for_eval  = val_masks_for_eval.reshape(len(val_masks_for_eval),   -1)[:, 0].astype(np.int64) if val_masks_for_eval  is not None else None
            test_labels_for_eval = test_masks_for_eval.reshape(len(test_masks_for_eval), -1)[:, 0].astype(np.int64) if test_masks_for_eval is not None else None
        elif task_evaluation == "phase_classification":
            val_labels_for_eval  = np.load(val_root_path  / "labels.npy")[:, 0] if val_frames_for_eval  is not None and (val_root_path  / "labels.npy").exists() else None
            test_labels_for_eval = np.load(test_root_path / "labels.npy")[:, 0] if test_frames_for_eval is not None and (test_root_path / "labels.npy").exists() else None

        # embeddings, color_map, class_names, num_classes
        all_emb_path = os.path.join(div_out_path, "all_embeddings.npy") if div_out_path is not None else None

        color_map_path = split_data_dir / "color_map.json"
        color_map_for_eval = None
        if color_map_path.exists():
            with open(color_map_path) as f:
                color_map_for_eval = {int(k): v for k, v in json.load(f).items()}

        # training labels sliced to train split, making sure indices work between diverse/random
        if labels is not None:
            if skip_split:
                train_labels_for_eval = labels[:, 0] if labels.ndim == 2 else labels
            else:
                train_indices_path = split_data_dir / "train" / "indices.npy"
                train_indices = np.load(train_indices_path) if train_indices_path.exists() else None
                if train_indices is not None:
                    train_labels_for_eval = labels[train_indices, 0] if labels.ndim == 2 else labels[train_indices]
                else:
                    train_labels_for_eval = labels[:, 0] if labels.ndim == 2 else labels

                if len(train_labels_for_eval) != len(train_frames):
                    logger.warning(
                        f"train_labels_for_eval size {len(train_labels_for_eval)} != "
                        f"train_frames size {len(train_frames)} — truncating to match"
                    )
                    train_labels_for_eval = train_labels_for_eval[:len(train_frames)]
        else:
            train_labels_for_eval = None

        # Make sure that num_classes is determined either from color_map, train_labels_for_eval, or passed explicitly
        if num_classes is None and color_map_for_eval is not None:
            num_classes = len(color_map_for_eval)

        # If num_classes is still None, try to infer from train_labels_for_eval (exclude -1 label)
        if num_classes is None and train_labels_for_eval is not None:
            valid_labels = train_labels_for_eval[train_labels_for_eval >= 0]
            num_classes = len(np.unique(valid_labels))
            logger.info(f"Inferred num_classes={num_classes} from {len(np.unique(valid_labels))} unique labels")
 
        if num_classes is None:
            raise ValueError(
                "num_classes could not be inferred — set it explicitly in the config under annotation.num_classes"
            )

        # Handle negative label name for pitvis phase classification if available
        negative_label_name = None
        if dataset_style == "pitvis" and task_evaluation == "phase_classification":
            phase_classes_path = split_data_dir / "phase_classes.json"
            if phase_classes_path.exists():
                with open(phase_classes_path) as f:
                    phase_meta = json.load(f)
                num_classes = phase_meta["num_classes"]
                class_names_by_raw_label = {int(k): v for k, v in phase_meta["class_names_by_raw_label"].items()}
                class_names_for_eval = [class_names_by_raw_label[k] for k in sorted(class_names_by_raw_label)]
                negative_label_name = phase_meta["unlabeled_name"]
            else:
                logger.warning(f"{phase_classes_path} not found — using generic class names.")
                class_names_for_eval = [str(i) for i in range(num_classes)]
        else:
            class_names_for_eval = [str(i) for i in range(num_classes)]

        ev.main(
            task=task_evaluation,
            dataset_style=dataset_style,
            output_dir=str(output_folder / "eval_outputs"),
            sampling_dir=str(split_data_dir / "train"),
            use_pretrained=use_pretrained,
            model_name=model_name if task_evaluation == "segmentation" else "PhaseClassifier",
            num_classes=num_classes,
            num_epochs=num_epochs,
            batch_size=batch_size,
            all_emb=np.load(all_emb_path) if all_emb_path is not None and os.path.exists(all_emb_path) else None,
            div_frames=div_frames,
            div_masks=div_masks,
            div_indices=np.array(div_indices),
            rand_frames=train_frames[random_indices],
            rand_masks=train_masks[random_indices] if train_masks is not None else None,
            rand_indices=random_indices,
            val_frames=val_frames_for_eval,
            val_masks=val_masks_for_eval,
            test_frames=test_frames_for_eval,
            test_masks=test_masks_for_eval,
            color_map=color_map_for_eval,
            train_labels=train_labels_for_eval,
            val_labels=val_labels_for_eval,
            test_labels=test_labels_for_eval,
            class_names=class_names_for_eval,
            negative_label_name=negative_label_name,
        )

        if run_efficiency_curve:
            logger.info("Running data efficiency curve...")
            ev.run_data_efficiency_curve(
                task=task_evaluation,
                output_dir=str(output_folder / "eval_outputs"),
                model_name=model_name if task_evaluation == "segmentation" else "LightweightPhaseClassifier",
                num_classes=num_classes,
                num_epochs=num_epochs,
                batch_size=batch_size,
                all_train_emb=np.load(all_emb_path) if all_emb_path is not None and os.path.exists(all_emb_path) else None,
                all_train_frames=train_frames,
                all_train_masks=train_masks,
                all_train_labels=train_labels_for_eval,
                test_frames=test_frames_for_eval,
                test_masks=test_masks_for_eval,
                test_labels=test_labels_for_eval,
                color_map=color_map_for_eval,
                class_names=class_names_for_eval,
                sample_sizes=sample_sizes,
                negative_label_name=negative_label_name,
                use_pretrained=use_pretrained,
                emb_model=emb_model,
                method=method,
                reduce_dims=reduce_dims,
                n_components=n_components,
                spread_sampling=spread_sampling,
            )

    logger.info("Done!")