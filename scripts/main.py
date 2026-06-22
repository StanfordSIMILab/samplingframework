# main.py - Main script for running diversity sampling pipeline end to end
# process raw/annotated data, optionally split into train/val/test, and create training set using diversity sampling or random sampling
# evaluate diversity training set on model training

import sys
import os
import json

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(os.path.abspath('..'))

from pathlib import Path

import numpy as np
import argparse

from diversity_sampler import DiversitySampler
from auxiliary.data_manager import load_frames_and_masks, load_frames_from_dir
from auxiliary.data_partitioner import train_val_test_split, split_by_directory, infer_split_from_path
import eval_data as eval

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Split dataset into train, val, and test sets, and create the training set using diversity sampling or random sampling"
    )

    # Data processing parameters
    parser.add_argument(
        "--data-dir", required=True, type=str,
        help="Root data folder directory containing raw images/videos and masks"
    )
    parser.add_argument(
        "--annotated", action="store_true",
        help="Whether the dataset has paired mask annotations"
    )
    parser.add_argument(
        "--mask-type", default="color_mask", type=str,
        help="Mask type: 'color_mask', 'mask', or 'watershed_mask'"
    )
    parser.add_argment(
        "--num_classes", default=None, type=int,
        help="number of classes for evaluating model"
    )

    # Data Partioning (train/val/test) parameters
    parser.add_argument(
        "--skip-split", action="store_true",
        help="Skip train/val/test splitting and sample directly from all processed frames"
    )
    parser.add_argument(
        "--split-by-dir", action="store_true",
        help="Infer train/val/test splits from directory names rather than random splitting"
    )
    parser.add_argument(
        "--train-prop", default=0.1, type=float,
        help="Proportion of training frames to select via sampling"
    )
    parser.add_argument(
        "--test-prop", default=0.1, type=float,
        help="Proportion of the dataset to use as the test set"
    )
    parser.add_argument(
        "--val-prop", default=0.1, type=float,
        help="Proportion of the dataset to use as the validation set (0 to skip)"
    )

    # Diversity Sampling parameters
    parser.add_argument(
        "--div-eval", action="store_true",
        help="Run model training evaluation comparing diversity vs random sampling"
    )
    parser.add_argument(
        "--filter-clusters-manually", action="store_true",
        help="Manually filter clusters from embedding/cluster results"
    )
    # Dataset evaluation and structure parameters
    parser.add_argument(
        "--task-evaluation", default="segmentation", type=str,
        help="Task for model evaluation: 'segmentation' or 'phase_classification'"
    )
    parser.add_argument(
        "--dataset-style", default="cholec", type=str,
        help="Dataset folder structure: 'cholec', 'flat', 'nested', or 'pitvis'"
    )
    parser.add_argument(
        "--model-name", default=None, type=str,
        help="Model to use for eval training: 'deeplab', 'unet', 'mask2former', 'segformer', 'upernet'"
    )

    args = parser.parse_args()

    data_folder = Path(args.data_dir)
    split_data_dir = data_folder / "split_data"

    sampler = DiversitySampler(
        emb_model="openclip",
        method="kmeans_elbow",
    )

    # Load and process data
    if args.annotated:
        print("Loading annotated dataset...")
        result = load_frames_and_masks(
            data_folder=data_folder,
            mask_type=args.mask_type,
            dataset_style=args.dataset_style,
        )
        if len(result) == 4:
            frames, masks, color_map, labels = result
        else:
            frames, masks, color_map = result
            labels = None
    else:
        print("Loading unannotated dataset...")
        frames = load_frames_from_dir(data_folder=data_folder)
        masks  = None
        labels = None

    len_total_frames = len(frames)

    # Split data
    if args.split_by_dir:
        print("Splitting by directory name...")
        split_by_directory(data_folder=data_folder)
    elif not args.skip_split:
        print("Performing random train/val/test split...")
        train_val_test_split(
            data_folder=data_folder,
            test_size=args.test_prop,
            val_size=args.val_prop,
        )

    # Load train split for sampling
    if args.skip_split:
        train_frames = frames
        train_masks  = masks
    else:
        print("Loading train split...")
        train_frames = np.load(split_data_dir / "train" / "frames.npy")
        train_masks_path = split_data_dir / "train" / "masks.npy"
        train_masks = np.load(train_masks_path) if train_masks_path.exists() else None

    num_train_samples = int(args.train_prop * len_total_frames)

    # Diversity Sampling
    diversity_out = (split_data_dir / "train") if not args.skip_split else (data_folder / "split_data" / "train")
    _, div_frames, div_masks, div_indices = sampler.sample(
        data_arr=train_frames,
        mask_arr=train_masks,
        num_samples=num_train_samples,
        run_eval=args.div_eval,
        run_manual_filter=args.filter_clusters_manually,
        save_data=True,
        data_dir=str(diversity_out),
    )

    # Random sampling
    print("Performing random sampling...")
    random_indices = np.random.choice(len(train_frames), size=num_train_samples, replace=False)
    random_out = (split_data_dir / "train" / "random") if not args.skip_split else (data_folder / "split_data" / "train" / "random")
    random_out.mkdir(parents=True, exist_ok=True)
    np.save(random_out / "frames.npy", train_frames[random_indices])
    if train_masks is not None:
        np.save(random_out / "masks.npy", train_masks[random_indices])

    if args.div_eval:
        print("Running model training evaluation...")

        train_videos = None
        val_video = None

        if args.dataset_style == "pitvis":
            train_videos = []
            print("Enter video IDs to use for training (one or more per line, blank line to finish):")
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

            print(f"Training videos: {train_videos}")
            print(f"Validation video: {val_video}")

        eval.main(
            task=args.task_evaluation,
            dataset_root=args.data_folder,
            dataset_style=args.dataset_style,
            train_videos=train_videos,
            val_video=val_video,
            output_dir=str(data_folder / "eval_outputs"),
            model_name=args.model_name if hasattr(args, "model_name") else None,
            num_classes=args.num_classes
        )

    print("Done!")