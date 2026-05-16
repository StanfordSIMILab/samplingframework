import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(os.path.abspath('..'))

from pathlib import Path

from div_sampling import DiversitySampler
import numpy as np
import argparse

if __name__ == "__main__":
    """
    Specifics for Dataset Structures:
        Split raw data by directory name directly (no processed .npy saved):
        python get_main_dataset.py --parse-raw --split-by-dir

        Full pipeline, if split not designated by file name: parse raw → save processed .npy → random split:
        python get_main_dataset.py --parse-raw

        Re-run random split on existing processed .npy:
        python get_main_dataset.py

        CAN'T DO:
        python get_main_dataset.py --split-by-dir w/o --parse-raw because will expect processed .npy

        Skip all splitting, just re-run sampling on existing split_data/train:
        python get_main_dataset.py --skip-split

    Additional options:
        --div-eval: run evaluation metrics when diversity sampling the training set
        --filter-clusters-manually: whether user wants capacity to manually filter clusters from embed/cluster results
        --mask-type: mask type to use when parsing raw data: 'color_mask', 'mask', or 'watershed_mask'
        --dataset-style: dataset folder structure style: 'cholec', 'flat', or 'nested' (only relevant if parsing raw data)
    """
    parser = argparse.ArgumentParser(
        description="Split dataset into train, val, and test sets, and create the training set using diversity sampling or random sampling"
    )
    parser.add_argument(
        "--base-dir", default="../../../Cholec8K/data", type=str,
        help="Root directory of the dataset"
    )
    parser.add_argument(
        "--data-dir", default="raw", type=str,
        help="Directory of the frames and masks"
    )
    parser.add_argument(
        "--train-size", default=0.1, type=float,
        help="Proportion of the dataset to be used for training"
    )
    parser.add_argument(
        "--test-size", default=0.1, type=float,
        help="Proportion of the dataset to be used as the test set"
    )
    parser.add_argument(
        "--val-size", default=0.1, type=float,
        help="Proportion of the dataset to be used as the validation set"
    )
    parser.add_argument(
        "--skip-split", action='store_true',
        help="Whether to skip the train-val-test split and directly perform sampling on the entire dataset"
    )
    parser.add_argument(
        "--div-eval", action='store_true',
        help="Whether to run evaluation metrics when diversity sampling the training set"
    )
    parser.add_argument(
        "--filter-clusters-manually", action='store_true',
        help="Whether user wants capacity to manually filter clusters from embed/cluster results"
    )
    parser.add_argument(
        "--parse-raw", action='store_true',
        help="Whether to parse raw dataset from scratch, saving all frames and masks to processed_data/"
    )
    parser.add_argument(
        "--mask-type", default="color_mask", type=str,
        help="Mask type to use when parsing raw data: 'color_mask', 'mask', or 'watershed_mask'"
    )
    parser.add_argument(
        "--dataset-style", default="cholec", type=str,
        help="Dataset folder structure style: 'cholec', 'flat', or 'nested'"
    )
    parser.add_argument(
        "--split-by-dir", action='store_true',
        help="Split dataset based on directory names (val/validation/test) rather than random splitting"
    )
    args = parser.parse_args()

    # Resolve paths
    data_dir      = os.path.join(args.base_dir, args.data_dir)
    processed_dir = os.path.join(args.base_dir, "processed_data")
    split_dir     = os.path.join(args.base_dir, "split_data")

    # Create DiversitySampler
    sampler = DiversitySampler(
        emb_model='openclip',
        method='kmeans_elbow',
        save_path=args.base_dir
    )

    len_total_frames = None

    # ------------------------------------------------------------------ #
    # Path 1: Parse raw dataset from scratch
    # ------------------------------------------------------------------ #
    if args.parse_raw:
        if args.split_by_dir:
            # Directly split raw data by directory name, no intermediate .npy needed
            print("Splitting raw dataset by directory name...")
            sampler.separate_by_dir_name(
                data_dir=data_dir,
                mask_type=args.mask_type,
                output_dir="split_data"
            )

        else:
            # Full pipeline: raw → processed .npy → random split
            print("Parsing raw dataset...")
            frames, masks, color_map = parse_dataset_with_masks(
                dataset_root=data_dir,
                mask_type=args.mask_type,
                dataset_style=args.dataset_style,
            )

            os.makedirs(processed_dir, exist_ok=True)
            print(f"Saving processed frames and masks to {processed_dir}...")
            np.save(os.path.join(processed_dir, "all_frames.npy"), frames)
            np.save(os.path.join(processed_dir, "all_masks.npy"), masks)
            if color_map:
                with open(os.path.join(processed_dir, "color_map.json"), "w") as f:
                    json.dump(color_map, f, indent=2)
            print(f"Saved {len(frames)} frames and masks.")

            if not args.skip_split:
                print("Performing train-val-test split...")
                len_total_frames = len(frames)
                sampler.separate_test_val(
                    data_arr=frames,
                    mask_arr=masks,
                    test_size=args.test_size,
                    val_size=args.val_size,
                    shuffle=True,
                    seed=42,
                    output_dir="split_data"
                )

    # ------------------------------------------------------------------ #
    # Path 2: Use existing processed .npy files
    # ------------------------------------------------------------------ #
    else:
        if not args.skip_split:
            if not os.path.exists(os.path.join(processed_dir, "all_frames.npy")):
                raise FileNotFoundError(
                    "processed_data/all_frames.npy not found. "
                    "Run with --parse-raw first, or use --parse-raw --split-by-dir "
                    "if your data is already organized by directory name."
                )
            print("Loading processed frames and masks...")
            frames = np.load(os.path.join(processed_dir, "all_frames.npy"))
            masks  = np.load(os.path.join(processed_dir, "all_masks.npy"))
            len_total_frames = len(frames)

            print("Performing train-val-test split...")
            sampler.separate_test_val(
                data_arr=frames,
                mask_arr=masks,
                test_size=args.test_size,
                val_size=args.val_size,
                shuffle=True,
                seed=42,
                output_dir="split_data"
            )

    # ------------------------------------------------------------------ #
    # Load train split for sampling (always needed)
    # ------------------------------------------------------------------ #
    print("Loading train split...")
    train_frames = np.load(os.path.join(split_dir, "train", "frames.npy"))
    train_masks  = np.load(os.path.join(split_dir, "train", "masks.npy"))

    # Determine number of training samples
    if len_total_frames is not None:
        num_train_samples = int(args.train_size * len_total_frames)
    else:
        num_train_samples = int(args.train_size * len(train_frames))

    # ------------------------------------------------------------------ #
    # Diversity sampling
    # ------------------------------------------------------------------ #
    print("Performing diversity sampling to create the diverse training set...")
    div_frames, div_masks, _ = sampler.create_train(
        data_arr=train_frames,
        mask_arr=train_masks,
        num_train=num_train_samples,
        run_eval=args.div_eval,
        run_manual_filter=args.filter_clusters_manually,
        train_dir="split_data/train/diversity"
    )

    # ------------------------------------------------------------------ #
    # Random sampling
    # ------------------------------------------------------------------ #
    print("Performing random sampling to create the random training set...")
    random_indices = np.random.choice(len(train_frames), size=num_train_samples, replace=False)
    os.makedirs(os.path.join(split_dir, "train", "random"), exist_ok=True)
    np.save(os.path.join(split_dir, "train", "random", "frames.npy"), train_frames[random_indices])
    np.save(os.path.join(split_dir, "train", "random", "masks.npy"),  train_masks[random_indices])

    print("Done!")
