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
from data_manager import load_frames_and_masks, infer_split_from_path
from data_partitioner import train_val_test_split, split_by_directory_name

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Split dataset into train, val, and test sets, and create the training set using diversity sampling or random sampling"
    )
    parser.add_argument(
        "--base-dir", default="./Cholec8K/data", type=str,
        help="Root directory of the dataset"
    )
    parser.add_argument(
        "--data-dir", default="raw", type=str,
        help="Directory of the frames and masks"
    )
    parser.add_argument(
        "--parse-unannotated", action='store_true',
        help="Whether to parse unannotated dataset from scratch"
    )
    parser.add_argument(
        "--skip-split", action='store_true',
        help="Whether to skip the train-val-test split and directly perform sampling on the entire dataset"
    )
    parser.add_argument(
        "--train-prop", default=0.1, type=float,
        help="Proportion of the dataset to be used for training"
    )
    parser.add_argument(
        "--test-prop", default=0.1, type=float,
        help="Proportion of the dataset to be used as the test set"
    )
    parser.add_argument(
        "--val-prop", default=0.1, type=float,
        help="Proportion of the dataset to be used as the validation set"
    )
    parser.add_argument(
        "--split-by-dir", action='store_true',
        help="Split dataset based on directory names (train/validation/test) rather than random splitting"
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
        "--parse-annotated", action='store_true',
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

    args = parser.parse_args()

    data_dir      = os.path.join(args.base_dir, args.data_dir)
    processed_dir = os.path.join(args.base_dir, "processed_data")
    split_dir     = os.path.join(args.base_dir, "split_data") if not args.skip_split else None

    sampler = DiversitySampler(
        emb_model='openclip',
        method='kmeans_elbow',
    )

    len_total_frames = None

    if args.parse_annotated:
        if args.split_by_dir:
            print("Splitting raw dataset by directory name...")
            split_subdirs = {}
            for d in sorted(Path(data_dir).iterdir()):
                if d.is_dir():
                    split_name = infer_split_from_path(d.parts)
                    split_subdirs.setdefault(split_name, []).append(d)

            frames_by_split = {}
            masks_by_split = {}
            color_map = None
            global_color_map = None

            for split_name, subdirs in split_subdirs.items():
                split_frames_list = []
                split_masks_list = []
                for subdir in subdirs:
                    f, m, global_color_map = load_frames_and_masks(
                        dataset_root=subdir,
                        mask_type=args.mask_type,
                        dataset_style=args.dataset_style,
                        global_color_map=global_color_map,
                    )
                    split_frames_list.append(f)
                    split_masks_list.append(m)
                frames_by_split[split_name] = np.concatenate(split_frames_list, axis=0)
                masks_by_split[split_name]  = np.concatenate(split_masks_list, axis=0)
                color_map = global_color_map

            split_by_directory_name(
                frames_by_split=frames_by_split,
                masks_by_split=masks_by_split if masks_by_split else None,
                color_map=color_map,
                out_folder=split_dir,
            )

        else:
            print("Parsing raw dataset...")
            frames, masks, color_map = load_frames_and_masks(
                dataset_root=data_dir,
                mask_type=args.mask_type,
                dataset_style=args.dataset_style,
            )

            os.makedirs(processed_dir, exist_ok=True)
            print(f"Saving processed frames and masks to {processed_dir}...")
            np.save(os.path.join(processed_dir, "all_frames.npy"), frames)
            if masks is not None:
                np.save(os.path.join(processed_dir, "all_masks.npy"), masks)
            if color_map:
                with open(os.path.join(processed_dir, "color_map.json"), "w") as f:
                    json.dump({str(k): v for k, v in color_map.items()}, f, indent=2)
            print(f"Saved {len(frames)} frames and masks.")

            if not args.skip_split:
                print("Performing train-val-test split...")
                len_total_frames = len(frames)
                train_val_test_split(
                    data_arr=frames,
                    mask_arr=masks,
                    test_size=args.test_prop,
                    val_size=args.val_prop,
                    shuffle=True,
                    seed=42,
                    out_folder=split_dir,
                )

    else:
        if not args.skip_split:
            if not os.path.exists(os.path.join(processed_dir, "all_frames.npy")):
                raise FileNotFoundError(
                    "processed_data/all_frames.npy not found. "
                    "Run with --parse-annotated first, or use --parse-annotated --split-by-dir "
                    "if your data is already organized by directory name."
                )
            print("Loading processed frames and masks...")
            frames = np.load(os.path.join(processed_dir, "all_frames.npy"))
            masks_path = os.path.join(processed_dir, "all_masks.npy")
            masks = np.load(masks_path) if os.path.exists(masks_path) else None
            len_total_frames = len(frames)

            print("Performing train-val-test split...")
            train_val_test_split(
                data_arr=frames,
                mask_arr=masks,
                test_size=args.test_prop,
                val_size=args.val_prop,
                shuffle=True,
                seed=42,
                out_folder=split_dir,
            )

    print("Loading train split...")
    train_frames = np.load(os.path.join(split_dir, "train", "frames.npy"))
    train_masks_path = os.path.join(split_dir, "train", "masks.npy")
    train_masks = np.load(train_masks_path) if os.path.exists(train_masks_path) else None

    if len_total_frames is not None:
        num_train_samples = int(args.train_prop * len_total_frames)
    else:
        num_train_samples = int(args.train_prop * len(train_frames))

    print("Performing diversity sampling to create the diverse training set...")
    diversity_out = os.path.join(split_dir, "train", "diversity")
    _, div_frames, div_masks, div_indices = sampler.sample(
        data_arr=train_frames,
        mask_arr=train_masks,
        num_samples=num_train_samples,
        run_eval=args.div_eval,
        run_manual_filter=args.filter_clusters_manually,
        save_data=True,
        data_dir=diversity_out,
    )

    print("Performing random sampling to create the random training set...")
    random_indices = np.random.choice(len(train_frames), size=num_train_samples, replace=False)
    random_out = os.path.join(split_dir, "train", "random")
    os.makedirs(random_out, exist_ok=True)
    np.save(os.path.join(random_out, "frames.npy"), train_frames[random_indices])
    if train_masks is not None:
        np.save(os.path.join(random_out, "masks.npy"), train_masks[random_indices])

    print("Done!")