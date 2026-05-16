### Runs Mask2Former from huggingface and runs mean IOU and mean Dice coeff for all classes
from transformers import Mask2FormerImageProcessor, Mask2FormerForUniversalSegmentation
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np

from PIL import Image
from tqdm.auto import tqdm
import os
import argparse

class SegmentationDataset(Dataset):
    def __init__(self, frames: np.ndarray, masks: np.ndarray):
        self.frames = frames
        self.masks = masks

    def __len__(self):
        return len(self.frames)

    def __getitem__(self, idx):
        return self.frames[idx], self.masks[idx]


def collate_fn(batch):
    """Pass a batch through Mask2FormerImageProcessor.
    Returns BatchFeature with pixel_values, pixel_mask, mask_labels."""
    frames, masks = zip(*batch)
    images = [Image.fromarray(f.astype(np.uint8)) for f in frames]
    seg_maps = [m.astype(np.int32) for m in masks]
    return processor(images=images, segmentation_maps=seg_maps, return_tensors="pt")

def evaluate(mdl, frames: np.ndarray, masks: np.ndarray):

    mdl.eval()

    # Only evaluate on classes present in this split's ground truth
    present_classes = [c for c in np.unique(masks) if c != IGNORE_INDEX]

    intersections = np.zeros(NUM_CLASSES, dtype=np.float64)
    unions = np.zeros(NUM_CLASSES, dtype=np.float64)
    pred_pixels = np.zeros(NUM_CLASSES, dtype=np.float64)
    target_pixels = np.zeros(NUM_CLASSES, dtype=np.float64)

    with torch.no_grad():
        for start in range(0, len(frames), 4):
            imgs = [
                Image.fromarray(f.astype(np.uint8))
                for f in frames[start:start + 4]
            ]
            msks = masks[start:start + 4]
            inputs = processor(images=imgs, return_tensors="pt")
            inputs = {k: v.to(device) for k, v in inputs.items()}
            outputs = mdl(**inputs)
            pred_maps = processor.post_process_semantic_segmentation(
                outputs,
                target_sizes=[m.shape[:2] for m in msks]
            )
            for pred, gt in zip(pred_maps, msks):
                pred = pred.cpu().numpy()
                for cls in present_classes:
                    p = pred == cls
                    t = gt == cls
                    inter = np.logical_and(p, t).sum()
                    union = np.logical_or(p, t).sum()
                    intersections[cls] += inter
                    unions[cls] += union
                    pred_pixels[cls] += p.sum()
                    target_pixels[cls] += t.sum()

    class_ious = {}
    class_dices = {}

    for cls in present_classes:
        iou = (
            intersections[cls] / unions[cls]
            if unions[cls] > 0
            else np.nan
        )
        denom = pred_pixels[cls] + target_pixels[cls]
        dice = (
            2 * intersections[cls] / denom
            if denom > 0
            else np.nan
        )
        class_ious[cls] = float(iou)
        class_dices[cls] = float(dice)

    valid_ious = [v for v in class_ious.values() if not np.isnan(v)]
    valid_dices = [v for v in class_dices.values() if not np.isnan(v)]

    miou = float(np.mean(valid_ious)) if valid_ious else float("nan")
    mdice = float(np.mean(valid_dices)) if valid_dices else float("nan")

    return miou, mdice, class_ious, class_dices


if __name__ == "__main__":
    # Parseable parameters
    parser = argparse.ArgumentParser(
        description="Train Mask2Former on Diversity Sampled vs. Randomly Sampled Data"
    )
    parser.add_argument(
        "--base_dir", default="/mnt/sda1/nishanr/Diversity_Sampling/",
        help="Root directory of the Diversity Sampling"
    )
    parser.add_argument(
        "--model_dir", default="/mnt/sda1/nishanr/Models/Diversity_Sampling",
        help="Root directory of the dataset"
    )
    parser.add_argument(
        "--data-name", default="Cholec8K",
        help="Name of the dataset for organizing data and model directories"
    )
    parser.add_argument(
        "--use-diversity-sampling", action="store_true",
        help="Use diversity_sampling, default is Random"
    )
    parser.add_argument(
        "--num-epochs", default=30, type=int, 
        help="How many epochs for training the Mask2Former"
    )
    args = parser.parse_args()

    #Make sure matplot fonts are properly managed:
    try:
        path = fm.findfont('DejaVu Sans', fallback_to_default=False)
        print(f"✓ Fonts OK: {path}")
    except Exception as e:
        raise RuntimeError(f"Matplotlib font cache is broken — clear it and rerun.\n{e}")

    # Load Model
    model_id = "facebook/mask2former-swin-large-coco-panoptic"

    processor = Mask2FormerImageProcessor.from_pretrained(
        model_id,
        do_reduce_labels=False,
        ignore_index=255,
    )

    # Determine which sampling framework and create output directory
    data_type = None
    diversity_masks = np.load(f"{args.base_dir}/{args.data_name}/data/split_data/train/diversity/masks.npy")
    random_masks    = np.load(f"{args.base_dir}/{args.data_name}/data/split_data/train/random/masks.npy")
    if args.use_diversity_sampling:
        print("**** USING DIVERSE SAMPLED DATASET *****")
        data_type = "diversity"
        main_data_frames = np.load(f"{args.base_dir}/{args.data_name}/data/split_data/train/diversity/frames.npy")
        main_data_masks = diversity_masks
        print("Loaded diversityframes: ", main_data_frames.shape)
        print("Loaded diversity masks:  ", main_data_masks.shape)
        print("")
    else:
        print("**** USING RANDOMLY SAMPLED DATASET *****")
        data_type = "random"
        main_data_frames = np.load(f"{args.base_dir}/{args.data_name}/data/split_data/train/random/frames.npy")
        main_data_masks = random_masks
        print("Loaded random frames: ", main_data_frames.shape)
        print("Loaded random masks:  ", main_data_masks.shape)
        print("")

    val_frames_raw = np.load(f"{args.base_dir}/{args.data_name}/data/split_data/val/frames.npy")
    val_masks_raw = np.load(f"{args.base_dir}/{args.data_name}/data/split_data/val/masks.npy")

    test_frames_raw = np.load(f"{args.base_dir}/{args.data_name}/data/split_data/test/frames.npy")
    test_masks_raw = np.load(f"{args.base_dir}/{args.data_name}/data/split_data/test/masks.npy")

    print("Loaded val frames: ", val_frames_raw.shape)
    print("Loaded val masks:  ", val_masks_raw.shape)
    print("")
    print("Loaded test frames: ", test_frames_raw.shape)
    print("Loaded test masks:  ", test_masks_raw.shape)

    os.makedirs(f'{args.base_dir}/{args.data_name}/{data_type}', exist_ok=True)
    output_dir = f"{args.base_dir}/{args.data_name}/{data_type}"

    fig, ax = plt.subplots(nrows=1, ncols=2)
    ax[0].imshow(main_data_frames[0])
    ax[1].imshow(main_data_masks[0])

    plt.suptitle("Sample train frame and mask")
    plt.savefig(f"{output_dir}/sample_train_frame_mask.png")
    plt.show()
    plt.close()

    #Create metrics file and write header:
    with open(f'{output_dir}/metrics.txt', 'w') as f:
        f.write(f"Model: {model_id}\n")
        f.write(f"Data type: {data_type}\n")
        f.write(f"Num epochs: {args.num_epochs}\n")
        f.write("-" * 30 + "\n")

    # Remap sparse dataset class IDs (e.g. 0,5,11,12,...,50) to contiguous 0-N.
    # Mask2Former and the colormap both require contiguous indices starting at 0.

    # 1. Build raw_ids excluding background
    _raw_ids = np.unique(np.concatenate([
        main_data_masks.ravel(),
        val_masks_raw.ravel(),
        test_masks_raw.ravel(),
    ]))
    assert _raw_ids.max() < 255, "Class IDs must be < 255 for uint8 LUT remapping"
    _raw_ids = _raw_ids[_raw_ids != 0]

    # 2. Build LUT (background 0 stays 0 in LUT, we'll overwrite it after)
    _lut = np.zeros(int(_raw_ids.max()) + 1, dtype=np.uint8)
    with open(f'{output_dir}/metrics.txt', 'a') as f:
        for new_id, raw_id in enumerate(_raw_ids):
            _lut[int(raw_id)] = new_id
            f.write(f"{raw_id} -> {new_id}\n")

    IGNORE_INDEX = 255
    _lut[0] = IGNORE_INDEX  # background maps to ignore WITHIN the LUT

    # 3. Apply LUT to all splits
    main_data_masks = _lut[main_data_masks]
    val_masks_raw   = _lut[val_masks_raw]
    test_masks_raw  = _lut[test_masks_raw]

    # Free LUT source arrays no longer needed
    del diversity_masks, random_masks, _lut


    NUM_CLASSES = len(_raw_ids)  # foreground only
    print(f"Raw class IDs: {_raw_ids.tolist()}")
    print(f"Remapped to 0-{NUM_CLASSES - 1} ({NUM_CLASSES} classes)")
    print(np.sum(main_data_masks == 12))

    # Create dataset
    dataset = SegmentationDataset(main_data_frames, main_data_masks)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True, collate_fn=collate_fn)

    print(f"Dataset size: {len(dataset)}  |  Batches: {len(dataloader)}")
    print(f"Val size: {len(val_masks_raw)}")
    print(f"Test size: {len(test_masks_raw)}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    id2label = {i: str(i) for i in range(NUM_CLASSES)}
    label2id = {v: k for k, v in id2label.items()}
    model = Mask2FormerForUniversalSegmentation.from_pretrained(
        model_id,
        num_labels=NUM_CLASSES,
        id2label=id2label,
        label2id=label2id,
        ignore_mismatched_sizes=True,
    ).to(device)

    # lower LR on pretrained backbone to preserve learned features
    backbone_params = list(model.model.pixel_level_module.parameters())
    backbone_param_ids = {id(p) for p in backbone_params}

    other_params = [
        p for p in model.parameters()
        if id(p) not in backbone_param_ids
    ]

    optimizer = torch.optim.AdamW(
        [{"params": backbone_params, "lr": 1e-5}, {"params": other_params, "lr": 1e-4}],
        weight_decay=1e-4,
    )

    NUM_EPOCHS = args.num_epochs
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)

    with open(f'{output_dir}/metrics.txt', 'a') as f:
            f.write("\nTraining + Test Metrics")

    # training
    print("\nStarting training...\n")
    epoch_losses, epoch_ious, epoch_dices = [], [], []
    for epoch in range(NUM_EPOCHS):
        model.train()
        train_loss = 0.0
        pbar = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{NUM_EPOCHS}", leave=False, position=1)
        for batch in pbar:
            batch = {
                k: (
                    v.to(device)
                    if isinstance(v, torch.Tensor)
                    else [t.to(device) for t in v] if isinstance(v, list) else v
                )
                for k, v in batch.items()
            }
            loss = model(**batch).loss
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}")
        scheduler.step()
        train_loss /= len(dataloader)
        epoch_losses.append(train_loss)

        miou, mdice, class_ious, class_dices = evaluate(model, val_frames_raw, val_masks_raw)
        epoch_ious.append(miou)
        epoch_dices.append(mdice)

        with open(f'{output_dir}/metrics.txt', 'a') as f:
            f.write(f"\nEpoch {epoch + 1:>3}/{NUM_EPOCHS} | loss {train_loss:.4f} | "
            f"mIoU {miou:.4f} | mDice {mdice:.4f}\n")
            for cls in sorted(class_ious):
                iou_val = class_ious[cls]
                dice_val = class_dices[cls]
                f.write(f"class {cls}: IoU={iou_val:.4f}, Dice={dice_val:.4f}\n")
            f.write("-" * 30 + "\n")
        
        print(
            f"Epoch {epoch + 1:>3}/{NUM_EPOCHS} | loss {train_loss:.4f} | "
            f"mIoU {miou:.4f} | mDice {mdice:.4f}"
        )
        for cls, val in class_ious.items():
            print(f"class {cls}: IoU={val:.4f}")

    os.makedirs(f"{args.model_dir}/{args.data_name}/", exist_ok=True)
    torch.save(
            model.state_dict(),
            f"{args.model_dir}/{args.data_name}/mask2former_tuned_cseg_{data_type}.pth",
        )

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(range(1, NUM_EPOCHS + 1), epoch_losses, linewidth=1.5)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Training loss per epoch")
    ax.grid(True, alpha=0.3)
    plt.savefig(f"{output_dir}/train_loss_graph.png", bbox_inches='tight')
    plt.show()
    plt.close()

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(range(1, NUM_EPOCHS + 1), epoch_ious, linewidth=1.5)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("mIoU")
    ax.set_title("Training mIoU per epoch")
    ax.grid(True, alpha=0.3)
    plt.savefig(f"{output_dir}/train_iou_graph.png", bbox_inches='tight')
    plt.show()
    plt.close()

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(range(1, NUM_EPOCHS + 1), epoch_dices, linewidth=1.5)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("mDice")
    ax.set_title("Training mDice per epoch")
    ax.grid(True, alpha=0.3)
    plt.savefig(f"{output_dir}/train_dice_graph.png", bbox_inches='tight')
    plt.show()
    plt.close()

    CMAP = plt.get_cmap("tab20", NUM_CLASSES)

    os.makedirs(f'{output_dir}/samples', exist_ok=True)
    for i in range(10):
        frame, mask = dataset[i]

        fig, (ax_img, ax_msk) = plt.subplots(1, 2, figsize=(8, 4))
        fig.suptitle(
            f"Sample {i}  |  classes present: {np.unique(mask).tolist()}", fontsize=10
        )

        ax_img.imshow(frame.astype(np.uint8))
        ax_img.set_title("image")
        ax_img.axis("off")

        im = ax_msk.imshow(mask, cmap=CMAP, vmin=0, vmax=NUM_CLASSES - 1)
        ax_msk.set_title("mask (remapped)")
        ax_msk.axis("off")
        fig.colorbar(im, ax=ax_msk, fraction=0.046, pad=0.04, label="class id")

        plt.savefig(f"{output_dir}/samples/sample{i}_data.png", bbox_inches='tight')
        plt.show()
        plt.close()

    print("\nFinal Evaluation ")
    train_miou, train_mdice, train_class_ious, train_class_dices = evaluate(model, main_data_frames, main_data_masks)
    test_miou, test_mdice, test_class_ious, test_class_dices = evaluate(model, test_frames_raw, test_masks_raw)
    print(f"Train  |  mIoU {train_miou:.4f}  mDice {train_mdice:.4f}")
    print(f"Test   |  mIoU {test_miou:.4f}  mDice {test_mdice:.4f}")
    with open(f'{output_dir}/metrics.txt', 'a') as f:
        f.write("\nFinal Metrics:\n"
                f"Train | mIoU {train_miou:.4f} mDice {train_mdice:.4f}\n")
        for cls in sorted(train_class_ious):
            iou_val = train_class_ious[cls]
            dice_val = train_class_dices[cls]
            f.write(f"class {cls}: Train IoU={iou_val:.4f}, Train Dice={dice_val:.4f}\n")
                
        f.write(f"\nTest | mIoU {test_miou:.4f} mDice {test_mdice:.4f}\n")
        for cls in sorted(test_class_ious):
            iou_val = test_class_ious[cls]
            dice_val = test_class_dices[cls]
            f.write(f"class {cls}: Test IoU={iou_val:.4f}, Test Dice={dice_val:.4f}\n")

    N_VIS = 8
    vis_idx = np.linspace(0, len(test_frames_raw) - 1, N_VIS, dtype=int)
    model.eval()

    with torch.no_grad():
        os.makedirs(f"{output_dir}/samples", exist_ok=True)
        for i in vis_idx:
            frame = test_frames_raw[i]
            gt = test_masks_raw[i]

            inp = processor(
                images=[Image.fromarray(frame.astype(np.uint8))], return_tensors="pt"
            )
            inp = {k: v.to(device) for k, v in inp.items()}
            pred = (
                processor.post_process_semantic_segmentation(
                    model(**inp), target_sizes=[(frame.shape[0], frame.shape[1])]
                )[0]
                .cpu()
                .numpy()
            )

            fig, axes = plt.subplots(1, 3, figsize=(14, 4))
            fig.suptitle(
                f"Test sample {i}\n"
                f"Global  |  Train: mIoU {train_miou:.3f}  mDice {train_mdice:.3f}"
                f"     Test: mIoU {test_miou:.3f}  mDice {test_mdice:.3f}",
                fontsize=9,
            )

            axes[0].imshow(frame.astype(np.uint8))
            axes[0].set_title("image")
            axes[0].axis("off")

            im = axes[1].imshow(gt, cmap=CMAP, vmin=0, vmax=NUM_CLASSES - 1)
            axes[1].set_title("GT mask")
            axes[1].axis("off")
            fig.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04, label="class id")

            im2 = axes[2].imshow(pred, cmap=CMAP, vmin=0, vmax=NUM_CLASSES - 1, alpha=0.55)
            axes[2].set_title("pred mask")
            axes[2].axis("off")
            fig.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04, label="class id")

            plt.savefig(f"{output_dir}/samples/{i}_metrics_graph.png", bbox_inches='tight')
            plt.show()
            plt.close()
