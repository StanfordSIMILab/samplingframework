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


def eval_metrics(pred: np.ndarray, target: np.ndarray) -> tuple[float, float]:
    """Mean IoU and Dice over classes present in target."""
    ious, dices = [], []
    for cls in range(NUM_CLASSES):
        p = pred == cls
        t = target == cls
        if not t.any():
            continue
        inter = (p & t).sum()
        union = (p | t).sum()
        ious.append(inter / union if union else 0.0)
        dices.append(2 * inter / (p.sum() + t.sum()) if (p.sum() + t.sum()) else 0.0)
    return float(np.mean(ious)), float(np.mean(dices))


def evaluate(mdl, frames: np.ndarray, masks: np.ndarray) -> tuple[float, float]:
    mdl.eval()
    all_ious, all_dices = [], []
    with torch.no_grad():
        for start in range(0, len(frames), 4):
            imgs = [
                Image.fromarray(f.astype(np.uint8)) for f in frames[start : start + 4]
            ]
            msks = masks[start : start + 4]
            inputs = processor(images=imgs, return_tensors="pt")
            inputs = {k: v.to(device) for k, v in inputs.items()}

            outputs = mdl(**inputs)
            pred_maps = processor.post_process_semantic_segmentation(
                outputs, target_sizes=[m.shape[:2] for m in msks]
            )
            for pred, gt in zip(pred_maps, msks):
                iou, dice = eval_metrics(pred.cpu().numpy(), gt)
                all_ious.append(iou)
                all_dices.append(dice)
    return float(np.mean(all_ious)), float(np.mean(all_dices))

if __name__ == "__main__":
    # Parseable parameters
    parser = argparse.ArgumentParser(
        description="Train Mask2Former on Diversity Sampled vs. Randomly Sampled Data"
    )
    parser.add_argument(
        "--base_dir", default=".",
        help="Root directory of the dataset"
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

    processor = Mask2FormerImageProcessor.from_pretrained(model_id)

    #Create Dataset
    main_data_frames = np.load("./EndoVis_processed/diverse_sampled_frames.npy")
    main_data_masks = np.load("./EndoVis_processed/diverse_sampled_masks.npy")

    random_data_frames = np.load("./EndoVis_processed/random_sample_frames.npy")
    random_data_masks = np.load("./EndoVis_processed/random_sample_masks.npy")

    val_frames_raw = np.load("./EndoVis_processed/val_frames.npy")
    val_masks_raw = np.load("./EndoVis_processed/val_masks.npy")

    print("Loaded frames: ", main_data_frames.shape)
    print("Loaded masks:  ", main_data_masks.shape)
    print("")
    print("Loaded random frames: ", random_data_frames.shape)
    print("Loaded random masks:  ", random_data_masks.shape)
    print("")
    print("Loaded val frames: ", val_frames_raw.shape)
    print("Loaded val masks:  ", val_masks_raw.shape)

    # Determine which sampling framework and create output directory
    data_type = None
    if args.use_diversity_sampling:
        print("**** USING DIVERSE SAMPLED DATASET *****")
        data_type = "diversity"
    else:
        print("**** USING RANDOMLY SAMPLED DATASET *****")
        main_data_frames = random_data_frames
        main_data_masks = random_data_masks
        data_type = "random"

    os.makedirs(f"{args.base_dir}/{data_type}", exist_ok=True)
    output_dir = f"./{data_type}"

    fig, ax = plt.subplots(nrows=1, ncols=2)
    ax[0].imshow(main_data_frames[0])
    ax[1].imshow(main_data_masks[0])

    plt.suptitle("Sample train frame and mask")
    plt.savefig(f"{output_dir}/sample_train_frame_mask.png")
    plt.show()
    plt.close()

    # Remap sparse dataset class IDs (e.g. 0,5,11,12,...,50) to contiguous 0-N.
    # Mask2Former and the colormap both require contiguous indices starting at 0.
    _raw_ids = np.unique(main_data_masks)  # sorted unique values in the data
    _lut = np.zeros(int(_raw_ids.max()) + 1, dtype=np.uint8)
    for new_id, raw_id in enumerate(_raw_ids):
        _lut[raw_id] = new_id
    main_data_masks = _lut[main_data_masks]  # vectorized remap
    NUM_CLASSES = int(_raw_ids.shape[0])

    print(f"Raw class IDs: {_raw_ids.tolist()}")
    print(f"Remapped to 0-{NUM_CLASSES - 1} ({NUM_CLASSES} classes)")

    # Create dataset
    dataset = SegmentationDataset(main_data_frames, main_data_masks)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True, collate_fn=collate_fn)

    print(f"Dataset size: {len(dataset)}  |  Batches: {len(dataloader)}")

    val_masks_raw = _lut[val_masks_raw.clip(0, len(_lut) - 1)]

    val_dataset = SegmentationDataset(val_frames_raw, val_masks_raw)
    val_dataloader = DataLoader(
        val_dataset, batch_size=4, shuffle=False, collate_fn=collate_fn
    )
    print(f"Val size: {len(val_dataset)}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = Mask2FormerForUniversalSegmentation.from_pretrained(
        model_id,
        num_labels=NUM_CLASSES,
        ignore_mismatched_sizes=True,  # backbone loads; classification head reinits
    ).to(device)

    # lower LR on pretrained backbone to preserve learned features
    backbone_params = list(model.model.pixel_level_module.encoder.parameters())
    other_params = [
        p for p in model.parameters() if not any(p is q for q in backbone_params)
    ]

    optimizer = torch.optim.AdamW(
        [{"params": backbone_params, "lr": 1e-5}, {"params": other_params, "lr": 1e-4}],
        weight_decay=1e-4,
    )

    NUM_EPOCHS = args.num_epochs
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)

    with open(f'{output_dir}/metrics.txt', 'w') as f:
            f.write("Training + Test Metrics")

    # training
    epoch_losses = []
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

        miou, mdice = evaluate(model, val_frames_raw, val_masks_raw)
        with open(f'{output_dir}/metrics.txt', 'a') as f:
            f.write(f"\nEpoch {epoch + 1:>3}/{NUM_EPOCHS} | loss {train_loss:.4f} | "
            f"mIoU {miou:.4f} | mDice {mdice:.4f}\n")
        
        print(
            f"Epoch {epoch + 1:>3}/{NUM_EPOCHS} | loss {train_loss:.4f} | "
            f"mIoU {miou:.4f} | mDice {mdice:.4f}"
        )

    torch.save(
            model.state_dict(),
            f"../Models/diverse_sample/mask2former_tuned_cseg_{data_type}.pth",
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
    train_miou, train_mdice = evaluate(model, main_data_frames, main_data_masks)
    test_miou, test_mdice = evaluate(model, val_frames_raw, val_masks_raw)
    print(f"Train  |  mIoU {train_miou:.4f}  mDice {train_mdice:.4f}")
    print(f"Test   |  mIoU {test_miou:.4f}  mDice {test_mdice:.4f}")
    with open(f'{output_dir}/metrics.txt', 'a') as f:
        f.write("\nFinal Metrics:\n"
                f"Train | mIoU {train_miou:.4f} mDice {train_mdice:.4f}\n"
                f"Test | mIoU {test_miou:.4f} mDice {test_mdice:.4f}")

    N_VIS = 8
    vis_idx = np.linspace(0, len(val_frames_raw) - 1, N_VIS, dtype=int)
    model.eval()

    with torch.no_grad():
        for i in vis_idx:
            frame = val_frames_raw[i]
            gt = val_masks_raw[i]

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

            iou, dice = eval_metrics(pred, gt)

            red_overlay = np.zeros((*pred.shape, 4), dtype=np.float32)
            red_overlay[..., 0] = 1.0
            red_overlay[..., 3] = (pred > 0).astype(np.float32) * 0.55

            fig, axes = plt.subplots(1, 3, figsize=(14, 4))
            fig.suptitle(
                f"Test sample {i}  —  mIoU {iou:.3f}  mDice {dice:.3f}\n"
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

            axes[2].imshow(frame.astype(np.uint8))
            axes[2].imshow(red_overlay)
            axes[2].set_title("pred overlay (red = foreground)")
            axes[2].axis("off")

            plt.savefig(f"{output_dir}/metrics_graph.png", bbox_inches='tight')
            plt.show()
            plt.close()
