USE_RANDOM = False

#   "mask2former" 
#   "segformer"    
#   "upernet"     
#   "deeplab"      
#   "unet"        
MODEL = "mask2former"

from transformers import (
    Mask2FormerImageProcessor,
    Mask2FormerForUniversalSegmentation,
    AutoImageProcessor,
    SegformerForSemanticSegmentation,
    UperNetForSemanticSegmentation,
)
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from tqdm import tqdm

_HF_IDS = {
    "mask2former": "facebook/mask2former-swin-large-coco-panoptic",
    "segformer":   "nvidia/segformer-b5-finetuned-ade-640-640",
    "upernet":     "openmmlab/upernet-swin-large",
}
_IS_HF = MODEL in _HF_IDS


def build_model(num_classes: int):
    """Return (processor, model) for the chosen MODEL."""
    if MODEL == "mask2former":
        mid  = _HF_IDS["mask2former"]
        proc = Mask2FormerImageProcessor.from_pretrained(mid)
        mdl  = Mask2FormerForUniversalSegmentation.from_pretrained(
            mid, num_labels=num_classes, ignore_mismatched_sizes=True)
        return proc, mdl

    elif MODEL == "segformer":
        mid  = _HF_IDS["segformer"]
        proc = AutoImageProcessor.from_pretrained(mid)
        mdl  = SegformerForSemanticSegmentation.from_pretrained(
            mid, num_labels=num_classes, ignore_mismatched_sizes=True)
        return proc, mdl

    elif MODEL == "upernet":
        mid  = _HF_IDS["upernet"]
        proc = AutoImageProcessor.from_pretrained(mid)
        mdl  = UperNetForSemanticSegmentation.from_pretrained(
            mid, num_labels=num_classes, ignore_mismatched_sizes=True)
        return proc, mdl

    elif MODEL == "deeplab":
        import torchvision
        mdl = torchvision.models.segmentation.deeplabv3_resnet101(weights="DEFAULT")
        mdl.classifier[-1]     = torch.nn.Conv2d(256, num_classes, 1)
        mdl.aux_classifier[-1] = torch.nn.Conv2d(256, num_classes, 1)
        return None, mdl

    elif MODEL == "unet":
        import segmentation_models_pytorch as smp
        mdl = smp.Unet(encoder_name="resnet50", encoder_weights="imagenet",
                       in_channels=3, classes=num_classes)
        return None, mdl

    else:
        raise ValueError(f"Unknown model: {MODEL!r}")



main_data_frames   = np.load("diverse_sampled_frames.npy")
main_data_masks    = np.load("diverse_sampled_masks.npy")
random_data_frames = np.load("random_sample_frames.npy")
random_data_masks  = np.load("random_sample_masks.npy")
val_frames_raw     = np.load("val_frames.npy")
val_masks_raw      = np.load("val_masks.npy")

print("Loaded frames: ", main_data_frames.shape)
print("Loaded masks:  ", main_data_masks.shape)
print("")
print("Loaded random frames: ", random_data_frames.shape)
print("Loaded random masks:  ", random_data_masks.shape)
print("")
print("Loaded val frames: ", val_frames_raw.shape)
print("Loaded val masks:  ", val_masks_raw.shape)

if USE_RANDOM:
    print("**** USING RANDOMLY SAMPLED DATASET *****")
    main_data_frames = random_data_frames
    main_data_masks  = random_data_masks
else:
    print("**** USING DIVERSE SAMPLED DATASET *****")

fig, ax = plt.subplots(nrows=1, ncols=2)
ax[0].imshow(main_data_frames[0])
ax[1].imshow(main_data_masks[0])
plt.suptitle("Sample train frame and mask")
plt.show()

# Remap sparse class IDs to contiguous 0..N
_raw_ids = np.unique(main_data_masks)
_lut = np.zeros(int(_raw_ids.max()) + 1, dtype=np.uint8)
for new_id, raw_id in enumerate(_raw_ids):
    _lut[raw_id] = new_id
main_data_masks = _lut[main_data_masks]
NUM_CLASSES = int(_raw_ids.shape[0])
val_masks_raw = _lut[val_masks_raw.clip(0, len(_lut) - 1)]

print(f"Raw class IDs: {_raw_ids.tolist()}")
print(f"Remapped to 0-{NUM_CLASSES - 1} ({NUM_CLASSES} classes)")


# ── Model + optimizer ────────────────────────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
processor, model = build_model(NUM_CLASSES)
model = model.to(device)

if MODEL == "mask2former":
    backbone_params = list(model.model.pixel_level_module.encoder.parameters())
    other_params    = [p for p in model.parameters()
                       if not any(p is q for q in backbone_params)]
    optimizer = torch.optim.AdamW(
        [{"params": backbone_params, "lr": 1e-5},
         {"params": other_params,    "lr": 1e-4}],
        weight_decay=1e-4,
    )
else:
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)

NUM_EPOCHS = 100
scheduler  = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)


# ── Dataset / DataLoader ─────────────────────────────────────────────────────
class SegmentationDataset(Dataset):
    def __init__(self, frames: np.ndarray, masks: np.ndarray):
        self.frames = frames
        self.masks  = masks

    def __len__(self):
        return len(self.frames)

    def __getitem__(self, idx):
        return self.frames[idx], self.masks[idx]


_IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
_IMAGENET_STD  = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def _cnn_preprocess(frames):
    imgs = torch.stack([
        torch.from_numpy(f.astype(np.float32) / 255.0).permute(2, 0, 1)
        for f in frames
    ])
    return (imgs - _IMAGENET_MEAN) / _IMAGENET_STD


def collate_fn(batch):
    frames, masks = zip(*batch)
    if _IS_HF:
        images   = [Image.fromarray(f.astype(np.uint8)) for f in frames]
        seg_maps = [m.astype(np.int32) for m in masks]
        return processor(images=images, segmentation_maps=seg_maps, return_tensors="pt")
    else:
        return {
            "pixel_values": _cnn_preprocess(list(frames)),
            "labels":       torch.from_numpy(np.stack(masks).astype(np.int64)),
        }


dataset     = SegmentationDataset(main_data_frames, main_data_masks)
val_dataset = SegmentationDataset(val_frames_raw, val_masks_raw)
dataloader  = DataLoader(dataset,     batch_size=4, shuffle=True,  collate_fn=collate_fn)

print(f"Dataset size: {len(dataset)}  |  Batches: {len(dataloader)}")
print(f"Val size: {len(val_dataset)}")


def _move_batch(batch):
    return {
        k: (v.to(device) if isinstance(v, torch.Tensor)
            else [t.to(device) for t in v] if isinstance(v, list) else v)
        for k, v in batch.items()
    }


def model_forward(batch) -> torch.Tensor:
    batch = _move_batch(batch)
    if _IS_HF:
        return model(**batch).loss
    if MODEL == "deeplab":
        out = model(batch["pixel_values"])
        return (F.cross_entropy(out["out"], batch["labels"]) +
                0.4 * F.cross_entropy(out["aux"], batch["labels"]))
    return F.cross_entropy(model(batch["pixel_values"]), batch["labels"])  # unet


def model_predict(frames: np.ndarray) -> list[np.ndarray]:
    """Return a list of H×W integer segmentation maps (one per frame)."""
    if _IS_HF:
        imgs   = [Image.fromarray(f.astype(np.uint8)) for f in frames]
        inputs = processor(images=imgs, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        outputs = model(**inputs)
        return [
            p.cpu().numpy()
            for p in processor.post_process_semantic_segmentation(
                outputs, target_sizes=[f.shape[:2] for f in frames])
        ]
    else:
        pixel_values = _cnn_preprocess(list(frames)).to(device)
        raw = model(pixel_values)
        if MODEL == "deeplab":
            raw = raw["out"]
        preds = []
        for i, f in enumerate(frames):
            p = F.interpolate(raw[i:i+1], size=f.shape[:2],
                              mode="bilinear", align_corners=False)
            preds.append(p.squeeze(0).argmax(0).cpu().numpy())
        return preds


def eval_metrics(pred: np.ndarray, target: np.ndarray) -> tuple[float, float]:
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


def evaluate(frames: np.ndarray, masks: np.ndarray) -> tuple[float, float]:
    model.eval()
    all_ious, all_dices = [], []
    with torch.no_grad():
        for start in range(0, len(frames), 4):
            preds = model_predict(frames[start:start + 4])
            for pred, gt in zip(preds, masks[start:start + 4]):
                iou, dice = eval_metrics(pred, gt)
                all_ious.append(iou)
                all_dices.append(dice)
    return float(np.mean(all_ious)), float(np.mean(all_dices))


epoch_losses = []
for epoch in range(NUM_EPOCHS):
    model.train()
    train_loss = 0.0
    pbar = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{NUM_EPOCHS}", leave=False)
    for batch in pbar:
        loss = model_forward(batch)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        train_loss += loss.item()
        pbar.set_postfix(loss=f"{loss.item():.4f}")
    scheduler.step()
    train_loss /= len(dataloader)
    epoch_losses.append(train_loss)

    torch.save(model.state_dict(), f"{MODEL}_models/{MODEL}_epoch{epoch}.pth")

    miou, mdice = evaluate(val_frames_raw, val_masks_raw)
    print(f"Epoch {epoch + 1:>3}/{NUM_EPOCHS} | loss {train_loss:.4f} | "
          f"mIoU {miou:.4f} | mDice {mdice:.4f}")


fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(range(1, NUM_EPOCHS + 1), epoch_losses, linewidth=1.5)
ax.set_xlabel("Epoch")
ax.set_ylabel("Loss")
ax.set_title(f"Training loss per epoch ({MODEL})")
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

CMAP = plt.get_cmap("tab20", NUM_CLASSES)

for i in range(10):
    frame, mask = dataset[i]
    fig, (ax_img, ax_msk) = plt.subplots(1, 2, figsize=(8, 4))
    fig.suptitle(f"Sample {i}  |  classes present: {np.unique(mask).tolist()}", fontsize=10)
    ax_img.imshow(frame.astype(np.uint8))
    ax_img.set_title("image")
    ax_img.axis("off")
    im = ax_msk.imshow(mask, cmap=CMAP, vmin=0, vmax=NUM_CLASSES - 1)
    ax_msk.set_title("mask (remapped)")
    ax_msk.axis("off")
    fig.colorbar(im, ax=ax_msk, fraction=0.046, pad=0.04, label="class id")
    plt.tight_layout()
    plt.show()


print("\nFinal Evaluation")
train_miou, train_mdice = evaluate(main_data_frames, main_data_masks)
test_miou,  test_mdice  = evaluate(val_frames_raw, val_masks_raw)
print(f"Train  |  mIoU {train_miou:.4f}  mDice {train_mdice:.4f}")
print(f"Test   |  mIoU {test_miou:.4f}  mDice {test_mdice:.4f}")

N_VIS   = 8
vis_idx = np.linspace(0, len(val_frames_raw) - 1, N_VIS, dtype=int)
model.eval()

with torch.no_grad():
    for i in vis_idx:
        frame = val_frames_raw[i]
        gt    = val_masks_raw[i]
        pred  = model_predict([frame])[0]
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
        plt.tight_layout()
        plt.show()
