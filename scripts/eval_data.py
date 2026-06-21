# eval_data.py
# Evaluation utilities for determining effectiveness of diversity sampling vs. random sampling
import os
import cv2

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import (
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.preprocessing import normalize

import matplotlib.pyplot as plt
from transformers import (
    Mask2FormerImageProcessor,
    Mask2FormerForUniversalSegmentation,
    AutoImageProcessor,
    SegformerForSemanticSegmentation,
    UperNetForSemanticSegmentation,
)

from diversity_sampler import DiversitySampler
from frame_extractor import load_video
from fvi_computation import compute_fvi, fvi_filter, show_fvi_histogram
from auxiliary import pitvis_extractor


_HF_IDS = {
    "mask2former": "facebook/mask2former-swin-large-coco-panoptic",
    "segformer":   "nvidia/segformer-b5-finetuned-ade-640-640",
    "upernet":     "openmmlab/upernet-swin-large",
}


class SurgicalDataset(Dataset):
    def __init__(self, frames: np.ndarray, labels: np.ndarray, size: int = 128):
        self.frames = frames
        self.labels = labels
        self.size = size

    def __len__(self) -> int:
        return len(self.frames)

    def __getitem__(self, idx: int):
        img = cv2.resize(self.frames[idx], (self.size, self.size))
        img = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
        return img, self.labels[idx]


class LightCNN(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def build_model(num_classes: int):
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


def train_loop(
    x_train: np.ndarray,
    train_labels: np.ndarray,
    x_val: np.ndarray,
    val_labels: np.ndarray,
    label: str,
    num_classes: int,
    class_names: list,
    model: nn.Module,
    num_epochs: int = 30,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    train_dl = DataLoader(
        SurgicalDataset(x_train, train_labels),
        batch_size=32,
        shuffle=True,
        num_workers=0,
    )
    val_dl = DataLoader(
        SurgicalDataset(x_val, val_labels),
        batch_size=32,
        shuffle=False,
        num_workers=0,
    )

    counts = np.bincount(train_labels, minlength=num_classes).astype(float)
    weights = torch.tensor(1.0 / (counts + 1), dtype=torch.float32).to(device)
    crit = nn.CrossEntropyLoss(weight=weights)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)

    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    all_preds, all_gts = [], []

    for epoch in range(num_epochs):
        model.train()
        tl, tc, tt = 0.0, 0, 0
        for imgs, lbls in train_dl:
            imgs, lbls = imgs.to(device), lbls.to(device)
            opt.zero_grad()
            out = model(imgs)
            loss = crit(out, lbls)
            loss.backward()
            opt.step()
            tl += loss.item() * len(imgs)
            tc += (out.argmax(1) == lbls).sum().item()
            tt += len(imgs)

        model.eval()
        vl, vc, vt = 0.0, 0, 0
        all_preds, all_gts = [], []
        with torch.no_grad():
            for imgs, lbls in val_dl:
                imgs, lbls = imgs.to(device), lbls.to(device)
                out = model(imgs)
                vl += crit(out, lbls).item() * len(imgs)
                p = out.argmax(1)
                vc += (p == lbls).sum().item()
                vt += len(imgs)
                all_preds.extend(p.cpu().numpy())
                all_gts.extend(lbls.cpu().numpy())

        history["train_loss"].append(tl / tt)
        history["val_loss"].append(vl / vt)
        history["train_acc"].append(tc / tt)
        history["val_acc"].append(vc / vt)
        print(
            f"[{label}] {epoch + 1}/{num_epochs}"
            f"  loss={history['train_loss'][-1]:.3f}"
            f"  val_acc={history['val_acc'][-1]:.3f}"
        )

    preds = np.array(all_preds)
    gts = np.array(all_gts)
    bal_acc = balanced_accuracy_score(gts, preds)
    print(f"\n[{label}] Balanced accuracy: {bal_acc:.4f}")
    print(classification_report(gts, preds, target_names=class_names, zero_division=0))

    return model, history, preds, gts, bal_acc


def train_phase_classifier(
    x_train, train_labels, x_val, val_labels,
    label, num_classes, class_names, num_epochs=30,
):
    model = LightCNN(num_classes)
    return train_loop(
        x_train, train_labels, x_val, val_labels,
        label, num_classes, class_names, model, num_epochs,
    )


def train_segmentation_model(
    x_train, train_labels, x_val, val_labels,
    label, num_classes, class_names, num_epochs=30,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    processor, model = build_model(num_classes)

    if MODEL == "mask2former":
        backbone_params = list(model.model.pixel_level_module.encoder.parameters())
        other_params = [p for p in model.parameters()
                        if not any(p is q for q in backbone_params)]
        optimizer = torch.optim.AdamW(
            [{"params": backbone_params, "lr": 1e-5},
             {"params": other_params, "lr": 1e-4}],
            weight_decay=1e-4,
        )
        for param_group in optimizer.param_groups:
            for p in param_group["params"]:
                p.requires_grad_(True)

    return train_loop(
        x_train, train_labels, x_val, val_labels,
        label, num_classes, class_names, model, num_epochs,
    )


def plot_training_curves(hist_div: dict, hist_rand: dict) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    epochs = range(1, len(hist_div["train_loss"]) + 1)
    for ax, key, ylabel in [(axes[0], "loss", "Loss"), (axes[1], "acc", "Accuracy")]:
        ax.plot(epochs, hist_div[f"train_{key}"], color="steelblue", linestyle="--", label="diverse train")
        ax.plot(epochs, hist_div[f"val_{key}"], color="steelblue", linestyle="-", label="diverse val")
        ax.plot(epochs, hist_rand[f"train_{key}"], color="darkorange", linestyle="--", label="random train")
        ax.plot(epochs, hist_rand[f"val_{key}"], color="darkorange", linestyle="-", label="random val")
        ax.set_xlabel("Epoch")
        ax.set_ylabel(ylabel)
        ax.legend()
    fig.suptitle("Training curves — diverse vs random", fontsize=13)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/training_curves.png", dpi=150)
    plt.show()


def plot_confusion_matrices(
    preds_div: np.ndarray,
    gts_div: np.ndarray,
    preds_rand: np.ndarray,
    gts_rand: np.ndarray,
    bal_div: float,
    bal_rand: float,
    num_classes: int,
    class_names: list,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    for ax, preds, gts, label, bal in [
        (axes[0], preds_div, gts_div, "diverse", bal_div),
        (axes[1], preds_rand, gts_rand, "random", bal_rand),
    ]:
        cm = confusion_matrix(gts, preds, labels=list(range(num_classes)), normalize="true")
        im = ax.imshow(cm, cmap="Blues", vmin=0, vmax=1)
        ax.set_xticks(range(num_classes))
        ax.set_yticks(range(num_classes))
        ax.set_xticklabels(class_names, rotation=45, ha="right", fontsize=7)
        ax.set_yticklabels(class_names, fontsize=7)
        ax.set_title(f"{label}  (bal acc={bal:.3f})")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        plt.colorbar(im, ax=ax)
    fig.suptitle("Normalised confusion matrix", fontsize=13)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/confusion_matrices.png", dpi=150)
    plt.show()


def plot_per_class_f1(
    preds_div: np.ndarray,
    gts_div: np.ndarray,
    preds_rand: np.ndarray,
    gts_rand: np.ndarray,
    bal_div: float,
    bal_rand: float,
    num_classes: int,
    class_names: list,
) -> None:
    f1_div = f1_score(gts_div, preds_div, labels=list(range(num_classes)), average=None, zero_division=0)
    f1_rand = f1_score(gts_rand, preds_rand, labels=list(range(num_classes)), average=None, zero_division=0)
    x_pos = np.arange(num_classes)
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.bar(x_pos - 0.2, f1_div, 0.4, color="steelblue", label=f"diverse  (bal acc={bal_div:.3f})")
    ax.bar(x_pos + 0.2, f1_rand, 0.4, color="darkorange", label=f"random   (bal acc={bal_rand:.3f})")
    ax.set_xticks(x_pos)
    ax.set_xticklabels(class_names, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("F1 score")
    ax.set_ylim(0, 1)
    ax.set_title("Per-class F1 — diverse vs random")
    ax.legend()
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/per_class_f1.png", dpi=150)
    plt.show()


def plot_balanced_accuracy(bal_div: float, bal_rand: float) -> None:
    fig, ax = plt.subplots(figsize=(5, 5))
    bars = ax.bar(
        ["diverse", "random"],
        [bal_div, bal_rand],
        color=["steelblue", "darkorange"],
        width=0.4,
    )
    ax.bar_label(bars, fmt="%.4f", padding=3)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Balanced accuracy")
    ax.set_title("Balanced accuracy — diverse vs random")
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/balanced_accuracy.png", dpi=150)
    plt.show()


def main(task=None, model_name=None, dataset=None) -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    annot_index = pitvis_extractor.load_pitvis_annotations(PITVIS_DIR, TRAIN_VIDEOS + [VAL_VIDEO])

    all_frames = []
    all_meta = []
    for vid in TRAIN_VIDEOS:
        print(f"\nLoading video {vid:02d}...")
        frames = load_video(f"{PITVIS_DIR}/video_{vid:02d}.mp4")
        fvi_scores = compute_fvi(frames)
        show_fvi_histogram(fvi_scores, vid)
        thresh = float(input(f"FVI threshold for video {vid:02d}: "))
        filtered, kept_indices, _ = fvi_filter(frames, thresh)
        print(f"Kept {len(filtered)} / {len(frames)} frames")
        all_frames.append(filtered)
        all_meta.extend([(vid, fi) for fi in kept_indices])

    all_frames = np.concatenate(all_frames, axis=0)
    print(f"\nTotal train frames after FVI filtering: {len(all_frames)}")

    sampler = DiversitySampler(
        optim_clusters=True,
        dino_model_string="dinov2_vits14",
        n_samples_per_cluster=5,
        viz_clusters=True,
        plot_chosen_frames=False,
        openclip_model_string="ViT-B-32",
        openclip_pretrained="laion2b_s34b_b79k",
        emb_model="openclip",
        method="kmeans_sil",
    )

    all_emb = sampler.run_openclip(all_frames)

    n_sampled, _, _, diverse_indices = sampler.sample(
        data_arr=all_frames,
        emb_prev=all_emb,
        method="kmeans_sil",
        run_eval=True,
        save_data=True,
        data_dir=OUTPUT_DIR,
    )

    diverse_indices = np.array(diverse_indices)
    random_indices = np.random.choice(len(all_frames), size=len(diverse_indices), replace=False)

    x_div = all_frames[diverse_indices]
    y_div = pitvis_extractor.get_pitvis_labels(annot_index, [all_meta[i] for i in diverse_indices])
    x_rand = all_frames[random_indices]
    y_rand = pitvis_extractor.get_pitvis_labels(annot_index, [all_meta[i] for i in random_indices])

    print(f"\nLoading val video {VAL_VIDEO:02d}...")
    x_val = load_video(f"{PITVIS_DIR}/video_{VAL_VIDEO:02d}.mp4")
    y_val = pitvis_extractor.get_pitvis_labels(annot_index, [(VAL_VIDEO, fi) for fi in range(len(x_val))])

    full_n    = normalize(all_emb)
    diverse_n = normalize(all_emb[diverse_indices])
    random_n  = normalize(all_emb[random_indices])

    sampler.evaluate_vs_other(
        random_indices=random_indices,
        all_emb=full_n,
        diverse_indices=diverse_indices,
        save_dir=OUTPUT_DIR,
    )

    step_names = (
        pd.read_csv(f"{PITVIS_DIR}/map_steps.csv")
        .drop_duplicates("int_step")
        .set_index("int_step")["str_step"]
        .to_dict()
    )
    all_steps = np.unique(np.concatenate([y_div[:, 0], y_rand[:, 0], y_val[:, 0]]))
    class_map = {s: i for i, s in enumerate(all_steps)}
    num_classes = len(class_map)
    class_names = [step_names.get(s, str(s)).strip() for s in all_steps]

    def remap(col: np.ndarray) -> np.ndarray:
        return np.array([class_map[s] for s in col], dtype=np.int64)

    labels_div  = remap(y_div[:, 0])
    labels_rand = remap(y_rand[:, 0])
    labels_val  = remap(y_val[:, 0])

    print(f"\nx_div: {x_div.shape}  y_div: {y_div.shape}")
    print(f"x_rand: {x_rand.shape}  y_rand: {y_rand.shape}")
    print(f"x_val: {x_val.shape}  y_val: {y_val.shape}")

    if task == "phase_classification":
        print("\nTraining on diverse dataset...")
        model_div, hist_div, preds_div, gts_div, bal_div = train_phase_classifier(
            x_div, labels_div, x_val, labels_val, "diverse", num_classes, class_names
        )
        print("\nTraining on random dataset...")
        model_rand, hist_rand, preds_rand, gts_rand, bal_rand = train_phase_classifier(
            x_rand, labels_rand, x_val, labels_val, "random", num_classes, class_names
        )
    elif task == "segmentation":
        print("\nTraining on diverse dataset...")
        model_div, hist_div, preds_div, gts_div, bal_div = train_segmentation_model(
            x_div, labels_div, x_val, labels_val, "diverse", num_classes, class_names
        )
        print("\nTraining on random dataset...")
        model_rand, hist_rand, preds_rand, gts_rand, bal_rand = train_segmentation_model(
            x_rand, labels_rand, x_val, labels_val, "random", num_classes, class_names
        )
    else:
        raise ValueError(f"Unknown task: {task}, please choose from 'phase_classification', 'segmentation'")

    plot_training_curves(hist_div, hist_rand)
    plot_confusion_matrices(
        preds_div, gts_div, preds_rand, gts_rand,
        bal_div, bal_rand, num_classes, class_names,
    )
    plot_per_class_f1(
        preds_div, gts_div, preds_rand, gts_rand,
        bal_div, bal_rand, num_classes, class_names,
    )
    plot_balanced_accuracy(bal_div, bal_rand)

    print(f"\nBalanced accuracy — diverse: {bal_div:.4f}  |  random: {bal_rand:.4f}")


if __name__ == "__main__":
    main("phase_classification")