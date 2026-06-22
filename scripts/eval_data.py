# eval_data.py
# Evaluation utilities for determining effectiveness of diversity sampling vs. random sampling
import os
import cv2
import logging

from PIL import Image
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
from kneed import KneeLocator

import matplotlib.pyplot as plt
from transformers import (
    Mask2FormerImageProcessor,
    Mask2FormerForUniversalSegmentation,
    AutoImageProcessor,
    SegformerForSemanticSegmentation,
    UperNetForSemanticSegmentation,
)

from auxiliary import data_manager as dm
from diversity_sampler import DiversitySampler


_HF_IDS = {
    "mask2former": "facebook/mask2former-swin-large-coco-panoptic",
    "segformer":   "nvidia/segformer-b5-finetuned-ade-640-640",
    "upernet":     "openmmlab/upernet-swin-large",
}

# Logger for obtaining evaluation metrics
def setup_logger(output_dir: str) -> logging.Logger:
    os.makedirs(output_dir, exist_ok=True)
    logger = logging.getLogger("eval_metrics")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fh = logging.FileHandler(os.path.join(output_dir, "evaluation_metrics.txt"), mode="w")
    fh.setLevel(logging.INFO)
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    formatter = logging.Formatter("%(message)s")
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger

# Dataset utilities
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

class HFSegDataset(Dataset):
    def __init__(self, frames: np.ndarray, labels: np.ndarray, processor):
        self.frames = frames
        self.labels = labels
        self.processor = processor

    def __len__(self) -> int:
        return len(self.frames)

    def __getitem__(self, idx: int):
        img = Image.fromarray(self.frames[idx])
        mask = self.labels[idx]
        encoding = self.processor(images=img, segmentation_maps=mask, return_tensors="pt")
        return {k: v.squeeze(0) for k, v in encoding.items()}

# Model utilities
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


def build_model(num_classes: int, model_name: str | None = None):
    if model_name == "mask2former":
        mid  = _HF_IDS["mask2former"]
        proc = Mask2FormerImageProcessor.from_pretrained(mid)
        mdl  = Mask2FormerForUniversalSegmentation.from_pretrained(
            mid, num_labels=num_classes, ignore_mismatched_sizes=True)
        return proc, mdl

    elif model_name == "segformer":
        mid  = _HF_IDS["segformer"]
        proc = AutoImageProcessor.from_pretrained(mid)
        mdl  = SegformerForSemanticSegmentation.from_pretrained(
            mid, num_labels=num_classes, ignore_mismatched_sizes=True)
        return proc, mdl

    elif model_name == "upernet":
        mid  = _HF_IDS["upernet"]
        proc = AutoImageProcessor.from_pretrained(mid)
        mdl  = UperNetForSemanticSegmentation.from_pretrained(
            mid, num_labels=num_classes, ignore_mismatched_sizes=True)
        return proc, mdl

    elif model_name == "deeplab":
        import torchvision
        mdl = torchvision.models.segmentation.deeplabv3_resnet101(weights="DEFAULT")
        mdl.classifier[-1]     = torch.nn.Conv2d(256, num_classes, 1)
        mdl.aux_classifier[-1] = torch.nn.Conv2d(256, num_classes, 1)
        return None, mdl

    elif model_name == "unet":
        import segmentation_models_pytorch as smp
        mdl = smp.Unet(encoder_name="resnet50", encoder_weights="imagenet",
                       in_channels=3, classes=num_classes)
        return None, mdl

    else:
        raise ValueError(f"Unknown model: {model_name!r}")

# Training loop for hugging face vs. other
def iterations_to_converge(losses: list) -> int | None:
    if len(losses) < 3:
        return None
    x = list(range(len(losses)))
    knee = KneeLocator(x, losses, curve="convex", direction="decreasing")
    return knee.knee


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
    logger: logging.Logger | None = None,
):
    log = logger.info if logger else print

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
        log(
            f"[{label}] {epoch + 1}/{num_epochs}"
            f"  loss={history['train_loss'][-1]:.3f}"
            f"  val_acc={history['val_acc'][-1]:.3f}"
        )

    preds = np.array(all_preds)
    gts = np.array(all_gts)
    bal_acc = balanced_accuracy_score(gts, preds)
    log(f"\n[{label}] Balanced accuracy: {bal_acc:.4f}")
    log(classification_report(gts, preds, target_names=class_names, zero_division=0))

    convergence_epoch = iterations_to_converge(history["val_loss"])
    if convergence_epoch is not None:
        log(f"With sample size {len(x_train)}, it took {convergence_epoch + 1} iterations for loss to converge")
    else:
        log(f"With sample size {len(x_train)}, loss did not clearly converge within {num_epochs} epochs")

    return model, history, preds, gts, bal_acc


def train_loop_hf(
    x_train: np.ndarray,
    train_labels: np.ndarray,
    x_val: np.ndarray,
    val_labels: np.ndarray,
    label: str,
    num_classes: int,
    class_names: list,
    model: nn.Module,
    processor,
    model_name: str,
    num_epochs: int = 30,
    logger: logging.Logger | None = None,
):
    log = logger.info if logger else print

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    train_dl = DataLoader(
        HFSegDataset(x_train, train_labels, processor),
        batch_size=4,
        shuffle=True,
        num_workers=0,
    )
    val_dl = DataLoader(
        HFSegDataset(x_val, val_labels, processor),
        batch_size=4,
        shuffle=False,
        num_workers=0,
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}

    for epoch in range(num_epochs):
        model.train()
        tl, tc, tt = 0.0, 0, 0
        for batch in train_dl:
            batch = {k: v.to(device) for k, v in batch.items()}
            optimizer.zero_grad()
            outputs = model(**batch)
            loss = outputs.loss
            loss.backward()
            optimizer.step()

            if model_name == "mask2former":
                logits = outputs.masks_queries_logits
                preds = logits.argmax(1)
            else:
                logits = outputs.logits
                preds = logits.argmax(1)

            target = batch.get("labels", batch.get("mask_labels"))
            if target is not None:
                valid = target != 255
                tc += (preds[valid] == target[valid]).sum().item()
                tt += valid.sum().item()
            tl += loss.item() * len(batch["pixel_values"])

        model.eval()
        vl, vc, vt = 0.0, 0, 0
        all_preds, all_gts = [], []
        with torch.no_grad():
            for batch in val_dl:
                batch = {k: v.to(device) for k, v in batch.items()}
                outputs = model(**batch)
                vl += outputs.loss.item() * len(batch["pixel_values"])

                if model_name == "mask2former":
                    logits = outputs.masks_queries_logits
                    preds = logits.argmax(1)
                else:
                    logits = outputs.logits
                    preds = logits.argmax(1)

                target = batch.get("labels", batch.get("mask_labels"))
                if target is not None:
                    valid = target != 255
                    vc += (preds[valid] == target[valid]).sum().item()
                    vt += valid.sum().item()
                    all_preds.extend(preds[valid].cpu().numpy())
                    all_gts.extend(target[valid].cpu().numpy())

        history["train_loss"].append(tl / len(x_train))
        history["val_loss"].append(vl / len(x_val))
        history["train_acc"].append(tc / max(tt, 1))
        history["val_acc"].append(vc / max(vt, 1))
        log(
            f"[{label}] {epoch + 1}/{num_epochs}"
            f"  loss={history['train_loss'][-1]:.3f}"
            f"  val_acc={history['val_acc'][-1]:.3f}"
        )

    preds = np.array(all_preds)
    gts = np.array(all_gts)
    bal_acc = balanced_accuracy_score(gts, preds)
    log(f"\n[{label}] Balanced accuracy: {bal_acc:.4f}")
    log(classification_report(gts, preds, target_names=class_names, zero_division=0))

    convergence_epoch = iterations_to_converge(history["val_loss"])
    if convergence_epoch is not None:
        log(f"With sample size {len(x_train)}, it took {convergence_epoch + 1} iterations for loss to converge")
    else:
        log(f"With sample size {len(x_train)}, loss did not clearly converge within {num_epochs} epochs")

    return model, history, preds, gts, bal_acc


# Training scripts for phase_classifier vs. segmentation
def train_phase_classifier(
    x_train, train_labels, x_val, val_labels,
    label, num_classes, class_names, num_epochs=30, logger: logging.Logger | None = None,
):
    model = LightCNN(num_classes)
    return train_loop(
        x_train, train_labels, x_val, val_labels,
        label, num_classes, class_names, model, num_epochs, logger,
    )


def train_segmentation_model(
    x_train, train_labels, x_val, val_labels,
    label, num_classes, class_names, num_epochs=30, model_name: str | None = None,
    logger: logging.Logger | None = None,
):
    processor, model = build_model(num_classes, model_name=model_name)

    if model_name in {"mask2former", "segformer", "upernet"}:
        if model_name == "mask2former":
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

        return train_loop_hf(
            x_train, train_labels, x_val, val_labels,
            label, num_classes, class_names, model, processor, model_name, num_epochs, logger,
        )

    return train_loop(
        x_train, train_labels, x_val, val_labels,
        label, num_classes, class_names, model, num_epochs, logger,
    )

# Plot utilities
def plot_training_curves(hist_div: dict, hist_rand: dict, output_dir: str) -> None:
    os.makedirs(output_dir, exist_ok=True)

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
    plt.savefig(f"{output_dir}/training_curves.png", dpi=150)
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
    output_dir: str,
) -> None:
    os.makedirs(output_dir, exist_ok=True)

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
    plt.savefig(f"{output_dir}/confusion_matrices.png", dpi=150)
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
    output_dir: str,
) -> None:
    os.makedirs(output_dir, exist_ok=True)

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
    plt.savefig(f"{output_dir}/per_class_f1.png", dpi=150)
    plt.show()


def plot_balanced_accuracy(bal_div: float, bal_rand: float, output_dir: str) -> None:
    os.makedirs(output_dir, exist_ok=True)

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
    plt.savefig(f"{output_dir}/balanced_accuracy.png", dpi=150)
    plt.show()

# Main function to run the entire evaluation pipeline
def main(
    task: str,
    dataset_root: str,
    dataset_style: str = "cholec",
    val_root: str | None = None,
    train_videos: list | None = None,
    val_video: str | None = None,
    output_dir: str = "outputs",
    model_name: str | None = None,
    num_classes: int | None = None,
    num_epochs: int = 30,
) -> None:
    # Set-up output structure
    training_dir = os.path.join(output_dir, "training_comparison")
    coverage_dir = os.path.join(output_dir, "data_coverage")
    os.makedirs(training_dir, exist_ok=True)
    os.makedirs(coverage_dir, exist_ok=True)

    # Initialize logger
    logger = setup_logger(output_dir)

    logger.info(f"Task: {task} | Dataset: {dataset_style} | Model: {model_name}")
    logger.info(f"Dataset root: {dataset_root}")

    if dataset_style == "pitvis":
        frames_train, _, color_map, labels_train_raw = dm.load_frames_and_masks(
            data_folder=dataset_root,
            videos=[str(v) for v in train_videos] if train_videos else None,
            dataset_style="pitvis",
        )
        frames_val, _, _, labels_val_raw = dm.load_frames_and_masks(
            data_folder=dataset_root,
            videos=[str(val_video)] if val_video is not None else None,
            dataset_style="pitvis",
        )

        step_names = (
            pd.read_csv(f"{dataset_root}/map_steps.csv")
            .drop_duplicates("int_step")
            .set_index("int_step")["str_step"]
            .to_dict()
        )
        all_steps = np.unique(np.concatenate([labels_train_raw[:, 0], labels_val_raw[:, 0]]))
        class_map = {s: i for i, s in enumerate(all_steps)}
        inferred_classes = len(class_map)
        num_classes = num_classes if num_classes is not None else inferred_classes
        class_names = [step_names.get(s, str(s)).strip() for s in all_steps]

        def remap(col: np.ndarray) -> np.ndarray:
            return np.array([class_map[s] for s in col], dtype=np.int64)

        all_labels_train = remap(labels_train_raw[:, 0])
        labels_val = remap(labels_val_raw[:, 0])

    else:
        frames_train, masks_train, color_map = dm.load_frames_and_masks(
            data_folder=dataset_root,
            videos=[str(v) for v in train_videos] if train_videos else None,
            dataset_style=dataset_style,
        )
        _val_root = val_root if val_root is not None else dataset_root
        frames_val, masks_val, _ = dm.load_frames_and_masks(
            data_folder=_val_root,
            videos=[str(val_video)] if val_video is not None else None,
            dataset_style=dataset_style,
            global_color_map=color_map,
        )

        all_labels_train = masks_train.reshape(len(masks_train), -1)[:, 0].astype(np.int64)
        labels_val = masks_val.reshape(len(masks_val), -1)[:, 0].astype(np.int64)
        inferred_classes = len(color_map)
        num_classes = num_classes if num_classes is not None else inferred_classes
        class_names = [str(i) for i in range(num_classes)]

    print(f"Using {num_classes} classes: {class_names}")

    all_frames = frames_train

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
        data_dir=output_dir,
    )

    diverse_indices = np.array(diverse_indices)
    random_indices = np.random.choice(len(all_frames), size=len(diverse_indices), replace=False)

    x_div  = all_frames[diverse_indices]
    y_div  = all_labels_train[diverse_indices]
    x_rand = all_frames[random_indices]
    y_rand = all_labels_train[random_indices]

    sampler.evaluate_vs_other(
        random_indices=random_indices,
        all_emb=normalize(all_emb),
        diverse_indices=diverse_indices,
        save_dir=coverage_dir,
        save_plot=True,
    )

    print(f"\nx_div: {x_div.shape}  y_div: {y_div.shape}")
    print(f"x_rand: {x_rand.shape}  y_rand: {y_rand.shape}")
    print(f"x_val: {frames_val.shape}  labels_val: {labels_val.shape}")

    if task == "phase_classification":
        print("\nTraining on diverse dataset...")
        _, hist_div, preds_div, gts_div, bal_div = train_phase_classifier(
            x_div, y_div, frames_val, labels_val, "diverse", num_classes, class_names, 
            num_epochs, logger)
        print("\nTraining on random dataset...")
        _, hist_rand, preds_rand, gts_rand, bal_rand = train_phase_classifier(
            x_rand, y_rand, frames_val, labels_val, "random", num_classes, class_names, 
            num_epochs, logger)
    elif task == "segmentation":
        print("\nTraining on diverse dataset...")
        _, hist_div, preds_div, gts_div, bal_div = train_segmentation_model(
            x_div, y_div, frames_val, labels_val, "diverse", num_classes, class_names, num_epochs, 
            model_name, logger)
        print("\nTraining on random dataset...")
        _, hist_rand, preds_rand, gts_rand, bal_rand = train_segmentation_model(
            x_rand, y_rand, frames_val, labels_val, "random", num_classes, class_names, num_epochs, 
            model_name, logger)
    else:
        raise ValueError(f"Unknown task: {task!r}, choose from 'phase_classification', 'segmentation'")

    plot_training_curves(hist_div, hist_rand, training_dir)
    plot_confusion_matrices(
        preds_div, gts_div, preds_rand, gts_rand,
        bal_div, bal_rand, num_classes, class_names, training_dir,
    )
    plot_per_class_f1(
        preds_div, gts_div, preds_rand, gts_rand,
        bal_div, bal_rand, num_classes, class_names, training_dir,
    )
    plot_balanced_accuracy(bal_div, bal_rand, training_dir)

    logger.info(f"\nBalanced accuracy — diverse: {bal_div:.4f}  |  random: {bal_rand:.4f}")


if __name__ == "__main__":
    main(
        task="phase_classification",
        dataset_root="path/to/dataset",
        dataset_style="pitvis",
        train_videos=[1, 2, 3],
        val_video=4,
    )