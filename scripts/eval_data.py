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

import data_loader as dm
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
    def __init__(self, frames: np.ndarray, labels: np.ndarray, processor, model_name: str | None = None):
        self.frames = frames
        self.labels = labels
        self.processor = processor
        self.model_name = model_name

    def __len__(self) -> int:
        return len(self.frames)

    def __getitem__(self, idx: int):
        img = Image.fromarray(self.frames[idx])
        mask = self.labels[idx].astype(np.int32)
        
        if hasattr(self, 'model_name') and self.model_name == "mask2former":
            unique_ids = [int(i) for i in np.unique(mask) if i != 255]
            instance_id_to_semantic_id = {i: i for i in unique_ids}
            encoding = self.processor(
                images=img,
                segmentation_maps=Image.fromarray(mask.astype(np.int32)),
                instance_id_to_semantic_id=instance_id_to_semantic_id,
                return_tensors="pt"
            )
        else:
            encoding = self.processor(images=img, segmentation_maps=mask, return_tensors="pt")
        
        return {k: v.squeeze(0) if isinstance(v, torch.Tensor) else v
                for k, v in encoding.items()}

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

# Hugging face models
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

# Helper function to determine number of training steps (iterations) to convergence by elbow method
def iterations_to_converge(losses: list, val_losses: list | None = None) -> tuple[int | None, int | None]:
    def find_second_knee(data):
        cleaned = [v for v in data if v is not None]
        if len(cleaned) < 6:
            return None
        
        smoothed = np.convolve(cleaned, np.ones(3) / 3, mode="valid")
        x = list(range(len(smoothed)))
        
        # find first knee
        knee1 = KneeLocator(x, smoothed, curve="convex", direction="decreasing")
        if knee1.knee is None or knee1.knee >= len(smoothed) - 3:
            return None
        
        # run again on remaining curve after first knee (first knee shows the initial sharp loss decrease)
        remaining = smoothed[knee1.knee:]
        if len(remaining) < 3:
            return None
        x2 = list(range(len(remaining)))
        knee2 = KneeLocator(x2, remaining, curve="convex", direction="decreasing")
        
        if knee2.knee is None:
            return None
        
        # offset back to original index
        return knee1.knee + knee2.knee

    train_convergence = find_second_knee(losses)
    val_convergence = find_second_knee(val_losses) if val_losses is not None else None
    return train_convergence, val_convergence

# Non-Hugging Face Training Loop
def train_loop(
    x_train: np.ndarray | None = None,
    train_labels: np.ndarray | None = None,
    x_val: np.ndarray | None = None,
    val_labels: np.ndarray | None = None,
    x_test: np.ndarray | None = None,
    test_labels: np.ndarray | None = None,
    label: str = "",
    num_classes: int = 0,
    class_names: list = None,
    model: nn.Module = None,
    num_epochs: int = 30,
    batch_size: int = 1,
    logger: logging.Logger | None = None,
):
    log = logger.info if logger else print

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    train_dl = DataLoader(
        SurgicalDataset(x_train, train_labels),
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
    )

    val_dl = None
    if x_val is not None and val_labels is not None:
        val_dl = DataLoader(
            SurgicalDataset(x_val, val_labels),
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
        )

        # Create a best val_loss and best_model for running test:
        best_val_loss = float("inf")
        best_model_state = None

    test_dl = None
    if x_test is not None and test_labels is not None:
        test_dl = DataLoader(
            SurgicalDataset(x_test, test_labels),
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
        )

    counts = np.bincount(train_labels, minlength=num_classes).astype(float)
    weights = torch.tensor(1.0 / (counts + 1), dtype=torch.float32).to(device)
    crit = nn.CrossEntropyLoss(weight=weights)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)

    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    train_step_loss = []

    if num_epochs > 0: # Run test only by setting num_epochs to 0
        for epoch in range(num_epochs):
            model.train()
            tl, tc, tt = 0.0, 0, 0

            for imgs, lbls in train_dl:
                imgs, lbls = imgs.to(device), lbls.to(device)

                opt.zero_grad()
                out = model(imgs)
                loss = crit(out, lbls)
                train_step_loss.append(loss.item())
                loss.backward()
                opt.step()

                tl += loss.item() * len(imgs)
                tc += (out.argmax(1) == lbls).sum().item()
                tt += len(imgs)

            history["train_loss"].append(tl / tt)
            history["train_acc"].append(tc / tt)

            val_loss = None
            val_acc = None

            if val_dl is not None:
                model.eval()
                vl, vc, vt = 0.0, 0, 0

                with torch.no_grad():
                    for imgs, lbls in val_dl:
                        imgs, lbls = imgs.to(device), lbls.to(device)

                        out = model(imgs)
                        vl += crit(out, lbls).item() * len(imgs)

                        p = out.argmax(1)
                        vc += (p == lbls).sum().item()
                        vt += len(imgs)

                val_loss = vl / vt
                val_acc = vc / vt

            history["val_loss"].append(val_loss)
            history["val_acc"].append(val_acc)

            msg = (
                f"[{label}] {epoch + 1}/{num_epochs}"
                f"  loss={history['train_loss'][-1]:.3f}"
                f"  train_acc={history['train_acc'][-1]:.3f}"
            )
            if val_acc is not None:
                msg += (
                    f"  val_loss={val_loss:.3f}"
                    f"  val_acc={val_acc:.3f}"
                )

                # Check if it has lowest val loss (run test on model with best validation loss)
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_model_state = {k: v.clone() for k, v in model.state_dict().items()}

            log(msg)

    preds = np.array([])
    gts = np.array([])
    bal_acc = None

    if test_dl is not None:
        # If validation included, run test on model with best validation loss
        if best_model_state is not None:
            model.load_state_dict(best_model_state)
        model.eval()
        all_preds = []
        all_gts = []

        with torch.no_grad():
            for imgs, lbls in test_dl:
                imgs, lbls = imgs.to(device), lbls.to(device)

                out = model(imgs)
                p = out.argmax(1)

                all_preds.extend(p.cpu().numpy())
                all_gts.extend(lbls.cpu().numpy())

        preds = np.array(all_preds)
        gts = np.array(all_gts)

        test_acc = (preds == gts).mean()
        bal_acc = balanced_accuracy_score(gts, preds)

        log(f"\n[{label}] Test accuracy: {test_acc:.4f}")
        log(f"[{label}] Test balanced accuracy: {bal_acc:.4f}")
        log(classification_report(
                gts, preds,
                labels=list(range(num_classes)),
                target_names=class_names,
                zero_division=0
            ))
    
    # Determine the training loss convergence (training steps to learn data)
    # and validation loss convergence (training steps to generalize) if available
    train_convergence, val_convergence = iterations_to_converge(train_step_loss, val_losses=history["val_loss"])

    if train_convergence is not None:
        log(f"With sample size {len(x_train)}, train loss converged at iteration {train_convergence + 1}")
    else:
        log(f"With sample size {len(x_train)}, train loss did not clearly converge within {num_epochs} epochs")

    if val_convergence is not None:
        log(f"With sample size {len(x_train)}, val loss converged at epoch {val_convergence + 1}")
    else:
        log(f"With sample size {len(x_train)}, val loss did not clearly converge within {num_epochs} epochs")

    return model, history, preds, gts, bal_acc

# Hugging face model training loop
def train_loop_hf(
    x_train: np.ndarray | None = None,
    train_labels: np.ndarray | None = None,
    x_val: np.ndarray | None = None,
    val_labels: np.ndarray | None = None,
    x_test: np.ndarray | None = None,
    test_labels: np.ndarray | None = None,
    label: str = "",
    num_classes: int = 0,
    class_names: list = None,
    model: nn.Module = None,
    processor=None,
    model_name: str = "",
    num_epochs: int = 30,
    batch_size: int = 1,
    logger: logging.Logger | None = None,
):
    log = logger.info if logger else print

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    train_dl = DataLoader(
        HFSegDataset(x_train, train_labels, processor, model_name=model_name),
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
    )

    val_dl = None
    if x_val is not None and val_labels is not None:
        val_dl = DataLoader(
            HFSegDataset(x_val, val_labels, processor, model_name=model_name),
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
        )

        # Create a best val_loss and best_model for running test:
        best_val_loss = float("inf")
        best_model_state = None

    test_dl = None
    if x_test is not None and test_labels is not None:
        test_dl = DataLoader(
            HFSegDataset(x_test, test_labels, processor, model_name=model_name),
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
        )

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)

    history = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
    }

    train_step_loss = []

    if num_epochs > 0: # Run test only by setting num_epochs to 0
        for epoch in range(num_epochs):
            model.train()
            tl, tc, tt = 0.0, 0, 0

            for batch in train_dl:
                batch = {
                    k: v.to(device) if isinstance(v, torch.Tensor) else v
                    for k, v in batch.items()
                }

                optimizer.zero_grad()
                outputs = model(**batch)
                loss = outputs.loss
                train_step_loss.append(loss.item())
                loss.backward()
                optimizer.step()

                target = batch.get("labels", batch.get("mask_labels"))
                if model_name == "mask2former":
                    preds = outputs.masks_queries_logits.argmax(1)
                elif model_name in {"segformer", "upernet"}:
                    preds = torch.nn.functional.interpolate(
                        outputs.logits,
                        size=target.shape[-2:],
                        mode="bilinear",
                        align_corners=False
                    ).argmax(1)
                else:
                    preds = outputs.logits.argmax(1) if hasattr(outputs, "logits") else outputs["out"].argmax(1)

                if target is not None:
                    valid = target != 255
                    tc += (preds[valid] == target[valid]).sum().item()
                    tt += valid.sum().item()

                tl += loss.item() * len(batch["pixel_values"])

            history["train_loss"].append(tl / len(x_train))
            history["train_acc"].append(tc / max(tt, 1))

            val_loss = None
            val_acc = None

            if val_dl is not None:
                model.eval()
                vl, vc, vt = 0.0, 0, 0

                with torch.no_grad():
                    for batch in val_dl:
                        batch = {
                            k: v.to(device) if isinstance(v, torch.Tensor) else v
                            for k, v in batch.items()
                        }

                        outputs = model(**batch)
                        vl += outputs.loss.item() * len(batch["pixel_values"])

                        target = batch.get("labels", batch.get("mask_labels"))
                        if model_name == "mask2former":
                            preds = outputs.masks_queries_logits.argmax(1)
                        elif model_name in {"segformer", "upernet"}:
                            preds = torch.nn.functional.interpolate(
                                outputs.logits,
                                size=target.shape[-2:],
                                mode="bilinear",
                                align_corners=False
                            ).argmax(1)
                        else:
                            preds = outputs.logits.argmax(1) if hasattr(outputs, "logits") else outputs["out"].argmax(1)

                        if target is not None:
                            valid = target != 255
                            vc += (preds[valid] == target[valid]).sum().item()
                            vt += valid.sum().item()

                val_loss = vl / len(x_val)
                val_acc = vc / max(vt, 1)

            history["val_loss"].append(val_loss)
            history["val_acc"].append(val_acc)

            msg = (
                f"[{label}] {epoch + 1}/{num_epochs} "
                f"loss={history['train_loss'][-1]:.3f} "
                f"train_acc={history['train_acc'][-1]:.3f}"
            )
            
            if val_acc is not None:
                msg += (
                    f" val_loss={val_loss:.3f}"
                    f" val_acc={val_acc:.3f}"
                )

                # Check if it has lowest val loss (run test on model with best validation loss)
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_model_state = {k: v.clone() for k, v in model.state_dict().items()}

            log(msg)

    preds = np.array([])
    gts = np.array([])
    bal_acc = None

    if test_dl is not None:
        # If validation included, run test on model with best validation loss
        if best_model_state is not None:
            model.load_state_dict(best_model_state)
        model.eval()
        all_preds = []
        all_gts = []

        with torch.no_grad():
            for batch in test_dl:
                batch = {
                    k: v.to(device) if isinstance(v, torch.Tensor) else v
                    for k, v in batch.items()
                }

                outputs = model(**batch)
                target = batch.get("labels", batch.get("mask_labels"))

                if model_name == "mask2former":
                    pred = outputs.masks_queries_logits.argmax(1)
                elif model_name in {"segformer", "upernet"}:
                    pred = torch.nn.functional.interpolate(
                        outputs.logits,
                        size=target.shape[-2:],
                        mode="bilinear",
                        align_corners=False
                    ).argmax(1)
                else:
                    pred = outputs.logits.argmax(1) if hasattr(outputs, "logits") else outputs["out"].argmax(1)

                if target is None:
                    continue

                valid = target != 255
                all_preds.extend(pred[valid].cpu().numpy())
                all_gts.extend(target[valid].cpu().numpy())

        preds = np.array(all_preds)
        gts = np.array(all_gts)

        test_acc = (preds == gts).mean()
        bal_acc = balanced_accuracy_score(gts, preds)

        log(f"\n[{label}] Test accuracy: {test_acc:.4f}")
        log(f"[{label}] Test balanced accuracy: {bal_acc:.4f}")
        log(classification_report(
                gts, preds,
                labels=list(range(num_classes)),
                target_names=class_names,
                zero_division=0
            ))

    # Determine the training loss convergence (training steps to learn data)
    # and validation loss convergence (training steps to generalize) if available
    train_convergence, val_convergence = iterations_to_converge(train_step_loss, val_losses=history["val_loss"])

    if train_convergence is not None:
        log(f"With sample size {len(x_train)}, train loss converged at iteration {train_convergence + 1}")
    else:
        log(f"With sample size {len(x_train)}, train loss did not clearly converge within {num_epochs} epochs")

    if val_convergence is not None:
        log(f"With sample size {len(x_train)}, val loss converged at epoch {val_convergence + 1}")
    else:
        log(f"With sample size {len(x_train)}, val loss did not clearly converge within {num_epochs} epochs")

    return model, history, preds, gts, bal_acc


# Training scripts for phase_classifier vs. segmentation
def train_phase_classifier(
    x_train, train_labels, x_val, val_labels,
    x_test, test_labels,
    label, num_classes, class_names, num_epochs=30, batch_size=1, logger=None,
):
    model = LightCNN(num_classes)
    return train_loop(
        x_train, train_labels, x_val, val_labels,
        x_test, test_labels,
        label, num_classes, class_names, model, num_epochs, batch_size, logger,
    )

def train_segmentation_model(
    x_train, train_labels, x_val, val_labels,
    x_test, test_labels,
    label, num_classes, class_names, model_name=None, num_epochs=30, batch_size=1, logger=None,
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
            x_test, test_labels,
            label, num_classes, class_names, model, processor, model_name, num_epochs, batch_size, logger,
        )
    
    return train_loop(
        x_train, train_labels, x_val, val_labels,
        x_test, test_labels,
        label, num_classes, class_names, model, num_epochs, batch_size, logger,
    )

# Plot utilities
def plot_training_curves(hist_div: dict, hist_rand: dict, output_dir: str, model_name: str | None = None,) -> None:
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
    if model_name is not None:
        fig.suptitle(f"{model_name} Training curves — diverse vs random", fontsize=13)
    else:
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
    model_name: str | None = None,
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
    if model_name is not None:
        fig.suptitle(f"{model_name} Normalised confusion matrix", fontsize=13)
    else:
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
    model_name: str | None = None,
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
    if model_name is not None:
        ax.set_title(f"{model_name} Per-class F1 — diverse vs random")
    else:
        ax.set_title("Per-class F1 — diverse vs random")
    ax.legend()
    plt.tight_layout()
    plt.savefig(f"{output_dir}/per_class_f1.png", dpi=150)
    plt.show()


def plot_balanced_accuracy(
    bal_div: float, 
    bal_rand: float, 
    output_dir: str,
    model_name: str | None = None,
) -> None:
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

# Helper function to make sure that pitvis classes are mapped properly
def remap(col: np.ndarray) -> np.ndarray:
    return np.array([class_map[s] for s in col], dtype=np.int64)

# Main function to run the entire evaluation pipeline
def main(
    task: str,
    dataset_root: str,
    dataset_style: str = "cholec",
    output_dir: str = "outputs",
    sampling_dir: str | None = None, # If using function outside main.py, allow user to save to user specific location
    # Pass in this if you want to load data from scratch or using pitvis
    train_root: str | None = None,
    val_root: str | None = None,
    test_root: str | None = None,
    # Pass in the list of videos for pitvis
    train_videos: list | None = None,
    val_videos: list | None = None,
    test_videos: list | None = None,
    # Training specific parameters
    model_name: str | None = None,
    num_classes: int | None = None,
    num_epochs: int = 30,
    batch_size: int = 1,
    # If using the function after diversity + random sampling, allow user to pass frames/indices as numpy
    all_emb: np.ndarray | None = None,
    div_frames: np.ndarray | None = None,
    div_masks: np.ndarray | None = None,
    div_indices: np.ndarray | None = None,
    rand_frames: np.ndarray | None = None,
    rand_masks: np.ndarray | None = None,
    rand_indices: np.ndarray | None = None,
    # Validation / Test Frames
    val_frames: np.ndarray | None = None,
    val_masks: np.ndarray | None = None,
    test_frames: np.ndarray | None = None,
    test_masks: np.ndarray | None = None,
    # Targets
    color_map: dict | None = None,
    all_labels_train: np.ndarray | None = None,
) -> None:

    training_dir = os.path.join(output_dir, "training_comparison")
    coverage_dir = os.path.join(output_dir, "data_coverage")
    os.makedirs(training_dir, exist_ok=True)
    os.makedirs(coverage_dir, exist_ok=True)

    logger = setup_logger(output_dir)
    logger.info(f"Task: {task} | Dataset: {dataset_style} | Model: {model_name}")
    logger.info(f"Dataset root: {dataset_root}")

    # Initialize different evaluation data
    frames_train  = None
    frames_val    = None
    frames_test   = None
    masks_train   = None
    masks_val     = None
    masks_test    = None
    labels_train  = None
    labels_val    = None
    labels_test   = None
    class_names   = []

    # Assign values based on dataset_style, pitvis = phase class and everything else classification
    if dataset_style == "pitvis":
        if div_frames is None:
            _train_root = train_root if train_root is not None else dataset_root
            frames_train, _, color_map, labels_train_raw, _ = dm.load_frames_and_masks(
                data_folder=_train_root,
                videos=[str(v) for v in train_videos] if train_videos else None,
                dataset_style="pitvis",
            )

        if val_videos is not None and len(val_videos) > 0:
            _val_root = val_root if val_root is not None else dataset_root
            frames_val, _, _, labels_val_raw, _ = dm.load_frames_and_masks(
                data_folder=_val_root,
                videos=[str(v) for v in val_videos],
                dataset_style="pitvis",
            )

        if test_videos is not None and len(test_videos) > 0:
            _test_root = test_root if test_root is not None else dataset_root
            frames_test, _, _, labels_test_raw, _ = dm.load_frames_and_masks(
                data_folder=_test_root,
                videos=[str(v) for v in test_videos],
                dataset_style="pitvis",
            )

        step_names = (
            pd.read_csv(f"{dataset_root}/map_steps.csv")
            .drop_duplicates("int_step")
            .set_index("int_step")["str_step"]
            .to_dict()
        )

        label_arrays = []
        if all_labels_train is not None:
            label_arrays.append(all_labels_train)
        elif frames_train is not None:
            label_arrays.append(labels_train_raw[:, 0])
        if frames_val is not None:
            label_arrays.append(labels_val_raw[:, 0])
        if frames_test is not None:
            label_arrays.append(labels_test_raw[:, 0])

        all_steps = np.unique(np.concatenate(label_arrays))
        class_map = {s: i for i, s in enumerate(all_steps)}
        inferred_classes = len(class_map)
        num_classes = num_classes if num_classes is not None else inferred_classes
        class_names = [step_names.get(s, str(s)).strip() for s in all_steps]

        if div_frames is None and frames_train is not None:
            all_labels_train = remap(labels_train_raw[:, 0])
        elif all_labels_train is not None:
            all_labels_train = remap(all_labels_train)
        if frames_val is not None:
            labels_val = remap(labels_val_raw[:, 0])
        if frames_test is not None:
            labels_test = remap(labels_test_raw[:, 0])

    else: # cholec, nested, flat tasks for segmentation
        if div_frames is None:
            _train_root = train_root if train_root is not None else dataset_root
            frames_train, masks_train, color_map, frame_metadata = dm.load_frames_and_masks(
                data_folder=_train_root,
                videos=[str(v) for v in train_videos] if train_videos else None,
                dataset_style=dataset_style,
            )
            all_labels_train = masks_train.reshape(len(masks_train), -1)[:, 0].astype(np.int64)
        
        if color_map is not None:
            num_classes = num_classes if num_classes is not None else len(color_map)
        elif num_classes is None:
            raise ValueError(
                "num_classes could not be inferred — either pass num_classes explicitly "
                "or ensure color_map is available."
            )

        # Default to val_root if pass frames are not available
        if val_frames is not None:
            frames_val = val_frames
            masks_val  = val_masks
            labels_val = masks_val.reshape(len(masks_val), -1)[:, 0].astype(np.int64) if masks_val is not None else None
        elif val_root is not None:
            frames_val, masks_val, _, _ = dm.load_frames_and_masks(
                data_folder=val_root,
                videos=[str(v) for v in val_videos] if val_videos else None,
                dataset_style=dataset_style,
                global_color_map=color_map,
            )
            labels_val = masks_val.reshape(len(masks_val), -1)[:, 0].astype(np.int64)

        if test_frames is not None:
            frames_test = test_frames
            masks_test  = test_masks
            labels_test = masks_test.reshape(len(masks_test), -1)[:, 0].astype(np.int64) if masks_test is not None else None
        elif test_root is not None:
            frames_test, masks_test, _, _ = dm.load_frames_and_masks(
                data_folder=test_root,
                videos=[str(v) for v in test_videos] if test_videos else None,
                dataset_style=dataset_style,
                global_color_map=color_map,
            )
            labels_test = masks_test.reshape(len(masks_test), -1)[:, 0].astype(np.int64)

        class_names = [str(i) for i in range(num_classes)] if num_classes is not None else []

        class_names = [str(i) for i in range(num_classes)] if num_classes is not None else []

    logger.info(f"Using {num_classes} classes: {class_names}")

    if div_frames is None:
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

        # Set the sampling directory if user wants to save sampling outputs elsewhere
        _sampling_dir = sampling_dir if sampling_dir is not None else os.path.join(output_dir, "diversity")
        os.makedirs(_sampling_dir, exist_ok=True)

        _, div_frames, div_masks, div_indices, _ = sampler.sample(
            data_arr=all_frames,
            mask_arr=masks_train if task == "segmentation" else None,
            emb_prev=all_emb,
            method="kmeans_sil",
            run_eval=True,
            save_data=True,
            data_dir=_sampling_dir,
        )

        div_indices  = np.array(div_indices)
        rand_indices = np.random.choice(len(all_frames), size=len(div_indices), replace=False)
        rand_frames  = all_frames[rand_indices]
        rand_masks   = masks_train[rand_indices] if task == "segmentation" and masks_train is not None else None

        if all_labels_train is not None:
            rand_labels = all_labels_train[rand_indices]
            div_labels  = all_labels_train[div_indices]
        else:
            rand_labels = None
            div_labels  = None

        sampler.evaluate_vs_other(
            random_indices=rand_indices,
            all_emb=all_emb,
            diverse_indices=div_indices,
            save_dir=coverage_dir,
            save_plot=True,
            logger=logger,
        )
    else:
        if all_labels_train is not None:
            div_labels  = all_labels_train[div_indices] if div_indices is not None else None
            rand_labels = all_labels_train[rand_indices] if rand_indices is not None else None
        else:
            div_labels  = None
            rand_labels = None

        if all_emb is not None and rand_indices is not None and div_indices is not None:
            sampler = DiversitySampler(
                emb_model="openclip",
                method="kmeans_sil",
                viz_clusters=False,
                plot_chosen_frames=False,
                save_plots=False,
            )
            sampler.evaluate_vs_other(
                random_indices=rand_indices,
                all_emb=all_emb,
                diverse_indices=div_indices,
                save_dir=coverage_dir,
                save_plot=True,
                logger=logger,
            )

    x_div  = div_frames
    x_rand = rand_frames

    logger.info(f"x_div: {x_div.shape}  x_rand: {x_rand.shape}")
    if frames_val is not None:
        logger.info(f"x_val: {frames_val.shape}")
    if frames_test is not None:
        logger.info(f"x_test: {frames_test.shape}")

    if task == "phase_classification":
        y_div  = div_labels
        y_rand = rand_labels
        y_val  = labels_val
        y_test = labels_test
        x_val  = frames_val
        x_test = frames_test

        logger.info("Training on diverse dataset...")
        model_div, hist_div, preds_div, gts_div, bal_div = train_phase_classifier(
            x_div, y_div, x_val, y_val, x_test, y_test,
            "diverse", num_classes, class_names, num_epochs, batch_size, logger)

        logger.info("Training on random dataset...")
        model_rand, hist_rand, preds_rand, gts_rand, bal_rand = train_phase_classifier(
            x_rand, y_rand, x_val, y_val, x_test, y_test,
            "random", num_classes, class_names, num_epochs, batch_size, logger)

    elif task == "segmentation":
        if div_masks is None or rand_masks is None:
            raise ValueError(
                "Segmentation task requires spatial masks but div_masks or rand_masks is None. "
                "Make sure annotated=true and masks were loaded correctly."
            )
            
        y_div  = div_masks
        y_rand = rand_masks
        y_val  = masks_val
        y_test = masks_test
        x_val  = frames_val
        x_test = frames_test

        logger.info("Training on diverse dataset...")
        model_div, hist_div, preds_div, gts_div, bal_div = train_segmentation_model(
            x_div, y_div, x_val, y_val, x_test, y_test,
            "diverse", num_classes, class_names, model_name, num_epochs, batch_size, logger)

        logger.info("Training on random dataset...")
        model_rand, hist_rand, preds_rand, gts_rand, bal_rand = train_segmentation_model(
            x_rand, y_rand, x_val, y_val, x_test, y_test,
            "random", num_classes, class_names, model_name, num_epochs, batch_size, logger)

    else:
        raise ValueError(f"Unknown task: {task!r}, choose from 'phase_classification', 'segmentation'")

    plot_training_curves(hist_div, hist_rand, training_dir, model_name=model_name)
    plot_confusion_matrices(preds_div, gts_div, preds_rand, gts_rand, bal_div, bal_rand, num_classes, class_names, training_dir, model_name=model_name)
    plot_per_class_f1(preds_div, gts_div, preds_rand, gts_rand, bal_div, bal_rand, num_classes, class_names, training_dir, model_name=model_name)
    plot_balanced_accuracy(bal_div, bal_rand, training_dir, model_name=model_name)

    logger.info(f"\nBalanced accuracy — diverse: {bal_div:.4f}  |  random: {bal_rand:.4f}")

if __name__ == "__main__":
    main(
        task="phase_classification",
        dataset_root="path/to/dataset",
        dataset_style="pitvis",
        train_videos=[1, 2, 3],
        val_videos=4,
    )