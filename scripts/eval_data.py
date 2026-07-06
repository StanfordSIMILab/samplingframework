# eval_data.py
# Evaluation utilities for determining effectiveness of diversity sampling vs. random sampling
import os
import cv2
import logging
import warnings

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

# Different Hugging Face model IDs for segmentation models
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
        return img, torch.tensor(self.labels[idx], dtype=torch.long)

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

# Remap labels to a contiguous range starting from 0, ramapping negative values to fit (0, num_classes - 1]
def build_label_shift(*label_arrays) -> dict:
    """
    Using the train/val/test label arrays, build a mapping from the original label values 
    (can contain negative values) to a contiguous range starting from 0.
    """
    all_present = set()
    for arr in label_arrays:
        if arr is None:
            continue
        arr = np.asarray(arr)
        all_present.update(int(v) for v in np.unique(arr[arr >= 0]))
    return {v: i for i, v in enumerate(sorted(all_present))}

def remap_labels_for_training(labels: np.ndarray, num_classes: int, shift: dict | None = None) -> np.ndarray:
    """
        Compact non-negative label values into a dense 0..k-1 range and map every
        negative value to num_classes (the appended 'unlabeled/transition' class).
        If `shift` is not provided, it's derived from this array alone (backward
        compatible for single-array callers)
    """
    remapped = labels.copy()
    positive_mask = remapped >= 0
    if shift is None:
        present = np.unique(remapped[positive_mask])
        shift = {int(v): i for i, v in enumerate(present)}
    remapped[positive_mask] = np.vectorize(shift.get)(remapped[positive_mask])
    remapped[~positive_mask] = num_classes
    return remapped


def remap_labels_for_eval(labels: np.ndarray, num_classes: int) -> np.ndarray:
    """Remap num_classes back to original mapping if negative values are used for evaluation."""
    remapped = labels.copy()
    remapped[remapped == num_classes] = -1
    return remapped

def detect_negative_labels(*label_arrays) -> bool:
    """
    Return True if any given label/mask array contains the -1 sentinel used to
    mark an unannotated 'transition' frame (pitvis) or a 'background' pixel
    (segmentation).
    """
    found = False
    for arr in label_arrays:
        if arr is None:
            continue
        arr = np.asarray(arr)
        if np.issubdtype(arr.dtype, np.unsignedinteger) and (arr == 255).any():
            warnings.warn(
                f"Label/mask array has unsigned dtype {arr.dtype} and contains the "
                "value 255. If this is meant to be the -1 'transition/background' "
                "sentinel that underflowed during storage (uint8 can't represent -1), "
                "re-save labels with a signed dtype (e.g. int64) upstream in "
                "data_loader.py. Otherwise this will silently be treated as an "
                "ordinary ignore/padding pixel rather than a class the model learns "
                "to predict.",
                stacklevel=2,
            )
        if (arr < 0).any():
            found = True
    return found

def validate_labels(labels: np.ndarray | None, effective_num_classes: int, label: str = "") -> None:
    """
    Raise a clear error if any label falls outside [0, effective_num_classes)
    after remapping.
    """
    if labels is None:
        return
    arr = np.asarray(labels)
    bad_mask = (arr < 0) | (arr >= effective_num_classes)
    if bad_mask.any():
        bad_values = np.unique(arr[bad_mask])
        raise ValueError(
            f"[{label}] found label value(s) {bad_values.tolist()} outside the valid "
            f"range [0, {effective_num_classes}) — dtype is {arr.dtype}. This usually "
            "means either the -1 'transition/background' sentinel didn't survive as "
            "-1 (check for an unsigned dtype upstream in data_loader.py), or "
            "num_classes doesn't match the actual label encoding."
        )

# Phase Classifier Models:
class LightCNN(nn.Module):
    def __init__(self, num_classes: int, dropout: float = 0.4):
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
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

class PretrainedPhaseClassifier(nn.Module):
    def __init__(self, num_classes: int, freeze_backbone: bool = True):
        super().__init__()
        import torchvision
        backbone = torchvision.models.resnet18(weights="IMAGENET1K_V1")
        if freeze_backbone:
            for p in backbone.parameters():
                p.requires_grad_(False)
        in_features = backbone.fc.in_features
        backbone.fc = nn.Linear(in_features, num_classes)
        self.net = backbone

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
def iterations_to_converge(losses: list, min_delta: float = 0.005, smooth_window: int = 5) -> int | None:
    # If empty validation or small number of losses, return None
    if losses is None or len(losses) < smooth_window + 1:
        return None
    
    def find_convergence(data):
        cleaned = [v for v in data if v is not None]
        if len(cleaned) < smooth_window + 1:
            return None
        
        # smooth to remove spike artifacts
        smoothed = np.convolve(cleaned, np.ones(smooth_window) / smooth_window, mode="valid")
        
        # find last epoch where smoothed loss improved by more than min_delta
        best = smoothed[0]
        last_improvement = 0
        for i, v in enumerate(smoothed):
            if best - v > min_delta:
                best = v
                last_improvement = i
        
        # offset back to original index due to convolution shrinkage
        return last_improvement + smooth_window // 2

    iterations_to_convergence = find_convergence(losses)
    return iterations_to_convergence

# Function to compute class weights from frequency distribution
def compute_class_weights(train_labels: np.ndarray, num_classes: int, device) -> torch.Tensor:
    if train_labels.ndim > 1:
        flat = train_labels.flatten()
    else:
        flat = train_labels

    flat = flat[flat != 255]  # ignore padding

    counts = np.bincount(flat.astype(np.int64), minlength=num_classes).astype(float)

    weights = 1.0 / (counts + 1)
    weights = weights / weights.sum() * num_classes  # normalize

    return torch.tensor(weights, dtype=torch.float32).to(device)

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
    effective_num_classes: int = 0, # num_classes + 1 if has_negative_label
    class_names: list = None,
    has_negative_label: bool = False,
    model: nn.Module = None,
    num_epochs: int = 30,
    batch_size: int = 1,
    logger: logging.Logger | None = None,
):
    log = logger.info if logger else print

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # If negative labels (-1, -2, ...) are present, remap them to num_classes for training
    shift = None
    if has_negative_label:
        shift = build_label_shift(train_labels, val_labels, test_labels)
        train_labels = remap_labels_for_training(train_labels, num_classes, shift=shift)
        if val_labels is not None:
            val_labels = remap_labels_for_training(val_labels, num_classes, shift=shift)
        if test_labels is not None:
            test_labels = remap_labels_for_training(test_labels, num_classes, shift=shift)

    # Use validate labels to ensure that all labels are within the expected range
    validate_labels(train_labels, effective_num_classes, label=f"{label} train")
    validate_labels(val_labels, effective_num_classes, label=f"{label} val")
    validate_labels(test_labels, effective_num_classes, label=f"{label} test")

    weights = compute_class_weights(train_labels, effective_num_classes, device)
    crit = nn.CrossEntropyLoss(weight=weights)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)

    train_dl = DataLoader(
        SurgicalDataset(x_train, train_labels),
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
    )

    best_val_loss = float("inf")
    best_model_state = None 

    val_dl = None
    if x_val is not None and val_labels is not None:
        val_dl = DataLoader(
            SurgicalDataset(x_val, val_labels),
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
        )

    test_dl = None
    if x_test is not None and test_labels is not None:
        test_dl = DataLoader(
            SurgicalDataset(x_test, test_labels),
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
        )

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

        if has_negative_label:
            preds = remap_labels_for_eval(preds, num_classes)
            gts = remap_labels_for_eval(gts, num_classes)
            report_labels = list(range(-1, num_classes))
            report_names = [class_names[num_classes]] + class_names[:num_classes]
        else:
            report_labels = list(range(num_classes))
            report_names = class_names

        test_acc = (preds == gts).mean()
        bal_acc = balanced_accuracy_score(gts, preds)

        log(f"\n[{label}] Test accuracy: {test_acc:.4f}")
        log(f"[{label}] Test balanced accuracy: {bal_acc:.4f}")
        log(classification_report(
                gts, preds,
                labels=report_labels,
                target_names=report_names,
                zero_division=0
            ))
    
    # Determine the training loss convergence (training steps to learn data)
    # and validation loss convergence (training steps to generalize) if available
    train_convergence_steps = iterations_to_converge(train_step_loss, min_delta=0.01, smooth_window=100)
    val_convergence = iterations_to_converge(history["val_loss"], min_delta=0.005, smooth_window=5)

    if train_convergence_steps is not None:
        log(f"With sample size {len(x_train)}, train loss converged at iteration {train_convergence_steps + 1}")
    else:
        log(f"With sample size {len(x_train)}, train loss did not clearly converge within {num_epochs} epochs")

    if val_convergence is not None:
        log(f"With sample size {len(x_train)}, val loss converged at epoch {val_convergence + 1}\n")
    elif history["val_loss"][0] is not None:
        log(f"With sample size {len(x_train)}, val loss did not clearly converge within {num_epochs} epochs\n")
    else:
        log("\n")

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
    effective_num_classes: int = 0, # num_classes + 1 if has_negative_label
    class_names: list = None,
    has_negative_label: bool = False,
    model: nn.Module = None,
    processor=None,
    model_name: str = "",
    optimizer: torch.optim.Optimizer | None = None,
    num_epochs: int = 30,
    batch_size: int = 1,
    logger: logging.Logger | None = None,
):
    log = logger.info if logger else print

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    shift = None
    if has_negative_label:
        shift = build_label_shift(train_labels, val_labels, test_labels)
        train_labels = remap_labels_for_training(train_labels, num_classes, shift=shift)
        if val_labels is not None:
            val_labels = remap_labels_for_training(val_labels, num_classes, shift=shift)
        if test_labels is not None:
            test_labels = remap_labels_for_training(test_labels, num_classes, shift=shift)

    # Use validate labels function to ensure that all labels are within the expected range
    validate_labels(train_labels, effective_num_classes, label=f"{label} train")
    validate_labels(val_labels, effective_num_classes, label=f"{label} val")
    validate_labels(test_labels, effective_num_classes, label=f"{label} test")

    weights = compute_class_weights(train_labels, effective_num_classes, device)
    criterion = nn.CrossEntropyLoss(weight=weights, ignore_index=255)
    if optimizer is None:
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)

    train_dl = DataLoader(
        HFSegDataset(x_train, train_labels, processor, model_name=model_name),
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
    )

    best_val_loss = float("inf")
    best_model_state = None

    val_dl = None
    if x_val is not None and val_labels is not None:
        val_dl = DataLoader(
            HFSegDataset(x_val, val_labels, processor, model_name=model_name),
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
        )

    test_dl = None
    if x_test is not None and test_labels is not None:
        test_dl = DataLoader(
            HFSegDataset(x_test, test_labels, processor, model_name=model_name),
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
        )

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
                target = batch.get("labels", batch.get("mask_labels"))
                logits_upsampled = torch.nn.functional.interpolate(
                    outputs.logits,
                    size=target.shape[-2:],
                    mode="bilinear",
                    align_corners=False
                )
                loss = criterion(logits_upsampled, target.long())
                train_step_loss.append(loss.item())
                loss.backward()
                optimizer.step()

                target = batch.get("labels", batch.get("mask_labels"))
                if model_name == "mask2former":
                    preds = outputs.masks_queries_logits.argmax(1)
                elif model_name in {"segformer", "upernet"}:
                    preds = logits_upsampled.argmax(1)
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

        if has_negative_label:
            preds = remap_labels_for_eval(preds, num_classes)
            gts = remap_labels_for_eval(gts, num_classes)
            report_labels = list(range(-1, num_classes))
            report_names = [class_names[num_classes]] + class_names[:num_classes]
        else:
            report_labels = list(range(num_classes))
            report_names = class_names

        test_acc = (preds == gts).mean()
        bal_acc = balanced_accuracy_score(gts, preds)

        log(f"\n[{label}] Test accuracy: {test_acc:.4f}")
        log(f"[{label}] Test balanced accuracy: {bal_acc:.4f}")
        log(classification_report(
                gts, preds,
                labels=report_labels,
                target_names=report_names,
                zero_division=0
            ))

    # Determine the training loss convergence (training steps to learn data)
    # and validation loss convergence (training steps to generalize) if available
    train_convergence_steps = iterations_to_converge(train_step_loss, min_delta=0.01, smooth_window=100)
    val_convergence = iterations_to_converge(history["val_loss"], min_delta=0.005, smooth_window=5)

    if train_convergence_steps is not None:
        log(f"With sample size {len(x_train)}, train loss converged at iteration {train_convergence_steps + 1}")
    else:
        log(f"With sample size {len(x_train)}, train loss did not clearly converge within {num_epochs} epochs")

    if val_convergence is not None:
        log(f"With sample size {len(x_train)}, val loss converged at epoch {val_convergence + 1}\n")
    elif history["val_loss"][0] is not None:
        log(f"With sample size {len(x_train)}, val loss did not clearly converge within {num_epochs} epochs\n")
    else:
        log("\n")

    return model, history, preds, gts, bal_acc


# Training scripts for phase_classifier vs. segmentation
def train_phase_classifier(
    x_train, train_labels, x_val, val_labels,
    x_test, test_labels,
    label, num_classes, class_names, num_epochs=30, batch_size=1, logger=None,
    has_negative_label = None, negative_class_name = "transition"
):
    # Deal with -1 labels for transition frames (pitvis) by remapping to num_classes for training
    if has_negative_label is None:
        has_negative_label = detect_negative_labels(train_labels, val_labels, test_labels)
    effective_num_classes = num_classes + 1 if has_negative_label else num_classes
    if has_negative_label:
        class_names = list(class_names) + [negative_class_name]

    model = LightCNN(effective_num_classes)
    return train_loop(
        x_train=x_train, train_labels=train_labels,
        x_val=x_val, val_labels=val_labels,
        x_test=x_test, test_labels=test_labels,
        label=label, num_classes=num_classes,
        effective_num_classes=effective_num_classes,
        class_names=class_names, has_negative_label=has_negative_label,
        model=model, num_epochs=num_epochs, batch_size=batch_size, logger=logger,
    )

def train_segmentation_model(
    x_train, train_labels, x_val, val_labels,
    x_test, test_labels,
    label, num_classes, class_names, model_name=None, num_epochs=30, batch_size=1, logger=None,
    has_negative_label = None, negative_class_name = "background"
):
    # Deal with -1 labels for background pixels (segmentation) by remapping to num_classes for training
    if has_negative_label is None:
        has_negative_label = detect_negative_labels(train_labels, val_labels, test_labels)
    effective_num_classes = num_classes + 1 if has_negative_label else num_classes
    if has_negative_label:
        class_names = list(class_names) + [negative_class_name]

    processor, model = build_model(effective_num_classes, model_name=model_name)

    if model_name in {"mask2former", "segformer", "upernet"}:
        optimizer = None
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
            x_train=x_train, train_labels=train_labels, x_val=x_val, val_labels=val_labels,
            x_test=x_test, test_labels=test_labels,
            label=label, num_classes=num_classes, effective_num_classes=effective_num_classes,
            class_names=class_names, has_negative_label=has_negative_label, model=model, processor=processor,
            model_name=model_name, num_epochs=num_epochs, batch_size=batch_size, logger=logger,
            optimizer=optimizer,
        )
    
    # If return statement not used, then unsupported model used
    raise NotImplementedError(
        f"train_segmentation_model does not yet support model_name={model_name!r}. "
        "Only 'mask2former', 'segformer', and 'upernet' are wired up to a working "
        "training loop (train_loop_hf). 'deeplab'/'unet' need a dedicated spatial "
        "loop — one that resizes labels alongside images and unwraps dict outputs — "
        "before they can be enabled here."
    )

# Plot utilities
def plot_training_curves(hist_div: dict, hist_rand: dict, output_dir: str, model_name: str | None = None,) -> None:
    os.makedirs(output_dir, exist_ok=True)

    val_loss_div  = [v for v in hist_div["val_loss"]  if v is not None]
    val_loss_rand = [v for v in hist_rand["val_loss"] if v is not None]
    val_acc_div   = [v for v in hist_div["val_acc"]   if v is not None]
    val_acc_rand  = [v for v in hist_rand["val_acc"]  if v is not None]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    epochs = range(1, len(hist_div["train_loss"]) + 1)
    for ax, key, ylabel in [(axes[0], "loss", "Loss"), (axes[1], "acc", "Accuracy")]:
        ax.plot(epochs, hist_div[f"train_{key}"], color="steelblue", linestyle="--", label="diverse train")
        ax.plot(epochs, hist_rand[f"train_{key}"], color="darkorange", linestyle="--", label="random train")
        
        val_div  = [v for v in hist_div[f"val_{key}"]  if v is not None]
        val_rand = [v for v in hist_rand[f"val_{key}"] if v is not None]
        if val_div:
            ax.plot(range(1, len(val_div)+1),  val_div,  color="steelblue",  linestyle="-", label="diverse val")
        if val_rand:
            ax.plot(range(1, len(val_rand)+1), val_rand, color="darkorange", linestyle="-", label="random val")
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
    has_negative_label: bool = False,
    negative_class_name: str = "transition",
) -> None:
    os.makedirs(output_dir, exist_ok=True)

    # For proper confusion matrix, include the background/transition accuracy, even if labeled -1
    if has_negative_label:
        plot_labels = list(range(-1, num_classes))
        plot_names = [negative_class_name] + list(class_names[:num_classes])
    else:
        plot_labels = list(range(num_classes))
        plot_names = list(class_names[:num_classes])

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    for ax, preds, gts, label, bal in [
        (axes[0], preds_div, gts_div, "diverse", bal_div),
        (axes[1], preds_rand, gts_rand, "random", bal_rand),
    ]:
        cm = confusion_matrix(gts, preds, labels=plot_labels, normalize="true")
        im = ax.imshow(cm, cmap="Blues", vmin=0, vmax=1)
        ax.set_xticks(range(len(plot_labels)))
        ax.set_yticks(range(len(plot_labels)))
        ax.set_xticklabels(plot_names, rotation=45, ha="right", fontsize=7)
        ax.set_yticklabels(plot_names, fontsize=7)
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
    has_negative_label: bool = False,
    negative_class_name: str = "transition",
) -> None:
    os.makedirs(output_dir, exist_ok=True)

    # For proper F1 score, include the background/transition accuracy, even if labeled -1
    if has_negative_label:
        plot_labels = list(range(-1, num_classes))
        plot_names = [negative_class_name] + list(class_names[:num_classes])
    else:
        plot_labels = list(range(num_classes))
        plot_names = list(class_names[:num_classes])

    f1_div = f1_score(gts_div, preds_div, labels=plot_labels, average=None, zero_division=0)
    f1_rand = f1_score(gts_rand, preds_rand, labels=plot_labels, average=None, zero_division=0)
    x_pos = np.arange(len(plot_labels))
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.bar(x_pos - 0.2, f1_div, 0.4, color="steelblue", label=f"diverse  (bal acc={bal_div:.3f})")
    ax.bar(x_pos + 0.2, f1_rand, 0.4, color="darkorange", label=f"random   (bal acc={bal_rand:.3f})")
    ax.set_xticks(x_pos)
    ax.set_xticklabels(plot_names, rotation=45, ha="right", fontsize=8)
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

# Run diverse vs. random sampling at sample sizes of increments of 100
def run_data_efficiency_curve(
    task: str,
    dataset_style: str = "cholec",
    output_dir: str = "outputs",
    model_name: str | None = None,
    num_classes: int | None = None,
    num_epochs: int = 30,
    batch_size: int = 1,
    all_train_emb: np.ndarray | None = None,
    all_train_frames: np.ndarray | None = None,
    all_train_masks: np.ndarray | None = None,
    all_train_labels: np.ndarray | None = None,
    test_frames: np.ndarray | None = None,
    test_masks: np.ndarray | None = None,
    test_labels: np.ndarray | None = None,
    color_map: dict | None = None,
    class_names: list | None = None,
    sample_sizes: list | None = None,
    negative_label_name: str | None = None,
    # sampler parameters
    sampler: DiversitySampler | None = None,
    emb_model: str = "openclip",
    method: str = "kmeans_sil",
    spread_sampling: bool = True,
    reduce_dims: bool = True,
    n_components: int = 64,
) -> dict:

    if sample_sizes is None:
        sample_sizes = [100, 200, 300, 500, 750, 1000]

    if sampler is None:
        sampler = DiversitySampler(
            emb_model=emb_model,
            method=method,
            viz_clusters=False,
            plot_chosen_frames=False,
            save_plots=False,
            spread_sampling=spread_sampling,
            keep_interactive=False,
        )
    
    # Make sure that num_classes is determined either from color_map, all_labels, or passed explicitly
    if color_map is not None and num_classes is None:
        num_classes = len(color_map)

    if num_classes is None and all_train_labels is not None:
        valid_labels = all_train_labels[all_train_labels >= 0]
        num_classes = len(np.unique(valid_labels))

    if num_classes is None:
        raise ValueError(
            "num_classes could not be inferred — pass num_classes explicitly "
            "or ensure color_map or all_train_labels is available."
        )

    # If task is phase_classification, then the unlabeled class is "transition", else if segmentation, then the unlabeled class is "background"
    if negative_label_name is None:
        negative_label_name = "transition" if task == "phase_classification" else "background"
    if task == "phase_classification":
        has_negative_label = detect_negative_labels(all_train_labels, test_labels)
    else:
        has_negative_label = detect_negative_labels(all_train_masks, test_masks)

    # Make sure classes aren't empty
    if class_names is None or len(class_names) == 0:
        class_names = [str(i) for i in range(num_classes)]

    results = {}
    curve_dir = os.path.join(output_dir, "efficiency_curve")
    os.makedirs(curve_dir, exist_ok=True)

    logger = setup_logger(curve_dir)

    for n in sample_sizes:
        logger.info(f"\n{'='*50}")
        logger.info(f"Running n={n}...")

        # diversity sampling
        _, div_frames, div_masks, div_indices, _ = sampler.sample(
            data_arr=all_train_frames,
            mask_arr=all_train_masks if task == "segmentation" else None,
            emb_prev=all_train_emb,
            num_samples=n,
            method="kmeans_sil",
            n_components=n_components,
            reduce_dims=reduce_dims,
            run_eval=False,
            save_data=False,
        )
        div_indices = np.array(div_indices)
        rand_indices = np.random.choice(len(all_train_frames), size=n, replace=False)
        rand_frames = all_train_frames[rand_indices]
        rand_masks = all_train_masks[rand_indices] if all_train_masks is not None else None

        if all_train_labels is not None:
            if all_train_labels.ndim == 2:
                all_train_labels = all_train_labels[:, 0]
            div_labels  = all_train_labels[div_indices]
            rand_labels = all_train_labels[rand_indices]
        else:
            div_labels  = None
            rand_labels = None

        if task == "phase_classification":
            # Run diverse frames
            _, _, _, _, bal_div = train_phase_classifier(
                div_frames, div_labels, None, None, test_frames, test_labels,
                f"diverse_n{n}", num_classes, class_names, num_epochs,
                batch_size, logger,
                has_negative_label=has_negative_label, negative_class_name=negative_label_name)

            # Run random frames
            _, _, _, _, bal_rand = train_phase_classifier(
                rand_frames, rand_labels, None, None, test_frames, test_labels,
                f"random_n{n}", num_classes, class_names, num_epochs,
                batch_size, logger,
                has_negative_label=has_negative_label, negative_class_name=negative_label_name)

        elif task == "segmentation":
            # Run diverse frames
            _, _, _, _, bal_div = train_segmentation_model(
                div_frames, div_masks, None, None, test_frames, test_masks,
                f"diverse_n{n}", num_classes, class_names, model_name,
                num_epochs, batch_size, logger,
                has_negative_label=has_negative_label, negative_class_name=negative_label_name)

            # Run random frames
            _, _, _, _, bal_rand = train_segmentation_model(
                rand_frames, rand_masks, None, None, test_frames, test_masks,
                f"random_n{n}", num_classes, class_names, model_name,
                num_epochs, batch_size, logger,
                has_negative_label=has_negative_label, negative_class_name=negative_label_name)

        results[n] = {"diverse": bal_div, "random": bal_rand}
        logger.info(f"n={n} — diverse: {bal_div:.4f} | random: {bal_rand:.4f}")

    # plot the curve
    plot_efficiency_curve(results, curve_dir)
    return results

# Plot the efficiency curve of diverse vs. random sampling
def plot_efficiency_curve(results: dict, output_dir: str) -> None:
    ns = sorted(results.keys())
    div_accs  = [results[n]["diverse"] for n in ns]
    rand_accs = [results[n]["random"]  for n in ns]

    plt.figure(figsize=(8, 5))
    plt.plot(ns, div_accs,  marker="o", color="steelblue",   label="diverse")
    plt.plot(ns, rand_accs, marker="o", color="darkorange",  label="random")
    plt.xlabel("n samples")
    plt.ylabel("Balanced accuracy")
    plt.title("Data efficiency — diverse vs random")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "data_efficiency_curve.png"), dpi=150)
    plt.show()
    plt.close()

# Main function to run the entire evaluation pipeline
def main(
    task: str,
    dataset_style: str = "cholec",
    output_dir: str = "outputs",
    sampling_dir: str | None = None, # If using function outside main.py, allow user to save to user specific location
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
    train_labels: np.ndarray | None = None,
    val_labels: np.ndarray | None = None,
    test_labels: np.ndarray | None = None,
    class_names: list | None = None,
    negative_label_name: str | None = None,
    # Sampling parameters
    reduce_dims: bool = True,
    n_components: int = 64,
    spread_sampling: bool = True,
) -> None:

    training_dir = os.path.join(output_dir, "training_comparison")
    coverage_dir = os.path.join(output_dir, "data_coverage")
    os.makedirs(training_dir, exist_ok=True)
    os.makedirs(coverage_dir, exist_ok=True)

    logger = setup_logger(output_dir)
    logger.info(f"Task: {task} | Dataset: {dataset_style} | Model: {model_name}")

    # Assign labels for training, validation, and test sets based on available data
    if val_labels is not None:
        labels_val = val_labels
    elif val_masks is not None:
        labels_val = val_masks.reshape(len(val_masks), -1)[:, 0].astype(np.int64)
    else:
        labels_val = None

    if test_labels is not None:
        labels_test = test_labels
    elif test_masks is not None:
        labels_test = test_masks.reshape(len(test_masks), -1)[:, 0].astype(np.int64)
    else:
        labels_test = None

    if class_names is None:
        class_names = [str(i) for i in range(num_classes)] if num_classes is not None else []
    
    # Make sure that num_classes is determined either from color_map, all_labels_train, or passed explicitly
    if color_map is not None and num_classes is None:
        num_classes = len(color_map)

    if num_classes is None and train_labels is not None:
        valid_labels = train_labels[train_labels >= 0]
        num_classes = len(np.unique(valid_labels))
        logger.info(f"Inferred num_classes={num_classes} from {len(np.unique(valid_labels))} unique labels")

    if num_classes is None:
        raise ValueError(
            "num_classes could not be inferred — pass num_classes explicitly "
            "or ensure color_map or train_labels is available."
        )

    # Make sure classes aren't empty
    if class_names is None or len(class_names) == 0:
        class_names = [str(i) for i in range(num_classes)]

    frames_val = val_frames
    frames_test = test_frames
    masks_val = val_masks
    masks_test = test_masks

    logger.info(f"Using {num_classes} classes: {class_names}")

    # If div_frames not provided, raise an error for division of labor
    if div_frames is None or rand_frames is None:
        raise ValueError("eval_data.main() requires pre-computed div_frames/rand_frames.")

    if train_labels is not None:
        if train_labels.ndim == 2:
            train_labels = train_labels[:, 0]

        # guard against size mismatch between div vs rand
        max_idx = max(
            div_indices.max() if div_indices is not None else 0,
            rand_indices.max() if rand_indices is not None else 0
        )
        if max_idx >= len(train_labels):
            raise ValueError(
                f"Index {max_idx} out of bounds for train_labels of size {len(train_labels)}. "
                f"Ensure train_labels is sliced to the train split before passing."
            )

        # Passes guard, set div_labels and rand_labels accordingly
        div_labels  = train_labels[div_indices] if div_indices is not None else None
        rand_labels = train_labels[rand_indices] if rand_indices is not None else None
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

    # Check that the shapes of the datasets are compatible
    logger.info(f"x_div: {x_div.shape}  x_rand: {x_rand.shape}")
    if frames_val is not None:
        logger.info(f"x_val: {frames_val.shape}")
    if frames_test is not None:
        logger.info(f"x_test: {frames_test.shape}")

    # Set the negative-label class name based on the task (if not overridden by the caller)
    if negative_label_name is None:
        negative_label_name = "transition" if task == "phase_classification" else "background"

    if task == "phase_classification":
        y_div  = div_labels
        y_rand = rand_labels
        y_val  = labels_val
        y_test = labels_test
        x_val  = frames_val
        x_test = frames_test

        # Check if there are any negative labels (-1) in the training, validation, or test sets
        has_negative_label = detect_negative_labels(train_labels, labels_val, labels_test)

        logger.info("Training on diverse dataset...")
        model_div, hist_div, preds_div, gts_div, bal_div = train_phase_classifier(
            x_div, y_div, x_val, y_val, x_test, y_test,
            "diverse", num_classes, class_names, num_epochs, batch_size, logger,
            has_negative_label=has_negative_label, negative_class_name=negative_label_name)
 
        logger.info("Training on random dataset...")
        model_rand, hist_rand, preds_rand, gts_rand, bal_rand = train_phase_classifier(
            x_rand, y_rand, x_val, y_val, x_test, y_test,
            "random", num_classes, class_names, num_epochs, batch_size, logger,
            has_negative_label=has_negative_label, negative_class_name=negative_label_name)

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

        # Check if there are any negative labels (-1) in the training, validation, or test sets
        has_negative_label = detect_negative_labels(div_masks, rand_masks, masks_val, masks_test)

        logger.info("Training on diverse dataset...")
        model_div, hist_div, preds_div, gts_div, bal_div = train_segmentation_model(
            x_div, y_div, x_val, y_val, x_test, y_test,
            "diverse", num_classes, class_names, model_name, num_epochs, batch_size, logger,
            has_negative_label=has_negative_label, negative_class_name=negative_label_name)

        logger.info("Training on random dataset...")
        model_rand, hist_rand, preds_rand, gts_rand, bal_rand = train_segmentation_model(
            x_rand, y_rand, x_val, y_val, x_test, y_test,
            "random", num_classes, class_names, model_name, num_epochs, batch_size, logger,
            has_negative_label=has_negative_label, negative_class_name=negative_label_name)

    else:
        raise ValueError(f"Unknown task: {task!r}, choose from 'phase_classification', 'segmentation'")

    plot_training_curves(hist_div, hist_rand, training_dir, model_name=model_name)
    plot_confusion_matrices(
        preds_div, gts_div, preds_rand, gts_rand, bal_div, bal_rand, num_classes, class_names,
        training_dir, model_name=model_name,
        has_negative_label=has_negative_label, negative_class_name=negative_label_name)
    plot_per_class_f1(
        preds_div, gts_div, preds_rand, gts_rand, bal_div, bal_rand, num_classes, class_names,
        training_dir, model_name=model_name,
        has_negative_label=has_negative_label, negative_class_name=negative_label_name)
    plot_balanced_accuracy(bal_div, bal_rand, training_dir, model_name=model_name)

    logger.info(f"\nBalanced accuracy — diverse: {bal_div:.4f}  |  random: {bal_rand:.4f}")

if __name__ == "__main__":
    pass