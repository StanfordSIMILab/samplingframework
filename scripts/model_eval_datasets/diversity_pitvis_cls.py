import numpy as np
import pandas as pd
import cv2
import os
import sys
import threading
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics.pairwise import cosine_distances
from sklearn.preprocessing import normalize
from sklearn.decomposition import PCA
from sklearn.metrics import (
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
import umap
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

sys.path.insert(0, "scripts")
from diversity_sampling_main import DiversitySampling


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
PITVIS_DIR = "scripts/model_eval_datasets/neurosurg_data_videos/pitvis_v3"
OUTPUT_DIR = "output_pitvis"
TRAIN_VIDEOS = [3, 7]
VAL_VIDEO = 17


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_frames(video_path: str) -> np.ndarray:
    """Read every frame from *video_path* into an (N, H, W, 3) uint8 array."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open: {video_path}")
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames = []
    for i in range(total):
        result: dict = {}

        def _read(cap=cap, result=result):
            result["ret"], result["frame"] = cap.read()

        t = threading.Thread(target=_read)
        t.start()
        t.join(timeout=5.0)
        if t.is_alive():
            print(f"\nTimeout at {i}/{total}, stopping.")
            break
        if not result.get("ret", False):
            break
        frames.append(cv2.cvtColor(result["frame"], cv2.COLOR_BGR2RGB))
        print(f"{i + 1}/{total}", end="\r", flush=True)
    print()
    cap.release()
    return np.stack(frames, axis=0)


def load_annotations(video_ids: list) -> pd.DataFrame:
    dfs = [pd.read_csv(f"{PITVIS_DIR}/annotations_{v:02d}.csv") for v in video_ids]
    return pd.concat(dfs, ignore_index=True).set_index(["int_video", "int_time"])


def get_labels(annot_index: pd.DataFrame, meta_pairs: list) -> np.ndarray:
    rows = []
    for vid, fi in meta_pairs:
        key = (vid, fi)
        if key in annot_index.index:
            row = annot_index.loc[key][
                ["int_step", "int_instrument1", "int_instrument2"]
            ].tolist()
        else:
            row = [-1, -1, -2]
        rows.append(row)
    return np.array(rows, dtype=np.int32)


# ---------------------------------------------------------------------------
# FVI filtering
# ---------------------------------------------------------------------------

def compute_fvi(frames: np.ndarray) -> np.ndarray:
    return np.array(
        [
            np.mean(
                np.abs(
                    frames[i].astype(np.float32) - frames[i - 1].astype(np.float32)
                )
            )
            for i in range(1, len(frames))
        ]
    )


def fvi_filter(frames: np.ndarray, thresh: float):
    scores = compute_fvi(frames)
    kept = [j + 1 for j, s in enumerate(scores) if s > thresh]
    return frames[kept], kept, scores


def show_fvi_histogram(scores: np.ndarray, video_id: int) -> None:
    df = pd.DataFrame(scores, columns=["fvi"])
    mu, std = norm.fit(df["fvi"])
    sns.histplot(df, x="fvi", bins=20)
    x = np.linspace(df["fvi"].min(), df["fvi"].max(), 100)
    plt.plot(x, norm.pdf(x, mu, std) * len(df) * (x[1] - x[0]), "r--")
    for mult, color in [(0, "blue"), (1, "green"), (2, "orange")]:
        plt.axvline(mu + std * mult, color=color, linestyle="--")
        if mult > 0:
            plt.axvline(mu - std * mult, color=color, linestyle="--")
    plt.title(f"FVI distribution — video {video_id:02d}")
    plt.show()


# ---------------------------------------------------------------------------
# Coverage / diversity metrics
# ---------------------------------------------------------------------------

def hausdorff_coverage(X_full: np.ndarray, X_sub: np.ndarray):
    nn = NearestNeighbors(n_neighbors=1).fit(X_sub)
    dists = nn.kneighbors(X_full)[0].flatten()
    return dists.max(), dists.mean()


def intra_spread(X_sub: np.ndarray) -> float:
    D = cosine_distances(X_sub)
    np.fill_diagonal(D, np.nan)
    return float(np.nanmean(D))


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def plot_hausdorff_spread(
    full_n: np.ndarray,
    diverse_n: np.ndarray,
    random_n: np.ndarray,
    n_sampled: int,
) -> None:
    hd_div, mn_div = hausdorff_coverage(full_n, diverse_n)
    hd_ran, mn_ran = hausdorff_coverage(full_n, random_n)
    sp_div = intra_spread(diverse_n)
    sp_ran = intra_spread(random_n)
    print(f"Hausdorff  — diverse: {hd_div:.4f}  |  random: {hd_ran:.4f}")
    print(f"Mean NN    — diverse: {mn_div:.4f}  |  random: {mn_ran:.4f}")
    print(f"Spread     — diverse: {sp_div:.4f}  |  random: {sp_ran:.4f}")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(f"Diverse (n={n_sampled}) vs Random (n={n_sampled})", fontsize=13)
    for ax, vals, ylabel, title in [
        (
            axes[0],
            [hd_div, hd_ran],
            "Hausdorff distance",
            "Worst-case coverage (lower = better)",
        ),
        (
            axes[1],
            [sp_div, sp_ran],
            "Mean pairwise cosine dist",
            "Intra-subset spread (higher = less redundant)",
        ),
    ]:
        bars = ax.bar(
            ["diverse", "random"], vals, color=["steelblue", "darkorange"], width=0.4
        )
        ax.bar_label(bars, fmt="%.4f", padding=3)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.set_ylim(0, max(vals) * 1.2)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/coverage_redundancy.png", dpi=150)
    plt.show()


def plot_nn_coverage(
    full_n: np.ndarray, diverse_n: np.ndarray, random_n: np.ndarray
) -> None:
    for sub_n, label in [(diverse_n, "diverse"), (random_n, "random")]:
        nn = NearestNeighbors(n_neighbors=1).fit(sub_n)
        dists = nn.kneighbors(full_n)[0].flatten()
        eps = np.percentile(dists, 50)
        eps_grid = np.linspace(0, np.percentile(dists, 95), 100)

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        fig.suptitle(f"NN coverage — {label}")
        axes[0].hist(dists, bins=50, alpha=0.7)
        axes[0].axvline(eps, linestyle="--")
        axes[0].set_title(f"NN distance  |  Coverage={np.mean(dists < eps) * 100:.1f}%")
        axes[0].set_xlabel("Distance")
        axes[1].plot(eps_grid, [np.mean(dists < e) for e in eps_grid])
        axes[1].axvline(eps, linestyle="--")
        axes[1].set_title("Coverage vs ε")
        axes[1].set_xlabel("ε")
        axes[1].set_ylabel("Coverage")
        plt.tight_layout()
        plt.savefig(f"{OUTPUT_DIR}/nn_coverage_{label}.png", dpi=150)
        plt.show()


def plot_pca_scatter(
    full_n: np.ndarray, diverse_n: np.ndarray, random_n: np.ndarray
) -> None:
    pca = PCA(n_components=2)
    full_2d = pca.fit_transform(full_n)
    diverse_2d = pca.transform(diverse_n)
    random_2d = pca.transform(random_n)

    vmax = max(
        NearestNeighbors(n_neighbors=1).fit(diverse_n).kneighbors(full_n)[0].max(),
        NearestNeighbors(n_neighbors=1).fit(random_n).kneighbors(full_n)[0].max(),
    )
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    fig.suptitle("PCA coverage — green = covered, red = blind spot", fontsize=12)
    for ax, sub_2d, sub_n_emb, label in [
        (axes[0], diverse_2d, diverse_n, "diverse"),
        (axes[1], random_2d, random_n, "random"),
    ]:
        dists = (
            NearestNeighbors(n_neighbors=1)
            .fit(sub_n_emb)
            .kneighbors(full_n)[0]
            .flatten()
        )
        hd, mean_nn = hausdorff_coverage(full_n, sub_n_emb)
        sc = ax.scatter(
            full_2d[:, 0],
            full_2d[:, 1],
            c=dists,
            cmap="RdYlGn_r",
            s=4,
            alpha=0.7,
            vmin=0,
            vmax=vmax,
        )
        ax.scatter(
            sub_2d[:, 0],
            sub_2d[:, 1],
            c="black",
            s=25,
            marker="x",
            linewidths=0.8,
            label=f"{label} samples",
            zorder=5,
        )
        ax.set_title(
            f"{label} (n={len(sub_2d)})  |  Mean NN: {mean_nn:.4f}  |  HD: {hd:.4f}"
        )
        ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0] * 100:.1f}% var)")
        ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1] * 100:.1f}% var)")
        ax.legend(markerscale=2)
        plt.colorbar(sc, ax=ax, label="NN distance to subset")
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/pca_coverage_heatmap.png", dpi=150)
    plt.show()


def plot_umap(
    full_n: np.ndarray, diverse_indices: np.ndarray, random_indices: np.ndarray
) -> None:
    print("Fitting UMAP...")
    emb_2d = umap.UMAP(n_components=2, random_state=42).fit_transform(full_n)
    fig, axes = plt.subplots(1, 2, figsize=(18, 8))
    for ax, indices, title in [
        (axes[0], np.array(diverse_indices), "Diverse Samples"),
        (axes[1], np.array(random_indices), "Random Samples"),
    ]:
        mask = np.ones(len(full_n), dtype=bool)
        mask[indices] = False
        ax.scatter(
            emb_2d[mask, 0],
            emb_2d[mask, 1],
            s=4,
            alpha=0.4,
            color="steelblue",
            label="all frames",
        )
        ax.scatter(
            emb_2d[indices, 0],
            emb_2d[indices, 1],
            s=25,
            alpha=0.9,
            color="red",
            label=f"selected ({len(indices)})",
        )
        ax.set_title(title, fontsize=14)
        ax.set_xlabel("UMAP 1")
        ax.set_ylabel("UMAP 2")
        ax.legend(markerscale=2)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/umap_selected.png", dpi=150)
    plt.show()


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


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

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


def train_model(
    x_train: np.ndarray,
    train_labels: np.ndarray,
    x_val: np.ndarray,
    val_labels: np.ndarray,
    label: str,
    num_classes: int,
    class_names: list,
    num_epochs: int = 30,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

    model = LightCNN(num_classes).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    crit = nn.CrossEntropyLoss(weight=weights)

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


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    annot_index = load_annotations(TRAIN_VIDEOS + [VAL_VIDEO])

    # --- FVI filtering ---
    all_frames = []
    all_meta = []
    for vid in TRAIN_VIDEOS:
        print(f"\nLoading video {vid:02d}...")
        frames = load_frames(f"{PITVIS_DIR}/video_{vid:02d}.mp4")
        fvi_scores = compute_fvi(frames)
        show_fvi_histogram(fvi_scores, vid)
        thresh = float(input(f"FVI threshold for video {vid:02d}: "))
        filtered, kept_indices, _ = fvi_filter(frames, thresh)
        print(f"Kept {len(filtered)} / {len(frames)} frames")
        all_frames.append(filtered)
        all_meta.extend([(vid, fi) for fi in kept_indices])

    all_frames = np.concatenate(all_frames, axis=0)
    print(f"\nTotal train frames after FVI filtering: {len(all_frames)}")

    # --- Diversity sampling ---
    sampler = DiversitySampling(
        data_array=all_frames,
        optim_clusters=True,
        dino_model_string="dinov2_vits14",
        n_samples_per_cluster=5,
        viz_clusters=True,
        emb_prev=None,
        plot_chosen_frames=False,
        openclip_model_string="ViT-B-32",
        openclip_pretrained="laion2b_s34b_b79k",
        emb_model="openclip",
        save_path=f"{OUTPUT_DIR}/diverse_frames.npy",
    )
    _, n_clusters, n_per_cluster, diverse_indices = sampler.forward(
        eval=True,
        method="kmeans",
        export=OUTPUT_DIR,
        min_k=5,
        reduce_dims=True,
    )
    n_sampled = n_clusters * n_per_cluster
    random_indices = np.random.choice(len(all_frames), size=n_sampled, replace=False)

    # --- Labels ---
    x_div = all_frames[diverse_indices]
    y_div = get_labels(annot_index, [all_meta[i] for i in diverse_indices])
    x_rand = all_frames[random_indices]
    y_rand = get_labels(annot_index, [all_meta[i] for i in random_indices])

    print(f"\nLoading val video {VAL_VIDEO:02d}...")
    x_val = load_frames(f"{PITVIS_DIR}/video_{VAL_VIDEO:02d}.mp4")
    y_val = get_labels(annot_index, [(VAL_VIDEO, fi) for fi in range(len(x_val))])

    # --- Embedding-space coverage plots ---
    full_n = normalize(sampler.all_emb)
    diverse_n = normalize(sampler.all_emb[diverse_indices])
    random_n = normalize(sampler.all_emb[random_indices])

    plot_hausdorff_spread(full_n, diverse_n, random_n, n_sampled)
    plot_nn_coverage(full_n, diverse_n, random_n)
    plot_pca_scatter(full_n, diverse_n, random_n)
    plot_umap(full_n, diverse_indices, random_indices)

    # --- Class mapping ---
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

    labels_div = remap(y_div[:, 0])
    labels_rand = remap(y_rand[:, 0])
    labels_val = remap(y_val[:, 0])

    print(f"\nx_div: {x_div.shape}  y_div: {y_div.shape}")
    print(f"x_rand: {x_rand.shape}  y_rand: {y_rand.shape}")
    print(f"x_val: {x_val.shape}  y_val: {y_val.shape}")

    # --- Training ---
    print("\nTraining on diverse dataset...")
    model_div, hist_div, preds_div, gts_div, bal_div = train_model(
        x_div, labels_div, x_val, labels_val, "diverse", num_classes, class_names
    )

    print("\nTraining on random dataset...")
    model_rand, hist_rand, preds_rand, gts_rand, bal_rand = train_model(
        x_rand, labels_rand, x_val, labels_val, "random", num_classes, class_names
    )

    # --- Result plots ---
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
    main()