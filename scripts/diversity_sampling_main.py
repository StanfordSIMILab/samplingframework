from pathlib import Path
from typing import Union
import cv2
import tempfile, os
import re
import json
import threading

import numpy as np
import torch
import torchvision.transforms as T
import open_clip
from PIL import Image
import shutil
from kneed import KneeLocator

import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, DBSCAN, HDBSCAN
from sklearn.decomposition import PCA
from sklearn.metrics import pairwise_distances, davies_bouldin_score, silhouette_score

from skimage.metrics import structural_similarity as ssim
from sklearn.neighbors import NearestNeighbors
from scipy.spatial.distance import cdist
from scipy.stats import chi2

# Helper function for separating multiclass PNG style masks
def luminance(rgb):
    r, g, b = rgb
    return 0.2126 * r + 0.7152 * g + 0.0722 * b

# Dataset split function
def train_val_test_split(
    data_arr: np.ndarray,
    test_size: float = 0.1,
    val_size: float = 0.1,
    out_folder: str = ".",
    mask_arr: np.ndarray | None = None,
    shuffle: bool = True,
    seed: int = 42,
):
    n = len(data_arr)
    indices = np.arange(n)

    if shuffle:
        rng = np.random.default_rng(seed)
        rng.shuffle(indices)

    n_test = max(1, int(np.floor(n * test_size)))
    n_val  = max(1, int(np.floor(n * val_size)))
    n_train = n - n_test - n_val

    if n_train <= 0:
        raise ValueError(
            f"test_size={test_size} + val_size={val_size} leaves no training samples "
            f"for a dataset of {n} frames."
        )

    train_idx = indices[:n_train]
    val_idx   = indices[n_train : n_train + n_val]
    test_idx  = indices[n_train + n_val :]

    splits = {"train": train_idx, "val": val_idx, "test": test_idx}

    for split_name, idx in splits.items():
        split_dir = os.path.join(out_folder, split_name)
        os.makedirs(split_dir, exist_ok=True)

        frames_split = data_arr[idx]
        np.save(os.path.join(split_dir, "frames.npy"), frames_split)

        if mask_arr is not None:
            masks_split = mask_arr[idx]
            np.save(os.path.join(split_dir, "masks.npy"), masks_split)

        np.save(os.path.join(split_dir, "indices.npy"), idx)

    print(f"\nSplit summary: train={len(train_idx)}  val={len(val_idx)}  test={len(test_idx)}")
    return train_idx, val_idx, test_idx

# Extract masks based on different nestings
def parse_dataset_with_masks(
        dataset_root: str | os.PathLike = ".",
        mask_type: str = "color_mask",
        target_size: tuple[int, int] | None = (224, 224),
        videos: list[str] | None = None,
        dataset_style: str = "cholec",
        force_color_processing: bool = False,
        global_color_map: dict | None = None,
    ) -> tuple[np.ndarray, np.ndarray, dict]:

    dataset_root = Path(dataset_root)
    if not dataset_root.exists():
        raise FileNotFoundError(f"Dataset root not found: {dataset_root}")

    load_as_rgb = mask_type == "color_mask" or force_color_processing
    mask_mode = "RGB" if load_as_rgb else "L"
    excluded = {"masks", "annotations", "mask"}

    frame_paths = []

    if dataset_style == "cholec":
        video_dirs = sorted(
            d for d in dataset_root.iterdir()
            if d.is_dir() and (videos is None or d.name in videos)
        )
        if not video_dirs:
            raise FileNotFoundError(f"No video folders found under {dataset_root}")
        for video_dir in video_dirs:
            print(f"Scanning {video_dir.name}...")
            for sample_dir in sorted(video_dir.iterdir()):
                if sample_dir.is_dir():
                    for f in sorted(sample_dir.iterdir()):
                        if any(exc in part for part in f.parts for exc in excluded):
                            continue
                        if f.suffix.lower() in {".jpg", ".jpeg", ".png"}:
                            frame_paths.append(f)

    elif dataset_style == "flat":
        for f in sorted(dataset_root.iterdir()):
            if any(exc in part for part in f.parts for exc in excluded):
                continue
            if f.suffix.lower() in {".jpg", ".jpeg", ".png"}:
                frame_paths.append(f)

    elif dataset_style == "nested":
        subdirs = sorted(
            d for d in dataset_root.iterdir()
            if d.is_dir() and (videos is None or d.name in videos)
            and not any(exc in d.name.lower() for exc in excluded)
        )
        if not subdirs:
            raise FileNotFoundError(f"No subfolders found under {dataset_root}")
        for subdir in subdirs:
            print(f"Scanning {subdir.name}...")
            for f in sorted(subdir.iterdir()):
                if any(exc in part for part in f.parts for exc in excluded):
                    continue
                if f.suffix.lower() in {".jpg", ".jpeg", ".png"}:
                    frame_paths.append(f)

    else:
        raise ValueError(
            f"Unknown dataset_style '{dataset_style}'. Choose 'cholec', 'flat', or 'nested'."
        )

    if not frame_paths:
        raise FileNotFoundError(
            f"No image frames found under {dataset_root} with style='{dataset_style}'"
        )

    print(f"Found {len(frame_paths)} frames.")

    if target_size is not None:
        h, w = target_size
    else:
        h, w = Image.open(frame_paths[0]).size[::-1]

    n = len(frame_paths)
    frames = np.empty((n, h, w, 3), dtype=np.uint8)
    masks  = np.empty((n, h, w, 3 if load_as_rgb else 1), dtype=np.uint8)

    missing_masks = []

    for i, frame_path in enumerate(frame_paths):
        print(f"Loading frame {i+1}/{n}", end="\r", flush=True)

        mask_name = f"{frame_path.stem}_{mask_type}.png"
        mask_path = frame_path.parent / mask_name

        if not mask_path.exists():
            missing_masks.append(str(mask_path))
            continue

        frame_img = Image.open(frame_path).convert("RGB")
        mask_img  = Image.open(mask_path).convert(mask_mode)

        if target_size is not None:
            frame_img = frame_img.resize((w, h), Image.BILINEAR)
            mask_img  = mask_img.resize((w, h), Image.NEAREST)

        frames[i] = np.array(frame_img, dtype=np.uint8)
        mask_np   = np.array(mask_img, dtype=np.uint8)
        masks[i]  = mask_np if mask_np.ndim == 3 else mask_np[..., np.newaxis]

    if missing_masks:
        raise FileNotFoundError(
            f"{len(missing_masks)} mask(s) not found. First missing:\n  {missing_masks[0]}"
        )

    print(f"\nFinished loading {n} frames.")

    color_map = {}
    if load_as_rgb:
        if global_color_map is None:
            N, H, W, _ = masks.shape
            flat = masks.reshape(-1, 3)
            unique_colors = np.unique(flat, axis=0)
            unique_colors = sorted(unique_colors, key=luminance)
            global_color_map = {
                idx: list(color)
                for idx, color in enumerate(unique_colors)
            }

        rgb_to_cls = {tuple(color): idx for idx, color in global_color_map.items()}
        N, H, W, _ = masks.shape
        flat = masks.reshape(-1, 3)
        flat_out = np.zeros(flat.shape[0], dtype=np.uint8)
        for rgb_tuple, cls in rgb_to_cls.items():
            flat_out[np.all(flat == np.array(rgb_tuple), axis=1)] = cls
        masks = flat_out.reshape(N, H, W)
        color_map = global_color_map
        print(f"Finished processing masks. Unique classes found: {len(color_map)}")
    else:
        masks = masks.squeeze(-1)
        unique_vals = np.unique(masks)
        print(f"Finished processing masks. Unique values found: {unique_vals}")

    return frames, masks, color_map


class DiversitySampler:
    def __init__(
        self,
        data_array=None,
        n_samples_per_cluster=250,
        optim_clusters=True,
        viz_clusters=True,
        plot_chosen_frames=True,
        emb_model=None,
        method=None,
        dino_model_string="dinov2_vits14",
        openclip_model_string="ViT-B-32",
        openclip_pretrained="openai",
        save_path="data_hooray.npy",
        proportional_sampling=True,
        emb_prev=None,
        keep_interactive=False
    ):
        self.data_array = data_array
        self.n_samples_per_cluster = n_samples_per_cluster
        self.optimize_clusters = optim_clusters
        self.viz_clusters = viz_clusters
        self.plot_chosen_frames = plot_chosen_frames
        self.emb_model = emb_model
        self.method = method
        self.dino_model_string = dino_model_string
        self.openclip_model_string = openclip_model_string
        self.openclip_pretrained = openclip_pretrained
        self.save_path = save_path
        self.proportional_sampling = proportional_sampling
        self.emb_prev = emb_prev
        self.keep_interactive = keep_interactive

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if self.emb_model is None:
            raise ValueError(
                "Please provide a correct embedding model for clustering, choose 'dino' or 'openclip'."
            )
        elif self.emb_model == "dino":
            self.model = torch.hub.load(
                "facebookresearch/dinov2", dino_model_string, force_reload=True
            )
            self.model.eval().to(self.device)
        elif self.emb_model == "openclip":
            self.model, _, self.openclip_preprocess = (
                open_clip.create_model_and_transforms(
                    openclip_model_string, pretrained=openclip_pretrained
                )
            )
            self.model.eval().to(self.device)
        else:
            raise ValueError(
                f"Unknown emb_model '{emb_model}'. Choose 'dino' or 'openclip'."
            )

        if self.method is None:
            raise ValueError(
                "Please provide a correct default clustering method, choose 'hbdscan', 'dbscan', 'kmeans_elbow' or 'kmeans_sil'."
            )

    def forward(
        self,
        eval=True,
        method=None,
        export=None,
        min_k=2,
        reduce_dims=True,
        mask_arr=None,
        num_train=None,
        per_train=0.1,
        run_manual_filter=False,
        eval4_n=10,
        store_frames=False,
        train_dir="train/diversity",
    ):
        if self.data_array is None:
            raise ValueError("data_array must be provided in __init__ to use forward().")

        _save_path = export if export is not None else self.save_path
        _method_map = {
            "kmeans": "kmeans_sil",
            "kmeans_elbow": "kmeans_elbow",
            "kmeans_sil": "kmeans_sil",
            "hdbscan": "hdbscan",
            "dbscan": "dbscan",
        }
        _method = _method_map.get(method or self.method, method or self.method)

        filtered_frames, filtered_masks, all_indices = self.create_train(
            data_arr=self.data_array,
            mask_arr=mask_arr,
            num_train=num_train,
            per_train=per_train,
            method=_method,
            run_eval=eval,
            run_manual_filter=run_manual_filter,
            eval4_n=eval4_n,
            store_frames=store_frames,
            train_dir=train_dir,
        )

        emb_path = os.path.join(_save_path, train_dir, "all_embeddings.npy")
        if os.path.exists(emb_path):
            self.all_emb = np.load(emb_path)

        n_clusters_val = self._last_n_clusters if hasattr(self, "_last_n_clusters") else None
        n_per_cluster_val = self.n_samples_per_cluster

        return filtered_frames, n_clusters_val, n_per_cluster_val, all_indices

    def _n_for_cluster(self, num_train, cluster_labels, cluster_id):
        if not self.proportional_sampling:
            return self.n_samples_per_cluster
        total = np.sum(cluster_labels != -1)
        cluster_size = np.sum(cluster_labels == cluster_id)
        return max(1, round(num_train * cluster_size / total))

    def get_frames_from_mp4(self, video_path, to_rgb=True):
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")

        try:
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

            data_arr = np.empty((total_frames, height, width, 3), dtype=np.uint8)

            frames_read = 0
            for i in range(total_frames):
                ret, frame = cap.read()
                if not ret:
                    break
                if to_rgb:
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                data_arr[frames_read] = frame
                frames_read += 1
                print(f"{frames_read}/{total_frames}", end="\r", flush=True)

            if frames_read < total_frames:
                data_arr = data_arr[:frames_read]

        finally:
            cap.release()

        print(f"\nLoaded data (shape={data_arr.shape})")
        return data_arr

    def run_dino(self, data_arr):
        transform = T.Compose(
            [
                T.ToPILImage(),
                T.Resize((518, 518)),
                T.ToTensor(),
                T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ]
        )

        with torch.no_grad():
            all_out = []
            for i, n in enumerate(data_arr):
                transformed_img = transform(n).to(self.device)
                out = self.model.forward_features(transformed_img[np.newaxis, :])
                all_out.append(out)

        all_emb = []
        for out in all_out:
            all_emb.append(out["x_norm_clstoken"])

        print(f"Extracted DINO embeddings for {len(all_emb)} frames, each of shape {all_emb[0].shape}")
        return np.array(all_emb).squeeze(1)

    def run_openclip(self, data_arr):
        all_emb = []
        with torch.no_grad():
            for i, frame in enumerate(data_arr):
                img = Image.fromarray(frame.astype(np.uint8))
                img_tensor = (
                    self.openclip_preprocess(img).unsqueeze(0).to(self.device)
                )
                features = self.model.encode_image(img_tensor)
                all_emb.append(features.cpu().numpy())

        print(f"Extracted OpenCLIP embeddings for {len(all_emb)} frames, each of shape {all_emb[0].shape}")
        return np.concatenate(all_emb, axis=0)

    def run_dbscan(self, all_emb, num_train, epsilon=None, min_samples=5):
        if epsilon is None:
            k = min_samples
            nbrs = NearestNeighbors(n_neighbors=k).fit(all_emb)
            distances, _ = nbrs.kneighbors(all_emb)
            knn_dists = np.sort(distances[:, -1])
            knee = KneeLocator(
                np.arange(len(knn_dists)),
                knn_dists,
                curve="convex",
                direction="increasing",
            )
            epsilon = (
                knn_dists[knee.knee]
                if knee.knee is not None
                else knn_dists[int(len(knn_dists) * 0.9)]
            )
            print(f"Auto-selected eps={epsilon:.4f}")

        dbscan = DBSCAN(eps=epsilon, min_samples=min_samples)
        cluster_labels = dbscan.fit_predict(all_emb)
        n_clusters = len(set(cluster_labels) - {-1})
        n_noise = (cluster_labels == -1).sum()
        print(f"Finished fitting DBSCAN: {n_clusters} clusters, {n_noise} noise points")

        if self.viz_clusters:
            pca = PCA(n_components=2)
            out = pca.fit_transform(all_emb)
            plt.scatter(x=out[:, 0], y=out[:, 1], c=cluster_labels)
            plt.title("Embedding Scatter Plot (PC decomp)")
            plt.show()

        unique_labels = [l for l in np.unique(cluster_labels) if l != -1]
        centroids = np.array(
            [all_emb[cluster_labels == l].mean(axis=0) for l in unique_labels]
        )

        closest_points = {}
        for centroid, cluster_id in zip(centroids, unique_labels):
            distances = np.linalg.norm(all_emb - centroid, axis=1)
            closest_indices = np.argsort(distances)[:self._n_for_cluster(num_train, cluster_labels, cluster_id)]
            closest_points[cluster_id] = closest_indices

        return n_clusters, cluster_labels, centroids, closest_points

    def run_hdbscan(self, all_emb, num_train, min_cluster_size=10, min_samples=None):
        hdb = HDBSCAN(
            min_cluster_size=min_cluster_size,
            min_samples=min_samples,
            store_centers="centroid",
        )
        cluster_labels = hdb.fit_predict(all_emb)

        n_clusters = len(set(cluster_labels) - {-1})
        n_noise = (cluster_labels == -1).sum()
        print(f"HDBSCAN: {n_clusters} clusters, {n_noise} noise points")

        if self.viz_clusters:
            pca = PCA(n_components=2)
            pca_out = pca.fit_transform(all_emb)
            mask = cluster_labels != -1
            plt.scatter(
                x=pca_out[mask, 0],
                y=pca_out[mask, 1],
                c=cluster_labels[mask],
                cmap="tab20",
            )
            plt.title("Embedding Scatter Plot (PC decomp) — HDBSCAN")
            plt.savefig(f"{self.save_path}/embed_scatter_hbdscan.png", bbox_inches='tight')
            plt.show()
            plt.close()

        unique_labels = [l for l in np.unique(cluster_labels) if l != -1]
        centroids = np.array(
            [all_emb[cluster_labels == l].mean(axis=0) for l in unique_labels]
        )

        closest_points = {}
        for centroid, cluster_id in zip(centroids, unique_labels):
            distances = np.linalg.norm(all_emb - centroid, axis=1)
            closest_indices = np.argsort(distances)[:self._n_for_cluster(num_train, cluster_labels, cluster_id)]
            closest_points[cluster_id] = closest_indices

        return n_clusters, cluster_labels, centroids, closest_points

    def run_knn(
        self,
        all_emb,
        num_train,
        method="silhouette",
        reduce_dims=True,
        n_components=64,
        k_max=50,
        min_k=2,
    ):
        if reduce_dims:
            n_comp = min(n_components, all_emb.shape[0] - 1, all_emb.shape[1])
            pca = PCA(n_components=n_comp)
            all_emb = pca.fit_transform(all_emb)

        if self.optimize_clusters is True:
            k_range = range(max(2, min_k), min(k_max + 1, all_emb.shape[0]))

            if method == "elbow":
                scores = []
                for k in k_range:
                    km = KMeans(n_clusters=k, random_state=0, n_init="auto")
                    km.fit(all_emb)
                    scores.append(km.inertia_)

                knee_locator = KneeLocator(k_range, scores, curve="convex", direction="decreasing")
                optimal_k = knee_locator.knee if knee_locator.knee is not None else 10
                print("KMeans inertias:", dict(zip(k_range, scores)))

                plt.figure(figsize=(8, 5))
                plt.plot(list(k_range), scores, marker="o", linestyle="--")
                plt.xlabel("Number of Clusters (k)")
                plt.ylabel("Inertia")
                plt.title("Elbow Method for Optimal k")

            elif method == "silhouette":
                scores = []
                for k in k_range:
                    labels = KMeans(n_clusters=k, random_state=0, n_init="auto").fit_predict(all_emb)
                    scores.append(silhouette_score(all_emb, labels))

                optimal_k = list(k_range)[int(np.argmax(scores))]
                print("Silhouette scores:", dict(zip(k_range, scores)))

                plt.figure(figsize=(8, 5))
                plt.plot(list(k_range), scores, marker="o", linestyle="--")
                plt.xlabel("Number of Clusters (k)")
                plt.ylabel("Silhouette Score")
                plt.title("Silhouette Score for Optimal k")

            else:
                raise ValueError(f"Unknown method '{method}'. Choose 'elbow' or 'silhouette'.")

            plt.axvline(optimal_k, color="red", linestyle="--", label=f"Optimal k={optimal_k}")
            plt.legend()
            plt.grid(True)
            plt.savefig(f"{self.save_path}/optimal_k.png", bbox_inches='tight')
            plt.show()
            plt.close()
            print(f"Optimal k: {optimal_k}")

        else:
            optimal_k = self.optimize_clusters

        n_clusters = optimal_k
        kmeans = KMeans(n_clusters=optimal_k, random_state=0, n_init="auto")
        cluster_labels = kmeans.fit_predict(all_emb)

        if self.viz_clusters:
            pca_viz = PCA(n_components=2)
            out = pca_viz.fit_transform(all_emb)
            plt.scatter(x=out[:, 0], y=out[:, 1], c=cluster_labels)
            plt.title("Embedding Scatter Plot (PC decomp)")
            plt.savefig(f"{self.save_path}/embed_scatter_knn.png", bbox_inches='tight')
            plt.show()
            plt.close()

        centroids = kmeans.cluster_centers_
        closest_points = {}
        for cluster_id, centroid in enumerate(centroids):
            distances = np.linalg.norm(all_emb - centroid, axis=1)
            closest_points[cluster_id] = np.argsort(distances)[:self._n_for_cluster(num_train, cluster_labels, cluster_id)]

        return n_clusters, cluster_labels, centroids, closest_points, all_emb

    def filter_frames(self, data_arr, closest_points):
        all_indices = []
        all_clusters = []

        for cluster_id in closest_points.keys():
            idxs = closest_points[cluster_id]
            all_indices.extend(idxs)
            all_clusters.extend([cluster_id] * len(idxs))

        print(f"Chose {len(all_indices)} frames")

        if self.plot_chosen_frames:
            for idx, c in zip(all_indices, all_clusters):
                plt.imshow(data_arr[idx])
                plt.title(f"Selected frame idx={idx} cluster={c}")
                plt.show()

        chosen_frames = data_arr[all_indices]
        return chosen_frames, all_indices

    def pairwise_separation(self, X, metric):
        D = pairwise_distances(X, metric=metric)
        triu_idx = np.triu_indices_from(D, k=1)
        pairwise_vals = D[triu_idx]
        avg_sep = pairwise_vals.mean()
        return D, avg_sep

    def evaluate(self, data_arr, all_emb, cluster_labels, centroids, n=10):
        unique_labels = [l for l in np.unique(cluster_labels) if l != -1]

        valid_mask = cluster_labels != -1
        db_score = davies_bouldin_score(
            all_emb[valid_mask], cluster_labels[valid_mask]
        )
        print(f"Davies-Bouldin Index: {db_score:.4f} (lower is better)")

        pca = PCA(n_components=2)
        pca_out = pca.fit_transform(all_emb[valid_mask])
        plt.figure()
        plt.scatter(
            pca_out[:, 0],
            pca_out[:, 1],
            c=cluster_labels[valid_mask],
            cmap="tab20",
        )
        plt.title(f"Cluster Visualization (DB Index = {db_score:.4f})")
        plt.xlabel("PC 1")
        plt.ylabel("PC 2")
        plt.tight_layout()
        plt.savefig(f"{self.save_path}/cluster_vis.png", bbox_inches='tight')
        plt.show()
        plt.close()

        all_cluster = []
        for cid in unique_labels:
            mask = cluster_labels == cid
            data = all_emb[mask]
            pdist_matrix, avg = self.pairwise_separation(data, metric="manhattan")
            all_cluster.append(avg)
        all_cluster = np.array(all_cluster)

        plt.bar(x=np.arange(len(all_cluster)), height=all_cluster)
        plt.xticks(np.arange(len(all_cluster)), [str(l) for l in unique_labels])
        plt.xlabel("Cluster #")
        plt.ylabel("Avg Manhattan Distance")
        plt.title("All Clusters Inner-dist (want to be low)")
        plt.savefig(f"{self.save_path}/clusters_inner_dist.png", bbox_inches='tight')
        plt.show()
        plt.close()

        centroid_pdist = None
        if len(centroids) > 1:
            centroid_pdist, avg_centroid_dist = self.pairwise_separation(
                centroids, metric="manhattan"
            )
            plt.imshow(centroid_pdist)
            plt.colorbar()
            plt.title(f"Pairwise Distance Matrix (Avg = {avg_centroid_dist:.3f})")
            plt.show()
        else:
            centroid_pdist = None
            print("Skipping centroid pairwise eval: fewer than 2 clusters")

        rep_frames = []
        for cid, centroid in zip(unique_labels, centroids):
            cluster_indices = np.where(cluster_labels == cid)[0]
            dists = np.linalg.norm(all_emb[cluster_indices] - centroid, axis=1)
            closest_idx = cluster_indices[np.argmin(dists)]
            rep_frames.append(data_arr[closest_idx])

        n_rep = len(rep_frames)
        ssim_matrix = np.zeros((n_rep, n_rep))
        rep_frames_small = [
            cv2.resize(f, (256, 256), interpolation=cv2.INTER_AREA) for f in rep_frames
        ]
        for i in range(n_rep):
            ssim_matrix[i, i] = 1.0
            for j in range(i + 1, n_rep):
                val = ssim(
                    rep_frames_small[i],
                    rep_frames_small[j],
                    channel_axis=2,
                    data_range=255,
                )
                ssim_matrix[i, j] = val
                ssim_matrix[j, i] = val

        triu_vals = ssim_matrix[np.triu_indices(n_rep, k=1)]
        avg_ssim = triu_vals.mean() if len(triu_vals) > 0 else float("nan")
        plt.imshow(ssim_matrix, vmin=0, vmax=1)
        plt.colorbar()
        plt.title(
            f"SSIM Between Cluster Representatives (Avg = {avg_ssim:.3f}, want low)"
        )
        plt.xlabel("Cluster")
        plt.ylabel("Cluster")
        plt.savefig(f"{self.save_path}/ssim_cluster_rep.png", bbox_inches='tight')
        plt.show()
        plt.close()

        cluster_ssim_avgs = []
        for cid, centroid in zip(unique_labels, centroids):
            cluster_indices = np.where(cluster_labels == cid)[0]
            dists = np.linalg.norm(all_emb[cluster_indices] - centroid, axis=1)
            take = min(n, len(cluster_indices))
            closest_n_idx = cluster_indices[np.argsort(dists)[:take]]
            frames = [
                cv2.resize(data_arr[i], (256, 256), interpolation=cv2.INTER_AREA)
                for i in closest_n_idx
            ]

            ssim_vals = []
            for i in range(len(frames)):
                for j in range(i + 1, len(frames)):
                    val = ssim(frames[i], frames[j], channel_axis=2, data_range=255)
                    ssim_vals.append(val)
            cluster_ssim_avgs.append(np.mean(ssim_vals) if ssim_vals else float("nan"))

        cluster_ssim_avgs = np.array(cluster_ssim_avgs)

        plt.bar(x=np.arange(len(cluster_ssim_avgs)), height=cluster_ssim_avgs)
        plt.xticks(np.arange(len(cluster_ssim_avgs)), [str(l) for l in unique_labels])
        plt.xlabel("Cluster #")
        plt.ylabel("Avg Pairwise SSIM")
        plt.title(f"Intra-cluster SSIM (n={n} frames per cluster, want high)")
        plt.savefig(f"{self.save_path}/cluster_pairwise_ssim.png", bbox_inches='tight')
        plt.show()
        plt.close()

        n_clusters = len(unique_labels)

        mh_norm = (all_cluster - all_cluster.min()) / (
            all_cluster.max() - all_cluster.min() + 1e-9
        )
        compact_score = 1.0 - mh_norm

        ssim_min, ssim_max = np.nanmin(cluster_ssim_avgs), np.nanmax(cluster_ssim_avgs)
        intra_ssim_norm = (cluster_ssim_avgs - ssim_min) / (ssim_max - ssim_min + 1e-9)

        components = [compact_score, intra_ssim_norm]

        if n_clusters > 1:
            centroid_sep = np.array(
                [
                    centroid_pdist[i, np.arange(n_clusters) != i].mean()
                    for i in range(n_clusters)
                ]
            )
            sep_min, sep_max = centroid_sep.min(), centroid_sep.max()
            centroid_sep_norm = (centroid_sep - sep_min) / (sep_max - sep_min + 1e-9)
            components.append(centroid_sep_norm)

            inter_ssim = np.array(
                [
                    ssim_matrix[i, np.arange(n_clusters) != i].mean()
                    for i in range(n_clusters)
                ]
            )
            inter_min, inter_max = inter_ssim.min(), inter_ssim.max()
            inter_ssim_norm = 1.0 - (inter_ssim - inter_min) / (
                inter_max - inter_min + 1e-9
            )
            components.append(inter_ssim_norm)

        composite = np.mean(components, axis=0)
        avg_composite = float(np.nanmean(composite))

        plt.figure(figsize=(8, 5))
        plt.bar(np.arange(len(composite)), composite)
        plt.xticks(np.arange(len(composite)), [str(l) for l in unique_labels])
        plt.xlabel("Cluster #")
        plt.ylabel("Composite Score (0-1)")
        plt.title(f"Composite Cluster Quality Score (Avg = {avg_composite:.3f})")
        plt.ylim(0, 1)
        plt.tight_layout()
        plt.savefig(f"{self.save_path}/cluster_quality_score.png", bbox_inches='tight')
        plt.show()
        plt.close()

    def eval_iso_single(self, all_embs, all_labels, cid):
        embs_for_this_unit = all_embs[all_labels == cid, :]
        embs_for_other_units = all_embs[all_labels != cid, :]

        mean_value = np.expand_dims(np.mean(embs_for_this_unit, 0), 0)

        try:
            VI = np.linalg.inv(np.cov(embs_for_other_units.T))
        except np.linalg.LinAlgError:
            return np.nan, np.nan

        mahalanobis_other = np.sort(
            cdist(mean_value, embs_for_other_units, "mahalanobis", VI=VI)[0]
        )

        mahalanobis_self = np.sort(
            cdist(mean_value, embs_for_this_unit, "mahalanobis", VI=VI)[0]
        )

        n = np.min([embs_for_this_unit.shape[0], embs_for_other_units.shape[0]])

        if n >= 2:
            dof = embs_for_this_unit.shape[1]
            l_ratio = (
                np.sum(1 - chi2.cdf(pow(mahalanobis_other, 2), dof))
                / mahalanobis_self.shape[0]
            )
            isolation_distance = pow(mahalanobis_other[n - 1], 2)
        else:
            l_ratio = np.nan
            isolation_distance = np.nan

        return isolation_distance, l_ratio

    def eval_iso(self, data_arr, all_emb, cluster_labels, centroids, include_outliers=True):
        unique_labels = [l for l in np.unique(cluster_labels) if l != -1]
        all_iso = []
        for cid in unique_labels:
            iso, _ = self.eval_iso_single(all_emb, cluster_labels, cid)
            all_iso.append(iso)
        all_iso = np.array(all_iso)

        mean = all_iso.mean()
        std = all_iso.std()

        outliers = np.abs(all_iso - mean) > 2 * std
        indices = np.where(outliers)[0]
        print(
            f"Detected {np.sum(outliers)} outlier clusters (possibly out of body or occluded/blurry frames)"
        )
        print(f"Outlier cluster indices: {indices}")

        plt.bar(x=np.arange(len(all_iso)), height=all_iso)
        plt.xticks(np.arange(len(all_iso)), [str(l) for l in unique_labels])
        plt.title(f"Cluster Iso Dist | Outliers={indices}")
        plt.savefig(f"{self.save_path}/cluster_iso_dist.png", bbox_inches='tight')
        plt.show()
        plt.close()

        if self.keep_interactive:
            while True:
                user_input = input(
                    "Enter a cluster number to view a frame (or press Enter to skip): "
                ).strip()
                if user_input == "":
                    break
                try:
                    cid = int(user_input)
                    if cid not in unique_labels:
                        print(f"Cluster {cid} not found. Valid clusters: {unique_labels}")
                        continue
                    cid_idx = unique_labels.index(cid)
                    centroid = centroids[cid_idx]
                    cluster_indices = np.where(cluster_labels == cid)[0]
                    dists = np.linalg.norm(all_emb[cluster_indices] - centroid, axis=1)
                    closest_idx = cluster_indices[np.argmin(dists)]
                    plt.imshow(data_arr[closest_idx])
                    plt.title(f"Cluster{cid}_frame_idx{closest_idx}")
                    plt.axis("off")
                    plt.savefig(f"{self.save_path}/cluster_{cid}_frame_{closest_idx}.png", bbox_inches='tight')
                    plt.show()
                    plt.close()
                except ValueError:
                    print("Please enter a valid integer.")

    def eval_tightness(self, all_emb, cluster_labels, centroids):
        unique_labels = [l for l in np.unique(cluster_labels) if l != -1]
        avg_dists = []
        std_dists = []
        for cid, centroid in zip(unique_labels, centroids):
            mask = cluster_labels == cid
            dists = np.linalg.norm(all_emb[mask] - centroid, axis=1)
            avg_dists.append(dists.mean())
            std_dists.append(dists.std())
        avg_dists = np.array(avg_dists)
        std_dists = np.array(std_dists)

        x = np.arange(len(unique_labels))
        plt.figure(figsize=(max(6, len(unique_labels)), 5))
        plt.bar(x, avg_dists, yerr=std_dists, capsize=4)
        plt.xticks(x, [str(l) for l in unique_labels])
        plt.xlabel("Cluster #")
        plt.ylabel("Avg L2 Distance to Centroid")
        plt.title("Cluster Tightness (lower = tighter, error bars = std)")
        plt.savefig(f"{self.save_path}/cluster_tightness_graph.png", bbox_inches='tight')
        plt.show()
        plt.close()

    def filter_clusters_manually(self, all_emb, num_train, cluster_labels, centroids, closest_points):
        unique_labels = [l for l in np.unique(cluster_labels) if l != -1]
        n_clusters = len(unique_labels)
        print(f"\nCurrent clusters: {unique_labels}  (n={len(unique_labels)})")

        new_k = None
        if self.keep_interactive:
            while True:
                raw = input(
                    "Re-cluster with new k (enter integer, or Enter to keep current): "
                ).strip()
                if raw == "":
                    break
                try:
                    new_k = int(raw)
                    if new_k < 2:
                        print("  k must be >= 2.")
                        new_k = None
                        continue
                    break
                except ValueError:
                    print("  Enter an integer.")

        if new_k is not None:
            print(f"Re-clustering into k={new_k}...")
            kmeans = KMeans(n_clusters=new_k, random_state=0, n_init="auto")
            cluster_labels = kmeans.fit_predict(all_emb)
            centroids = kmeans.cluster_centers_
            n_clusters = new_k

            closest_points = {}
            for cluster_id, centroid in enumerate(centroids):
                distances = np.linalg.norm(all_emb - centroid, axis=1)
                closest_points[cluster_id] = np.argsort(distances)[:self._n_for_cluster(num_train, cluster_labels, cluster_id)]
            unique_labels = list(range(new_k))
            print(f"  Done. New clusters: {unique_labels}")

        if self.keep_interactive:
            while True:
                raw = input(
                    "Clusters to remove (comma-separated IDs, or Enter to skip): "
                ).strip()
                if raw == "":
                    clusters_to_remove = []
                    break
                try:
                    clusters_to_remove = [int(x.strip()) for x in raw.split(",") if x.strip()]
                    bad = [c for c in clusters_to_remove if c not in unique_labels]
                    if bad:
                        print(f"  Not found: {bad}. Valid: {unique_labels}")
                        continue
                    break
                except ValueError:
                    print("  Enter integers separated by commas.")

        if clusters_to_remove:
            for cid in clusters_to_remove:
                closest_points.pop(cid, None)
                cluster_labels[cluster_labels == cid] = -1
            remaining = [l for l in np.unique(cluster_labels) if l != -1]
            centroids = np.array(
                [all_emb[cluster_labels == l].mean(axis=0) for l in remaining]
            )
            n_clusters = len(remaining)
            print(f"Removed {clusters_to_remove}. Remaining clusters: {remaining}")

        return n_clusters, all_emb, cluster_labels, centroids, closest_points

    def export_frames(self, chosen_frames, out_folder_name, is_mask=False):
        os.makedirs(out_folder_name, exist_ok=True)

        n = chosen_frames.shape[0]

        for i in range(n):
            img = chosen_frames[i]

            if img.ndim == 2:
                pil_img = Image.fromarray(img, mode="L")
            elif img.ndim == 3 and img.shape[-1] == 1:
                img = img.squeeze(-1)
                pil_img = Image.fromarray(img, mode="L")
            elif img.ndim == 3 and img.shape[-1] == 3:
                pil_img = Image.fromarray(img, mode="RGB")
            else:
                raise ValueError(f"Unsupported image shape: {img.shape}")

            if is_mask:
                pil_img.save(os.path.join(out_folder_name, f"{i:06d}_mask.png"))
            else:
                pil_img.save(os.path.join(out_folder_name, f"{i:06d}.png"))

    def separate_test_val(self,
        data_arr: np.ndarray | None = None,
        data_dir: str | Path | None = None,
        mask_arr: np.ndarray | None = None,
        mask_dir: str | Path | None = None,
        extract_vid: bool = False,
        test_size: float = 0.1,
        val_size: float = 0.1,
        shuffle: bool = True,
        seed: int = 42,
        output_dir: str | Path | None = None,
        ):

        if output_dir is not None:
            output_dir = os.path.join(self.save_path, output_dir)
        else:
            output_dir = self.save_path

        if data_arr is None:
            if data_dir is None:
                raise ValueError("Provide either data_arr or data_dir.")
            data_dir = Path(data_dir)

        if extract_vid:
            print("Extracting frames from videos...")
            chunks = []
            excluded = {"masks", "annotations", "mask"}
            for mp4 in sorted(data_dir.rglob("*")):
                if any(exc in part for part in mp4.parts for exc in excluded):
                    continue
                if mp4.suffix.lower() == ".mp4":
                    chunks.append(self.get_frames_from_mp4(str(mp4)))
            if not chunks:
                raise FileNotFoundError(f"No .mp4 files found under {data_dir}")
            data_arr = np.concatenate(chunks, axis=0)
        elif data_arr is None:
            print("Loading frames from image files...")
            frames_list = []
            excluded = {"masks", "annotations", "mask"}
            for frame in sorted(data_dir.rglob("*")):
                if any(exc in part for part in frame.parts for exc in excluded):
                    continue
                if frame.suffix.lower() in {".jpg", ".jpeg", ".png"}:
                    img = Image.open(frame).convert("RGB")
                    frames_list.append(np.array(img, dtype=np.uint8))
            if not frames_list:
                raise FileNotFoundError(f"No image files found under {data_dir}")
            data_arr = np.stack(frames_list, axis=0)

        if data_arr is None or len(data_arr) == 0:
            raise ValueError(
                "Error in dataset creation, please make sure to input a correct data_dir or numpy data array"
            )

        if mask_arr is None and mask_dir is not None:
            print("Loading masks from mask_dir...")
            mask_dir = Path(mask_dir)
            masks_list = []
            included = {"masks", "annotations", "mask"}
            for mask_path in sorted(mask_dir.rglob("*")):
                is_mask = False
                if any(inc in part for part in mask_path.parts for inc in included):
                    is_mask = True
                if not is_mask:
                    continue
                if mask_path.suffix.lower() in {".jpg", ".jpeg", ".png"}:
                    mask_img = Image.open(mask_path)
                    masks_list.append(np.array(mask_img, dtype=np.uint8))
            if masks_list:
                mask_arr = np.stack(masks_list, axis=0)
                if len(mask_arr) != len(data_arr):
                    raise ValueError(
                        f"Frame count ({len(data_arr)}) and mask count ({len(mask_arr)}) don't match."
                    )

        print("Splitting data into train/val/test...")
        train_val_test_split(
            data_arr=data_arr,
            test_size=test_size,
            val_size=val_size,
            out_folder=output_dir,
            mask_arr=mask_arr,
            shuffle=shuffle,
            seed=seed,
        )

    def separate_by_dir_name(self,
        data_dir: str | Path | None = None,
        mask_dir: str | Path | None = None,
        extract_vid: bool = False,
        mask_type: str = "color_mask",
        global_color_map: dict | None = None,
        target_size: tuple[int, int] | None = (224, 224),
        output_dir: str | Path = "split_data",
    ):

        output_dir = os.path.join(self.save_path, output_dir)
        os.makedirs(output_dir, exist_ok=True)

        if data_dir is None:
            raise ValueError("Provide a data_dir.")

        data_dir = Path(data_dir)
        excluded = {"masks", "annotations", "mask"}
        load_as_rgb = mask_type == "color_mask"
        mask_mode = "RGB" if load_as_rgb else "L"

        splits = {"train": [], "val": [], "test": []}
        mask_splits = {"train": [], "val": [], "test": []}

        def get_split(parts):
            parts_lower = [p.lower() for p in parts]
            if any("test" in p for p in parts_lower):
                return "test"
            elif any("val" in p for p in parts_lower):
                return "val"
            return "train"

        def resize_if_needed(img, target_size, resample):
            if target_size is not None:
                h, w = target_size
                return img.resize((w, h), resample)
            return img

        if extract_vid:
            print("Extracting frames from videos...")
            for mp4 in sorted(data_dir.rglob("*")):
                if any(exc in part for part in mp4.parts for exc in excluded):
                    continue
                if mp4.suffix.lower() != ".mp4":
                    continue
                split = get_split(mp4.parts)
                splits[split].append(self.get_frames_from_mp4(str(mp4)))
        else:
            print("Loading frames from image files...")
            for frame_path in sorted(data_dir.rglob("*")):
                if any(exc in part for part in frame_path.parts for exc in excluded):
                    continue
                if frame_path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
                    continue

                split = get_split(frame_path.parts)

                frame_img = Image.open(frame_path).convert("RGB")
                frame_img = resize_if_needed(frame_img, target_size, Image.BILINEAR)
                splits[split].append(np.array(frame_img, dtype=np.uint8))

                if mask_dir is None:
                    mask_path = frame_path.with_name(f"{frame_path.stem}_{mask_type}.png")
                    if mask_path.exists():
                        mask_img = Image.open(mask_path).convert(mask_mode)
                        mask_img = resize_if_needed(mask_img, target_size, Image.NEAREST)
                        mask_splits[split].append(np.array(mask_img, dtype=np.uint8))

        if mask_dir is not None:
            print("Loading masks from mask_dir...")
            mask_dir = Path(mask_dir)
            included = {"masks", "annotations", "mask"}
            for mask_path in sorted(mask_dir.rglob("*")):
                if not any(inc in part for part in mask_path.parts for inc in included):
                    continue
                if mask_path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
                    continue
                split = get_split(mask_path.parts)
                mask_img = Image.open(mask_path).convert(mask_mode)
                mask_img = resize_if_needed(mask_img, target_size, Image.NEAREST)
                mask_splits[split].append(np.array(mask_img, dtype=np.uint8))

        print("Splitting data into train/val/test...")
        split_arrays = {}
        for split_name, items in splits.items():
            if not items:
                continue
            if extract_vid:
                split_arrays[split_name] = np.concatenate(items, axis=0)
            else:
                split_arrays[split_name] = np.stack(items, axis=0)

        mask_arrays = {}
        for split_name, masks in mask_splits.items():
            if not masks:
                continue
            mask_arrays[split_name] = np.stack(masks, axis=0)

        if not split_arrays:
            raise FileNotFoundError(f"No files found under {data_dir}")

        print("Validating frame and mask counts...")
        for split_name in split_arrays:
            if split_name in mask_arrays:
                if len(split_arrays[split_name]) != len(mask_arrays[split_name]):
                    raise ValueError(
                        f"Frame count ({len(split_arrays[split_name])}) and mask count "
                        f"({len(mask_arrays[split_name])}) don't match for split '{split_name}'."
                    )

        print("Processing color masks into class indices...")
        color_maps = {}
        if load_as_rgb:
            if global_color_map is None:
                all_mask_pixels = []
                for split_name, mask_arr in mask_arrays.items():
                    N, H, W, _ = mask_arr.shape
                    all_mask_pixels.append(mask_arr.reshape(-1, 3))
                all_mask_pixels = np.concatenate(all_mask_pixels, axis=0)
                unique_colors = np.unique(all_mask_pixels, axis=0)
                unique_colors = sorted(unique_colors, key=luminance)
                global_color_map = {
                    idx: [int(c) for c in color]
                    for idx, color in enumerate(unique_colors)
                }

            rgb_to_cls = {tuple(color): idx for idx, color in global_color_map.items()}

            for split_name, mask_arr in mask_arrays.items():
                N, H, W, _ = mask_arr.shape
                flat = mask_arr.reshape(-1, 3)
                flat_out = np.zeros(flat.shape[0], dtype=np.uint8)
                for rgb_tuple, cls in rgb_to_cls.items():
                    flat_out[np.all(flat == np.array(rgb_tuple), axis=1)] = cls
                mask_arrays[split_name] = flat_out.reshape(N, H, W)
                color_maps[split_name] = global_color_map

            with open(os.path.join(output_dir, "color_map.json"), "w") as f:
                json.dump({str(k): v for k, v in global_color_map.items()}, f, indent=2)
        else:
            for split_name in mask_arrays:
                color_maps[split_name] = {}

        print("Saving splits to output directory...")
        for split_name, data_arr in split_arrays.items():
            split_output = os.path.join(output_dir, split_name)
            os.makedirs(split_output, exist_ok=True)
            np.save(os.path.join(split_output, "frames.npy"), data_arr)
            if split_name in mask_arrays:
                np.save(os.path.join(split_output, "masks.npy"), mask_arrays[split_name])

        print("Finished processing all splits.")
        return split_arrays, mask_arrays, color_maps

    def create_train(self, data_arr=None, data_dir=None, mask_arr=None, extract_vid=False, store_frames=False, num_train=None, per_train=0.1, emb_model=None, method=None, run_eval=True, run_manual_filter=False, eval4_n=10, train_dir="train/diversity"):
        self.save_path = data_dir if data_dir is not None else self.save_path
        og_save = self.save_path

        out_path = os.path.join(self.save_path, train_dir)
        os.makedirs(out_path, exist_ok=True)
        self.save_path = out_path

        print("Loading the training data...")
        if data_arr is None:
            if data_dir is None:
                raise ValueError("Provide either data_arr or data_dir.")
            data_dir = Path(data_dir)

        excluded = {"masks", "annotations", "mask", "val", "validation", "test"}
        if extract_vid:
            chunks = []
            for mp4 in sorted(data_dir.rglob("*")):
                if any(exc in part for part in mp4.parts for exc in excluded):
                    continue
                if mp4.suffix.lower() == ".mp4":
                    chunks.append(self.get_frames_from_mp4(str(mp4)))
            if not chunks:
                raise FileNotFoundError(f"No .mp4 files found under {data_dir}")
            data_arr = np.concatenate(chunks, axis=0)
        elif data_arr is None:
            frames_list = []
            for frame in sorted(data_dir.rglob("*")):
                if any(exc in part for part in frame.parts for exc in excluded):
                    continue
                if frame.suffix.lower() in {".jpg", ".jpeg", ".png"}:
                    img = Image.open(frame).convert("RGB")
                    frames_list.append(np.array(img, dtype=np.uint8))
            if not frames_list:
                raise FileNotFoundError(f"No image files found under {data_dir}")
            data_arr = np.stack(frames_list, axis=0)

        if data_arr is None or len(data_arr) == 0:
            raise ValueError(
                "Error in training data creation, please make sure to input a correct data_dir or numpy data array"
            )

        if num_train is None:
            num_train = int(len(data_arr) * per_train)

        print("Computing embeddings...")
        _emb_model = emb_model or self.emb_model
        if _emb_model == "dino":
            all_emb = self.run_dino(data_arr)
        elif _emb_model == "openclip":
            all_emb = self.run_openclip(data_arr)
        else:
            raise ValueError("Must specify a valid embedding model. Choose 'dino' or 'openclip'.")

        self.all_emb = all_emb

        print("Performing clustering...")
        _method = method or self.method
        if _method == "hdbscan":
            n_clusters, cluster_labels, centroids, closest_points = self.run_hdbscan(all_emb, num_train)
        elif _method == "dbscan":
            n_clusters, cluster_labels, centroids, closest_points = self.run_dbscan(all_emb, num_train)
        elif _method == "kmeans_elbow":
            n_clusters, cluster_labels, centroids, closest_points, all_emb = self.run_knn(
                all_emb=all_emb, num_train=num_train, method="elbow"
            )
        elif _method == "kmeans_sil":
            n_clusters, cluster_labels, centroids, closest_points, all_emb = self.run_knn(
                all_emb=all_emb, num_train=num_train, method="silhouette"
            )
        else:
            raise ValueError(
                "Must specify a valid clustering method. Choose 'hdbscan', 'dbscan', 'kmeans_elbow', 'kmeans_sil'."
            )

        self.all_emb = all_emb
        self._last_n_clusters = n_clusters

        if run_eval:
            print("running dataset quality evaluation...")
            self.eval_iso(data_arr=data_arr, all_emb=all_emb, cluster_labels=cluster_labels,
                          centroids=centroids, include_outliers=True)
            self.eval_tightness(all_emb, cluster_labels, centroids)
            self.evaluate(data_arr=data_arr, all_emb=all_emb, cluster_labels=cluster_labels, centroids=centroids, n=eval4_n)

            if run_manual_filter:
                n_clusters, all_emb, cluster_labels, centroids, closest_points = self.filter_clusters_manually(all_emb, num_train, cluster_labels, centroids, closest_points)

        filtered_frames, all_indices = self.filter_frames(data_arr, closest_points)

        print(f"Saving files and metadata to: {out_path}...")
        if store_frames:
            frame_out_path = os.path.join(out_path, "frames")
            os.makedirs(frame_out_path, exist_ok=True)
            print(f"Also saving filtered frames to: {frame_out_path}...")
            self.export_frames(chosen_frames=filtered_frames, out_folder_name=frame_out_path, is_mask=False)

        np.save(os.path.join(out_path, "frames.npy"), filtered_frames)

        filtered_masks = None
        if mask_arr is not None:
            filtered_masks = mask_arr[all_indices]
            if store_frames:
                mask_out_path = os.path.join(out_path, "masks")
                os.makedirs(mask_out_path, exist_ok=True)
                print(f"Also saving filtered masks to: {mask_out_path}...")
                self.export_frames(chosen_frames=filtered_masks, out_folder_name=mask_out_path, is_mask=True)
        else:
            mask_dir = Path(og_save)
            masks_list = []
            included = {"masks", "annotations", "mask"}
            for mask_path in sorted(mask_dir.rglob("*")):
                is_mask = False
                if any(exc in part for part in mask_path.parts for exc in excluded):
                    continue
                if any(inc in part for part in mask_path.parts for inc in included):
                    is_mask = True
                if not is_mask:
                    continue
                if mask_path.suffix.lower() in {".jpg", ".jpeg", ".png"}:
                    mask_img = Image.open(mask_path)
                    masks_list.append(np.array(mask_img, dtype=np.uint8))
            if masks_list:
                loaded_masks = np.stack(masks_list, axis=0)
                filtered_masks = loaded_masks[all_indices]
                if store_frames:
                    mask_out_path = os.path.join(out_path, "masks")
                    os.makedirs(mask_out_path, exist_ok=True)
                    print(f"Also saving filtered masks to: {mask_out_path}...")
                    self.export_frames(chosen_frames=filtered_masks, out_folder_name=mask_out_path, is_mask=True)

        if filtered_masks is not None:
            np.save(os.path.join(out_path, "masks.npy"), filtered_masks)

        np.save(os.path.join(out_path, "all_embeddings.npy"), all_emb)

        metadata = {
            "n_clusters": int(n_clusters),
            "num_frames": len(data_arr),
            "all_indices": [int(i) for i in all_indices],
            "cluster_labels": [int(i) for i in cluster_labels],
        }
        metadata_path = os.path.join(out_path, "training_metadata.json")
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)

        self.save_path = og_save

        return filtered_frames, filtered_masks, all_indices