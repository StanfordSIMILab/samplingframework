# diversity_sampler.py 
# DiversitySampler class for performing diversity sampling on frames, with various clustering methods and evaluation metrics.
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
from sklearn.metrics.pairwise import cosine_distances

from skimage.metrics import structural_similarity as ssim
from sklearn.neighbors import NearestNeighbors
from scipy.spatial.distance import cdist
from scipy.stats import chi2
import umap

from auxiliary.fvi_computation import elbow_threshold, compute_fvi, fvi_filter, show_fvi_histogram

class DiversitySampler:
    def __init__(
        self,
        n_samples_per_cluster=None, # Value to sample equal number from clusters
        proportional_sampling=None, # Default true if n_samples_per cluster not provided
        optim_clusters=True, # Whether to automatically optimize the number of clusters for KMeans using silhouette or elbow method (if False, will use n_clusters specified in optim_clusters)
        viz_clusters=True, # Whether to visualize clusters after dimensionality reduction with PCA
        plot_chosen_frames=True, # Whether to plot the chosen frames after sampling
        save_plots=True, # Whether to save plots to disk
        emb_model=None, # Embedding model to use for clustering, choose 'dino' or 'openclip'
        method=None, # Clustering method to use: 'hbdscan', 'dbscan', 'kmeans_elbow', or 'kmeans_sil'
        dino_model_string="dinov2_vits14", # DINO model variant to use, e.g. 'dinov2_vits14', 'dinov2_vitb14', etc.
        openclip_model_string="ViT-B-32", # OpenCLIP model variant to use, e.g. 'ViT-B-32', 'ViT-L-14', etc.
        openclip_pretrained="openai", # OpenCLIP pretrained weights to use, e.g. 'openai', 'laion-400m', etc.
        keep_interactive=False # Whether to keep interactive plots open after showing (e.g. for manual cluster filtering), or to automatically close them after showing
    ):
        self.n_samples_per_cluster = n_samples_per_cluster
        self.proportional_sampling = proportional_sampling if proportional_sampling is not None else (n_samples_per_cluster is None)
        self.optimize_clusters = optim_clusters
        self.viz_clusters = viz_clusters
        self.plot_chosen_frames = plot_chosen_frames
        self.save_plots = save_plots
        self.emb_model = emb_model
        self.method = method
        self.dino_model_string = dino_model_string
        self.openclip_model_string = openclip_model_string
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

    # Helper function to determine number of samples from each cluster
    def _n_for_cluster(self, num_train, cluster_labels, cluster_id):
        if not self.proportional_sampling:
            return self.n_samples_per_cluster
        total = np.sum(cluster_labels != -1)
        cluster_size = np.sum(cluster_labels == cluster_id)
        return max(1, round(num_train * cluster_size / total))

    # Embedding Methods
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

    # Clustering Methods
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

    def run_hdbscan(self, all_emb, num_train, min_cluster_size=10, min_samples=None, save_path="./diversity"):
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
            plt.savefig(f"{save_path}/embed_scatter_hbdscan.png", bbox_inches='tight')
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
        save_path="./diversity"
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
            plt.savefig(f"{save_path}/optimal_k.png", bbox_inches='tight')
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
            plt.savefig(f"{save_path}/embed_scatter_knn.png", bbox_inches='tight')
            plt.show()
            plt.close()

        centroids = kmeans.cluster_centers_
        closest_points = {}
        for cluster_id, centroid in enumerate(centroids):
            distances = np.linalg.norm(all_emb - centroid, axis=1)
            closest_points[cluster_id] = np.argsort(distances)[:self._n_for_cluster(num_train, cluster_labels, cluster_id)]

        return n_clusters, cluster_labels, centroids, closest_points, all_emb

    # Cluster Quality Metrics
    def pairwise_separation(self, X, metric):
        D = pairwise_distances(X, metric=metric)
        triu_idx = np.triu_indices_from(D, k=1)
        pairwise_vals = D[triu_idx]
        avg_sep = pairwise_vals.mean()
        return D, avg_sep

    def calc_hausdorff_coverage(self, X_full: np.ndarray, X_sub: np.ndarray):
        nn = NearestNeighbors(n_neighbors=1).fit(X_sub)
        dists = nn.kneighbors(X_full)[0].flatten()
        return dists.max(), dists.mean()

    def calc_intra_spread(self, X_sub: np.ndarray) -> float:
        D = cosine_distances(X_sub)
        np.fill_diagonal(D, np.nan)
        return float(np.nanmean(D))

    # Visualization Plots for visualizing diversity sampling / comparing with random sampling
    def plot_hausdorff_spread(self, full_n: np.ndarray, diverse_n: np.ndarray, random_n: np.ndarray | None, n_sampled: int, save_path: str = "./data_quality", save_plot: bool = False) -> None:
        os.makedirs(save_path, exist_ok=True)

        hd_div, mn_div = self.calc_hausdorff_coverage(full_n, diverse_n)
        hd_ran, mn_ran = self.calc_hausdorff_coverage(full_n, random_n)
        sp_div = self.calc_intra_spread(diverse_n)
        sp_ran = self.calc_intra_spread(random_n)
        print(f"Hausdorff  — diverse: {hd_div:.4f}  |  random: {hd_ran:.4f}")
        print(f"Mean NN    — diverse: {mn_div:.4f}  |  random: {mn_ran:.4f}")
        print(f"Spread     — diverse: {sp_div:.4f}  |  random: {sp_ran:.4f}")

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        fig.suptitle(f"Diverse (n={n_sampled}) vs Random (n={n_sampled})", fontsize=13)
        for ax, vals, ylabel, title in [
            (axes[0], [hd_div, hd_ran], "Hausdorff distance", "Worst-case coverage (lower = better)"),
            (axes[1], [sp_div, sp_ran], "Mean pairwise cosine dist", "Intra-subset spread (higher = less redundant)"),
        ]:
            bars = ax.bar(["diverse", "random"], vals, color=["steelblue", "darkorange"], width=0.4)
            ax.bar_label(bars, fmt="%.4f", padding=3)
            ax.set_ylabel(ylabel)
            ax.set_title(title)
            ax.set_ylim(0, max(vals) * 1.2)
        plt.tight_layout()
        plt.savefig(f"{save_path}/coverage_redundancy.png", dpi=150) if self.save_plots or save_plot else None
        plt.show()
        plt.close()
    
    def plot_nn_coverage(self, full_n: np.ndarray, diverse_n: np.ndarray, random_n: np.ndarray | None, save_path: str = "./data_quality", save_plot: bool = False) -> None:
        os.makedirs(save_path, exist_ok=True)

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
            plt.savefig(f"{save_path}/nn_coverage_{label}.png", dpi=150) if self.save_plots or save_plot else None
            plt.show()
            plt.close()

    def plot_pca_scatter(self, full_n: np.ndarray, diverse_n: np.ndarray, random_n: np.ndarray | None, save_path: str = "./data_quality", save_plot: bool = False) -> None:
        os.makedirs(save_path, exist_ok=True)

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
            dists = NearestNeighbors(n_neighbors=1).fit(sub_n_emb).kneighbors(full_n)[0].flatten()
            hd, mean_nn = self.calc_hausdorff_coverage(full_n, sub_n_emb)
            sc = ax.scatter(full_2d[:, 0], full_2d[:, 1], c=dists, cmap="RdYlGn_r", s=4, alpha=0.7, vmin=0, vmax=vmax)
            ax.scatter(sub_2d[:, 0], sub_2d[:, 1], c="black", s=25, marker="x", linewidths=0.8, label=f"{label} samples", zorder=5)
            ax.set_title(f"{label} (n={len(sub_2d)})  |  Mean NN: {mean_nn:.4f}  |  HD: {hd:.4f}")
            ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0] * 100:.1f}% var)")
            ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1] * 100:.1f}% var)")
            ax.legend(markerscale=2)
            plt.colorbar(sc, ax=ax, label="NN distance to subset")
        plt.tight_layout()
        plt.savefig(f"{save_path}/pca_coverage_heatmap.png", dpi=150) if self.save_plots or save_plot else None
        plt.show()
        plt.close()

    def plot_umap(self, full_n: np.ndarray, diverse_indices: np.ndarray, random_indices: np.ndarray | None, save_path: str = "./data_quality", save_plot: bool = False) -> None:
        os.makedirs(save_path, exist_ok=True)
        
        print("Fitting UMAP...")
        emb_2d = umap.UMAP(n_components=2, random_state=42).fit_transform(full_n)
        fig, axes = plt.subplots(1, 2, figsize=(18, 8))
        for ax, indices, title in [
            (axes[0], np.array(diverse_indices), "Diverse Samples"),
            (axes[1], np.array(random_indices), "Random Samples"),
        ]:
            mask = np.ones(len(full_n), dtype=bool)
            mask[indices] = False
            ax.scatter(emb_2d[mask, 0], emb_2d[mask, 1], s=4, alpha=0.4, color="steelblue", label="all frames")
            ax.scatter(emb_2d[indices, 0], emb_2d[indices, 1], s=25, alpha=0.9, color="red", label=f"selected ({len(indices)})")
            ax.set_title(title, fontsize=14)
            ax.set_xlabel("UMAP 1")
            ax.set_ylabel("UMAP 2")
            ax.legend(markerscale=2)
        plt.tight_layout()
        plt.savefig(f"{save_path}/umap_selected.png", dpi=150) if self.save_plots or save_plot else None
        plt.show()
        plt.close()

    # Evaluation Functions
    def evaluate(self, data_arr, all_emb, cluster_labels, centroids, n=10, save_path="./metrics", save_plot=False) -> None:
        """ 
        
        Evaluate clustering and diversity metrics including: 
            Davies-Bouldin Index, 
            PCA visualization, 
            intra-cluster distances, 
            centroid distances, 
            and SSIM between cluster representatives.
        
        """
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
        plt.savefig(f"{save_path}/cluster_vis.png", bbox_inches='tight') if self.save_plots or save_plot else None
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
        plt.savefig(f"{save_path}/clusters_inner_dist.png", bbox_inches='tight') if self.save_plots or save_plot else None
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
            plt.savefig(f"{save_path}/centroid_pdist.png", bbox_inches='tight') if self.save_plots or save_plot else None
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
        plt.savefig(f"{save_path}/ssim_cluster_rep.png", bbox_inches='tight') if self.save_plots or save_plot else None
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
        plt.savefig(f"{save_path}/cluster_pairwise_ssim.png", bbox_inches='tight') if self.save_plots or save_plot else None
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
        plt.savefig(f"{save_path}/cluster_quality_score.png", bbox_inches='tight') if self.save_plots or save_plot else None
        plt.show()
        plt.close()

    # Evaluate the diversity sampled frames against other sampled frames provided
    def evaluate_vs_other(
        self,
        random_indices: np.ndarray,
        all_emb: np.ndarray = None,
        diverse_indices: np.ndarray = None,
        save_dir: str = "./comparison", # Sub-directory where data is stored and where evaluation plots will be saved
        save_plot: bool = False,
    ) -> None:

        os.makedirs(save_dir, exist_ok=True)

        # Load embeddings from disk if not supplied directly.
        if all_emb is None:
            emb_path = os.path.join(save_dir, "all_embeddings.npy")
            if not os.path.exists(emb_path):
                raise FileNotFoundError(
                    f"all_embeddings.npy not found at {emb_path}. "
                    "Run create_train() first or pass all_emb explicitly."
                )
            all_emb = np.load(emb_path)
            print(f"Loaded embeddings from {emb_path}")

        # Load diverse indices from disk if not supplied directly.
        if diverse_indices is None:
            idx_path = os.path.join(save_dir, "diverse_indices.npy")
            if not os.path.exists(idx_path):
                raise FileNotFoundError(
                    f"diverse_indices.npy not found at {idx_path}. "
                    "Run create_train() first or pass diverse_indices explicitly."
                )
            diverse_indices = np.load(idx_path)
            print(f"Loaded diverse indices from {idx_path}")

        diverse_indices = np.asarray(diverse_indices)
        random_indices = np.asarray(random_indices)
        diverse_n = all_emb[diverse_indices]
        random_n = all_emb[random_indices]
        n_sampled = len(diverse_indices)

        # Temporarily point save_path at the train sub-directory so plot
        # helpers write their PNGs in the right place.
        self.plot_hausdorff_spread(all_emb, diverse_n, random_n, n_sampled, save_path=save_dir, save_plot=save_plot)
        self.plot_nn_coverage(all_emb, diverse_n, random_n, save_path=save_dir, save_plot=save_plot)
        self.plot_pca_scatter(all_emb, diverse_n, random_n, save_path=save_dir, save_plot=save_plot)
        self.plot_umap(all_emb, diverse_indices, random_indices, save_path=save_dir, save_plot=save_plot)

    # Evaluate one cluster's isolation distance and L-ratio, then identify outliers based on isolation distance distribution
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

    # For each cluster, compute isolation distance and L-ratio, then identify outliers based on isolation distance distribution
    def eval_iso(self, data_arr, all_emb, cluster_labels, centroids, include_outliers=True, save_data=False, save_path="./metrics", save_plot=False):
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
        plt.savefig(f"{save_path}/cluster_iso_dist.png", bbox_inches='tight') if self.save_plots or save_plot else None
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
                    plt.savefig(f"{save_path}/cluster_{cid}_frame_{closest_idx}.png", bbox_inches='tight') if self.save_plots or save_plot else None
                    plt.show()
                    plt.close()
                except ValueError:
                    print("Please enter a valid integer.")
    
    # Evaluate tightness of each cluster
    def eval_tightness(self, all_emb, cluster_labels, centroids, save_path="./diversity", save_plot=False):
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
        plt.savefig(f"{save_path}/cluster_tightness_graph.png", bbox_inches='tight') if self.save_plots or save_plot else None
        plt.show()
        plt.close()

    # Allow user to manually filter clusters by changing k or specifying clusters to remove
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

    # Helper functions for filtering and exporting files
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

    def export_frames(self, chosen_frames, out_folder_name="./exported_frames", is_mask=False):
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

    # Main function to create diverse training dataset from frames w/ optional mask.
    # Saves all_embeddings.npy and diverse_indices.npy to <data_dir>/diversity/
    def sample(self, 
        data_arr, 
        mask_arr=None, 
        num_samples=None, 
        percent_sample=0.1, 
        emb_prev=None,
        emb_model=None, 
        method=None,
        fvi_filtering=True, 
        run_eval=True, 
        run_manual_filter=False, 
        eval4_n=10, 
        save_data=False,
        data_dir="."
    ):
        if data_arr is None or len(data_arr) == 0:
            raise ValueError(
                "Error in training data creation, please make sure to input a correct numpy data array"
            )

        if save_data:
            out_path = os.path.join(data_dir, "diversity")
            os.makedirs(out_path, exist_ok=True)
        else:
            out_path = os.path.join(data_dir, "diversity")

        if num_samples is None:
            # Sample 10% of frames by default unless user specifies otherwise
            num_samples = int(len(data_arr) * percent_sample)

        if emb_prev is not None:
            print("Using previously computed embeddings...")
            all_emb = emb_prev
        else:
            print("Computing embeddings...")
            _emb_model = emb_model or self.emb_model
            if _emb_model == "dino":
                all_emb = self.run_dino(data_arr)
            elif _emb_model == "openclip":
                all_emb = self.run_openclip(data_arr)
            else:
                raise ValueError("Must specify a valid embedding model. Choose 'dino' or 'openclip'.")

        print("Performing clustering...")
        _method = method or self.method
        if _method == "hdbscan":
            n_clusters, cluster_labels, centroids, closest_points = self.run_hdbscan(all_emb, num_samples)
        elif _method == "dbscan":
            n_clusters, cluster_labels, centroids, closest_points = self.run_dbscan(all_emb, num_samples)
        elif _method == "kmeans_elbow":
            n_clusters, cluster_labels, centroids, closest_points, all_emb = self.run_knn(
                all_emb=all_emb, num_train=num_samples, method="elbow"
            )
        elif _method == "kmeans_sil":
            n_clusters, cluster_labels, centroids, closest_points, all_emb = self.run_knn(
                all_emb=all_emb, num_train=num_samples, method="silhouette"
            )
        else:
            raise ValueError(
                "Must specify a valid clustering method. Choose 'hdbscan', 'dbscan', 'kmeans_elbow', 'kmeans_sil'."
            )

        _last_n_clusters = n_clusters
        _last_closest_points = closest_points

        if run_eval:
            print("running dataset quality evaluation...")
            self.eval_iso(data_arr=data_arr, all_emb=all_emb, cluster_labels=cluster_labels,
                          centroids=centroids, include_outliers=True, save_path=out_path, save_plot=save_data)
            self.eval_tightness(all_emb=all_emb, cluster_labels=cluster_labels, centroids=centroids, save_path=out_path, save_plot=save_data)
            self.evaluate(data_arr=data_arr, all_emb=all_emb, cluster_labels=cluster_labels,
                          centroids=centroids, n=eval4_n, save_path=out_path, save_plot=save_data)

            if run_manual_filter:
                n_clusters, all_emb, cluster_labels, centroids, closest_points = self.filter_clusters_manually(all_emb, num_samples, cluster_labels, centroids, closest_points)
                _last_closest_points = closest_points

        filtered_frames, all_indices = self.filter_frames(data_arr, _last_closest_points)
        filtered_masks = mask_arr[all_indices] if mask_arr is not None else None

        if save_data:
            print(f"Saving files and metadata to: {out_path}...")
            frame_out_path = os.path.join(out_path, "frames")
            os.makedirs(frame_out_path, exist_ok=True)
            print(f"Also saving filtered frames to: {frame_out_path}...")
            self.export_frames(chosen_frames=filtered_frames, out_folder_name=frame_out_path, is_mask=False)

            np.save(os.path.join(out_path, "frames.npy"), filtered_frames)

            filtered_masks = None
            if mask_arr is not None:
                filtered_masks = mask_arr[all_indices]
                if save_data:
                    mask_out_path = os.path.join(out_path, "masks")
                    os.makedirs(mask_out_path, exist_ok=True)
                    print(f"Also saving filtered masks to: {mask_out_path}...")
                    self.export_frames(chosen_frames=filtered_masks, out_folder_name=mask_out_path, is_mask=True)

            if filtered_masks is not None:
                np.save(os.path.join(out_path, "masks.npy"), filtered_masks)

            # Save embeddings and diverse indices so evaluate_vs_random() can be
            # called later with caller-supplied random_indices, without re-running
            # the full pipeline.
            np.save(os.path.join(out_path, "all_embeddings.npy"), all_emb)
            np.save(os.path.join(out_path, "diverse_indices.npy"), np.array(all_indices))

            metadata = {
                "n_clusters": int(n_clusters),
                "num_frames": len(data_arr),
                "all_indices": [int(i) for i in all_indices],
                "cluster_labels": [int(i) for i in cluster_labels],
            }
            metadata_path = os.path.join(out_path, "diversity_metadata.json")
            with open(metadata_path, "w") as f:
                json.dump(metadata, f, indent=2)

        return num_samples, filtered_frames, filtered_masks, all_indices