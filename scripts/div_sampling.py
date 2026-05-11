from pathlib import Path
import cv2
import tempfile, os
import json
import threading

import numpy as np
import torch
import torchvision.transforms as T
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

class DiversitySampler:
    def __init__(
        self,
        n_samples_per_cluster = 250,
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
    ):
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

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if self.emb_model is None:
            raise ValueError(
                "Please provide an embedding model for clustering, choose 'dino' or 'openclip'."
            )
        elif self.emb_model == "dino":
            # shutil.rmtree(r"C:\Users\krsid\.cache\torch\hub\facebookresearch_dinov2_main", ignore_errors=True)
            self.model = torch.hub.load(
                "facebookresearch/dinov2", dino_model_string, force_reload=True
            )
            self.model.eval().to(self.device)
        elif self.emb_model == "openclip":
            import open_clip

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
                "Please provide a default clustering method, choose 'hbdscan', 'dbscan', 'kmeans_elbow' or 'kmeans_sil'."
            )

    def _n_for_cluster(self, num_train, cluster_labels, cluster_id):
        """
        If proportional_sampling=True, each cluster gets a proportional
        number of samples matching up to approximately the number of samples
        determined from either the user's selected num_train or percent train.
        If proportional_sampling=False, use the sampler default n_samples_per_cluster
        """
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

            # Trim if we read fewer frames than expected (common with some codecs)
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
                print(f"{i}/ {len(data_arr)}")
                transformed_img = transform(n).to(self.device)
                out = self.model.forward_features(transformed_img[np.newaxis, :])
                all_out.append(out)

        all_emb = []
        for out in all_out:
            all_emb.append(out["x_norm_clstoken"])
        
        return np.array(all_emb).squeeze(1)

    def run_openclip(self, data_arr):
        all_emb = []
        with torch.no_grad():
            for i, frame in enumerate(data_arr):
                print(f"{i}/ {len(data_arr)}")
                img = Image.fromarray(frame.astype(np.uint8))
                img_tensor = (
                    self.openclip_preprocess(img).unsqueeze(0).to(self.device)
                )
                features = self.model.encode_image(img_tensor)
                all_emb.append(features.cpu().numpy())
        
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
        method = "silhouette",  #Allow choice between "elbow" vs "silhouette"
        reduce_dims = True,
        n_components = 64,
        k_max = 50,
        min_k = 2,
    ):
        # Allow for dimensionality reduction before clustering
        if reduce_dims:
            n_comp = min(n_components, all_emb.shape[0] - 1, all_emb.shape[1])
            pca = PCA(n_components=n_comp)
            all_emb = pca.fit_transform(all_emb)

        # Find optimal cluster through iterating through num cluster hyperparameter
        if self.optimize_clusters is True:
            k_range = range(max(2, min_k), min(k_max + 1, all_emb.shape[0]))

            if method == "elbow":
                scores = []
                for k in k_range:
                    km = KMeans(n_clusters=k, random_state=0, n_init="auto")
                    km.fit(all_emb)
                    scores.append(km.inertia_)
                    print(f"k={k}  inertia={scores[-1]:.2f}")

                knee_locator = KneeLocator(k_range, scores, curve="convex", direction="decreasing")
                optimal_k = knee_locator.knee if knee_locator.knee is not None else 10

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
                    print(f"k={k}  silhouette={scores[-1]:.4f}")

                optimal_k = list(k_range)[int(np.argmax(scores))]

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

        # Final KMeans clustering
        n_clusters = optimal_k
        kmeans = KMeans(n_clusters=optimal_k, random_state=0, n_init="auto")
        cluster_labels = kmeans.fit_predict(all_emb)

        # If you want to visualize clusters
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

        return n_clusters, cluster_labels, centroids, closest_points

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

        # take upper triangle excluding diagonal
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
        # copied from: https://github.com/SpikeInterface/spikemetrics/blob/master/spikemetrics/metrics.py#L939
        # Schmitzer-Torbert et al. (2005) Neurosci 131: 1-11

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

        # Step 1: optionally re-cluster with a new k
        new_k = None
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

        # Step 2: remove specific clusters
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

    def export_frames(self, chosen_frames, out_folder_name):
        os.makedirs(out_folder_name, exist_ok=True)

        n = chosen_frames.shape[0]

        for i in range(n):
            img = chosen_frames[i]

            if img.dtype != np.uint8:
                img = np.clip(img, 0, 255).astype(np.uint8)

            if img.shape[-1] == 1:
                img = img.squeeze(-1)
                pil_img = Image.fromarray(img, mode="L")
            else:
                pil_img = Image.fromarray(img, mode="RGB")

            pil_img.save(os.path.join(out_folder_name, f"{i:06d}.png"))

    def create_dataset(self, data_arr=None, data_dir=None, extract_vid=False, num_train=None, per_train = 0.1, emb_model=None, method=None, run_eval=True, eval4_n=10, out_dir=None):
        """
            Select training frames from scratch taking in: data_dir of video 
                                                           "         " frames
                                                           np data array of training frames

        """

        # Define output directory:
        if out_dir is not None:
            self.save_path = os.path.join(out_dir)
        else:
            self.save_path = os.path.join(".")

        # Create data_arr if not already created
        if data_arr is None and extract_vid:
            data_arr = self.get_frames_from_mp4(data_dir)
        elif data_arr is None:
            data_dir = Path(data_dir)
            data_arr = []
            excluded = {"masks", "annotations", "mask", "val", "validation", "test"}
            for frame in sorted(data_dir.rglob("*")):
                if any(exc in part for part in frame.parts for exc in excluded):
                    continue
                if frame.suffix.lower() in {".jpg", ".jpeg", ".png"}:
                    image = Image.open(frame)
                    image = np.array(image.convert("RGB"))
                    data_arr.append(image)
                
            data_arr = np.array(data_arr)

        # Make sure data array was created properly before preceeding
        if data_arr is None or len(data_arr) == 0:
            raise ValueError(
                "Error in dataset creation, please make sure to input a correct data_dir or numpy data array"
            )
        
        if num_train is None:
            num_train = int(len(data_arr) * per_train)

        #Allow override of embed_model for specific dataset creation, otherwise default to default method
        if emb_model is None:
            if self.emb_model == "dino":
                all_emb = self.run_dino(data_arr)
            elif self.emb_model == "openclip":
                all_emb = self.run_openclip(data_arr)
        else:
            if emb_model == "dino":
                all_emb = self.run_dino(data_arr)
            elif emb_model == "openclip":
                all_emb = self.run_openclip(data_arr)
            else:
                raise ValueError(
                "No correct emb_model specified in function call or DiversitySampler, Choose 'dino' or 'openclip'."
                )

        if method is None:
            if self.method == "hdbscan":
                n_clusters, cluster_labels, centroids, closest_points = self.run_hdbscan(all_emb, num_train)
            elif self.method == "dbscan":
                n_clusters, cluster_labels, centroids, closest_points = self.run_dbscan(all_emb, num_train)
            elif self.method == "kmeans_elbow":
                n_clusters, cluster_labels, centroids, closest_points = self.run_knn(all_emb=all_emb, num_train=num_train, method = "elbow")
            elif self.method == "kmeans_sil":
                n_clusters, cluster_labels, centroids, closest_points = self.run_knn(all_emb, num_train)
        else:
            if method == "hdbscan":
                n_clusters, cluster_labels, centroids, closest_points = self.run_hdbscan(all_emb, num_train)
            elif method == "dbscan":
                n_clusters, cluster_labels, centroids, closest_points = self.run_dbscan(all_emb, num_train)
            elif method == "kmeans_elbow":
                n_clusters, cluster_labels, centroids, closest_points = self.run_knn(all_emb=all_emb, num_train=num_train, method = "elbow")
            elif method == "kmeans_sil":
                n_clusters, cluster_labels, centroids, closest_points = self.run_knn(all_emb, num_train)
            else:
                raise ValueError("No correct clustering method specified in function call or DiversitySampler, Choose 'hbdscan', 'dbscan', 'kmeans_elbows', 'kmeans_sil'.")

        if run_eval:
            self.eval_iso(data_arr=data_arr, all_emb=all_emb, cluster_labels=cluster_labels, 
                          centroids=centroids, include_outliers=True)
            self.eval_tightness(all_emb, cluster_labels, centroids)
            self.evaluate(data_arr, all_emb, cluster_labels, centroids, n=eval4_n)
            n_clusters, all_emb, cluster_labels, centroids, closest_points = self.filter_clusters_manually(all_emb, num_train, cluster_labels, centroids, closest_points)

        filtered_frames, all_indices = self.filter_frames(data_arr, closest_points)

        # Save filtered frames, full dataset, and embeddings
        self.export_frames(filtered_frames, self.save_path)
        np.save(os.path.join(self.save_path, "data_array"), data_arr)
        np.save(os.path.join(self.save_path, "all_embeddings.npy"), all_emb)

        #write out to a meta_data.json with n_clusters and all_indices and num_training
        metadata = {
            "n_clusters": int(n_clusters),
            "num_frames": len(data_arr),
            "all_indices": [int(i) for i in all_indices],

        }
        metadata_path = os.path.join(self.save_path, "metadata.json")
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)

        return filtered_frames, all_indices