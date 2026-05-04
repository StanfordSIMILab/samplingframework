import numpy as np
import torch
import threading
import torchvision.transforms as T
import shutil
from kneed import KneeLocator
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, DBSCAN, HDBSCAN
from sklearn.decomposition import PCA
from sklearn.metrics import pairwise_distances, davies_bouldin_score, silhouette_score
import cv2
import tempfile, os
from PIL import Image
from skimage.metrics import structural_similarity as ssim
from sklearn.neighbors import NearestNeighbors
from scipy.spatial.distance import cdist
from scipy.stats import chi2


class DiversitySampling:
    def __init__(
        self,
        data_array,
        optim_clusters,
        dino_model_string,
        n_samples_per_cluster,
        viz_clusters,
        emb_prev,
        plot_chosen_frames,
        emb_model="dino",
        openclip_model_string="ViT-B-32",
        openclip_pretrained="openai",
        save_path="data_hooray.npy",
    ):
        self.data_array = data_array
        self.optimize_clusters = optim_clusters
        self.dino_model_string = dino_model_string
        self.n_samples_per_cluster = n_samples_per_cluster
        self.viz_clusters = viz_clusters
        self.emb_prev = emb_prev
        self.plot_chosen_frames = plot_chosen_frames
        self.emb_model = emb_model
        self.save_path = save_path

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if emb_prev is None:
            if emb_model == "dino":
                # shutil.rmtree(r"C:\Users\krsid\.cache\torch\hub\facebookresearch_dinov2_main", ignore_errors=True)
                self.model = torch.hub.load(
                    "facebookresearch/dinov2", dino_model_string, force_reload=True
                )
                self.model.eval().to(self.device)
            elif emb_model == "openclip":
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

    def get_frames_from_mp4(self, video_path, to_rgb):
        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frames = []
        i = 0

        while i < total_frames:
            result = {}

            def _read(cap=cap, result=result):
                result["ret"], result["frame"] = cap.read()

            t = threading.Thread(target=_read)
            t.start()
            t.join(timeout=5.0)
            if t.is_alive():
                print(f"\nFrame read timed out at {i}/{total_frames}, stopping early.")
                break

            ret, frame = result.get("ret", False), result.get("frame", None)
            if not ret:
                break

            if to_rgb:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            frames.append(frame)
            i += 1
            print(f"{i}/{total_frames}", end="\r", flush=True)

        print()
        cap.release()

        self.data_array = np.stack(frames, axis=0)
        print(f"Loaded data (shape={self.data_array.shape})")

    def run_dino(self):
        if self.emb_prev is None:
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
                for i, n in enumerate(self.data_array):
                    print(f"{i}/ {len(self.data_array)}")
                    transformed_img = transform(n).to(self.device)
                    out = self.model.forward_features(transformed_img[np.newaxis, :])
                    all_out.append(out)
            all_out = np.array(all_out)
        else:
            all_out = self.emb_prev

        all_emb = []
        for out in all_out:
            all_emb.append(out["x_norm_clstoken"])
        self.all_emb = np.array(all_emb).squeeze(1)

    def run_openclip(self):
        if self.emb_prev is None:
            all_emb = []
            with torch.no_grad():
                for i, frame in enumerate(self.data_array):
                    print(f"{i}/ {len(self.data_array)}")
                    img = Image.fromarray(frame.astype(np.uint8))
                    img_tensor = (
                        self.openclip_preprocess(img).unsqueeze(0).to(self.device)
                    )
                    features = self.model.encode_image(img_tensor)
                    all_emb.append(features.cpu().numpy())
            self.all_emb = np.concatenate(all_emb, axis=0)
        else:
            self.all_emb = self.emb_prev

    def run_dbscan(self, epsilon=None, min_samples=5):
        if epsilon is None:
            k = min_samples
            nbrs = NearestNeighbors(n_neighbors=k).fit(self.all_emb)
            distances, _ = nbrs.kneighbors(self.all_emb)
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
        cluster_labels = dbscan.fit_predict(self.all_emb)
        n_clusters = len(set(cluster_labels) - {-1})
        n_noise = (cluster_labels == -1).sum()
        print(f"Finished fitting DBSCAN: {n_clusters} clusters, {n_noise} noise points")

        self.cluster_labels = cluster_labels

        if self.viz_clusters:
            pca = PCA(n_components=2)
            out = pca.fit_transform(self.all_emb)

            plt.scatter(x=out[:, 0], y=out[:, 1], c=cluster_labels)
            plt.title("Embedding Scatter Plot (PC decomp)")
            plt.show()

        unique_labels = [l for l in np.unique(cluster_labels) if l != -1]
        centroids = np.array(
            [self.all_emb[cluster_labels == l].mean(axis=0) for l in unique_labels]
        )
        self.centroids = centroids
        m = self.n_samples_per_cluster

        closest_points = {}

        for centroid, cluster_id in zip(centroids, unique_labels):
            distances = np.linalg.norm(self.all_emb - centroid, axis=1)
            closest_indices = np.argsort(distances)[:m]
            closest_points[cluster_id] = closest_indices

        self.closest_points = closest_points

    def run_hdbscan(self, min_cluster_size=10, min_samples=None):
        hdb = HDBSCAN(
            min_cluster_size=min_cluster_size,
            min_samples=min_samples,
            store_centers="centroid",
        )
        cluster_labels = hdb.fit_predict(self.all_emb)

        n_clusters = len(set(cluster_labels) - {-1})
        n_noise = (cluster_labels == -1).sum()
        print(f"HDBSCAN: {n_clusters} clusters, {n_noise} noise points")

        self.cluster_labels = cluster_labels

        if self.viz_clusters:
            pca = PCA(n_components=2)
            pca_out = pca.fit_transform(self.all_emb)
            mask = cluster_labels != -1
            plt.scatter(
                x=pca_out[mask, 0],
                y=pca_out[mask, 1],
                c=cluster_labels[mask],
                cmap="tab20",
            )
            plt.title("Embedding Scatter Plot (PC decomp) — HDBSCAN")
            plt.show()

        unique_labels = [l for l in np.unique(cluster_labels) if l != -1]
        centroids = np.array(
            [self.all_emb[cluster_labels == l].mean(axis=0) for l in unique_labels]
        )
        self.centroids = centroids

        m = self.n_samples_per_cluster
        closest_points = {}
        for centroid, cluster_id in zip(centroids, unique_labels):
            distances = np.linalg.norm(self.all_emb - centroid, axis=1)
            closest_indices = np.argsort(distances)[:m]
            closest_points[cluster_id] = closest_indices
        self.closest_points = closest_points

    def run_knn(self):
        if self.optimize_clusters == True:
            k_values = range(1, 50)
            inertia_values = []
            for k in k_values:
                kmeans = KMeans(n_clusters=k, random_state=0, n_init="auto")
                print("Fitted k:", k)
                kmeans.fit(self.all_emb)
                inertia_values.append(kmeans.inertia_)

            knee_locator = KneeLocator(
                k_values, inertia_values, curve="convex", direction="decreasing"
            )
            optimal_k = knee_locator.knee

            plt.figure(figsize=(8, 5))
            plt.plot(k_values, inertia_values, marker="o", linestyle="--")
            plt.xlabel("Number of Clusters (k)")
            plt.ylabel("Inertia")
            plt.axvline(optimal_k, color="red", linestyle="--", label="Optimal k")
            plt.legend()
            plt.title("Elbow method for optimal number of clusters")
            plt.grid(True)
            plt.show()
            print(f"Optimal number of clusters: {optimal_k}")

        else:
            optimal_k = self.optimize_clusters

        n_clusters = optimal_k
        self.n_clusters = n_clusters
        kmeans = KMeans(n_clusters=n_clusters, random_state=0, n_init="auto")
        cluster_labels = kmeans.fit_predict(self.all_emb)

        self.cluster_labels = cluster_labels

        if self.viz_clusters:
            pca = PCA(n_components=2)
            out = pca.fit_transform(self.all_emb)

            plt.scatter(x=out[:, 0], y=out[:, 1], c=cluster_labels)
            plt.title("Embedding Scatter Plot (PC decomp)")
            plt.show()

        centroids = kmeans.cluster_centers_
        self.centroids = centroids
        m = self.n_samples_per_cluster

        closest_points = {}

        for cluster_id, centroid in enumerate(centroids):
            distances = np.linalg.norm(self.all_emb - centroid, axis=1)
            closest_indices = np.argsort(distances)[:m]
            closest_points[cluster_id] = closest_indices

        self.closest_points = closest_points

    def run_knn_km_version(self, n_components=64, k_max=50, min_k=2, reduce_dims=True):
        if reduce_dims:
            n_comp = min(n_components, self.all_emb.shape[0] - 1, self.all_emb.shape[1])
            pca = PCA(n_components=n_comp)
            self.all_emb = pca.fit_transform(self.all_emb)

        if self.optimize_clusters is True:
            k_start = max(2, min_k)
            k_range = range(k_start, min(k_max + 1, self.all_emb.shape[0]))
            sil_scores = []
            for k in k_range:
                labels = KMeans(
                    n_clusters=k, random_state=0, n_init="auto"
                ).fit_predict(self.all_emb)
                sil_scores.append(silhouette_score(self.all_emb, labels))
                print(f"k={k}  silhouette={sil_scores[-1]:.4f}")

            optimal_k = list(k_range)[int(np.argmax(sil_scores))]

            plt.figure(figsize=(8, 5))
            plt.plot(list(k_range), sil_scores, marker="o", linestyle="--")
            plt.axvline(
                optimal_k, color="red", linestyle="--", label=f"Optimal k={optimal_k}"
            )
            plt.xlabel("Number of Clusters (k)")
            plt.ylabel("Silhouette Score")
            plt.title("Silhouette Score for Optimal k")
            plt.legend()
            plt.grid(True)
            plt.show()
            print(f"Optimal k: {optimal_k}")
        else:
            optimal_k = self.optimize_clusters

        self.n_clusters = optimal_k
        kmeans = KMeans(n_clusters=optimal_k, random_state=0, n_init="auto")
        cluster_labels = kmeans.fit_predict(self.all_emb)
        self.cluster_labels = cluster_labels

        if self.viz_clusters:
            pca_viz = PCA(n_components=2)
            out = pca_viz.fit_transform(self.all_emb)
            plt.scatter(x=out[:, 0], y=out[:, 1], c=cluster_labels)
            plt.title("Embedding Scatter Plot (PC decomp)")
            plt.show()

        centroids = kmeans.cluster_centers_
        self.centroids = centroids
        m = self.n_samples_per_cluster

        closest_points = {}
        for cluster_id, centroid in enumerate(centroids):
            distances = np.linalg.norm(self.all_emb - centroid, axis=1)
            closest_indices = np.argsort(distances)[:m]
            closest_points[cluster_id] = closest_indices
        self.closest_points = closest_points

    def filter_frames(self):
        all_indices = []
        all_clusters = []

        for cluster_id in self.closest_points.keys():
            idxs = self.closest_points[cluster_id]
            all_indices.extend(idxs)
            all_clusters.extend([cluster_id] * len(idxs))

        print(f"Chose {len(all_indices)} frames")

        if self.plot_chosen_frames:
            for idx, c in zip(all_indices, all_clusters):
                plt.imshow(self.data_array[idx])
                plt.title(f"Selected frame idx={idx} cluster={c}")
                plt.show()

        self.all_indices = all_indices
        return self.data_array[all_indices]

    def pairwise_separation(self, X, metric):
        D = pairwise_distances(X, metric=metric)

        # take upper triangle excluding diagonal
        triu_idx = np.triu_indices_from(D, k=1)
        pairwise_vals = D[triu_idx]

        avg_sep = pairwise_vals.mean()

        return D, avg_sep

    def evaluate(self, n=10):
        unique_labels = [l for l in np.unique(self.cluster_labels) if l != -1]

        valid_mask = self.cluster_labels != -1
        db_score = davies_bouldin_score(
            self.all_emb[valid_mask], self.cluster_labels[valid_mask]
        )
        print(f"Davies-Bouldin Index: {db_score:.4f} (lower is better)")

        pca = PCA(n_components=2)
        pca_out = pca.fit_transform(self.all_emb[valid_mask])
        plt.figure()
        plt.scatter(
            pca_out[:, 0],
            pca_out[:, 1],
            c=self.cluster_labels[valid_mask],
            cmap="tab20",
        )
        plt.title(f"Cluster Visualization (DB Index = {db_score:.4f})")
        plt.xlabel("PC 1")
        plt.ylabel("PC 2")
        plt.tight_layout()
        plt.show()

        all_cluster = []
        for cid in unique_labels:
            mask = self.cluster_labels == cid
            data = self.all_emb[mask]
            pdist_matrix, avg = self.pairwise_separation(data, metric="manhattan")
            all_cluster.append(avg)
        all_cluster = np.array(all_cluster)

        plt.bar(x=np.arange(len(all_cluster)), height=all_cluster)
        plt.xticks(np.arange(len(all_cluster)), [str(l) for l in unique_labels])
        plt.xlabel("Cluster #")
        plt.ylabel("Avg Manhattan Distance")
        plt.title("All Clusters Inner-dist (want to be low)")
        plt.show()

        if len(self.centroids) > 1:
            centroid_pdist, avg_centroid_dist = self.pairwise_separation(
                self.centroids, metric="manhattan"
            )
            plt.imshow(centroid_pdist)
            plt.colorbar()
            plt.title(f"Pairwise Distance Matrix (Avg = {avg_centroid_dist:.3f})")
            plt.show()
        else:
            centroid_pdist = None
            print("Skipping centroid pairwise eval: fewer than 2 clusters")

        rep_frames = []
        for cid, centroid in zip(unique_labels, self.centroids):
            cluster_indices = np.where(self.cluster_labels == cid)[0]
            dists = np.linalg.norm(self.all_emb[cluster_indices] - centroid, axis=1)
            closest_idx = cluster_indices[np.argmin(dists)]
            rep_frames.append(self.data_array[closest_idx])

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
        plt.show()

        cluster_ssim_avgs = []
        for cid, centroid in zip(unique_labels, self.centroids):
            cluster_indices = np.where(self.cluster_labels == cid)[0]
            dists = np.linalg.norm(self.all_emb[cluster_indices] - centroid, axis=1)
            take = min(n, len(cluster_indices))
            closest_n_idx = cluster_indices[np.argsort(dists)[:take]]
            frames = [
                cv2.resize(self.data_array[i], (256, 256), interpolation=cv2.INTER_AREA)
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
        plt.show()

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
        plt.show()

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

    def eval_iso(self, include_outliers):
        unique_labels = [l for l in np.unique(self.cluster_labels) if l != -1]
        all_iso = []
        for cid in unique_labels:
            iso, _ = self.eval_iso_single(self.all_emb, self.cluster_labels, cid)
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
        plt.show()

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
                centroid = self.centroids[cid_idx]
                cluster_indices = np.where(self.cluster_labels == cid)[0]
                dists = np.linalg.norm(self.all_emb[cluster_indices] - centroid, axis=1)
                closest_idx = cluster_indices[np.argmin(dists)]
                plt.imshow(self.data_array[closest_idx])
                plt.title(f"Cluster {cid} — frame idx {closest_idx}")
                plt.axis("off")
                plt.show()
            except ValueError:
                print("Please enter a valid integer.")

    def eval_tightness(self):
        unique_labels = [l for l in np.unique(self.cluster_labels) if l != -1]
        avg_dists = []
        std_dists = []
        for cid, centroid in zip(unique_labels, self.centroids):
            mask = self.cluster_labels == cid
            dists = np.linalg.norm(self.all_emb[mask] - centroid, axis=1)
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
        plt.tight_layout()
        plt.show()

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

    def forward(self, eval, method="hdbscan", eval4_n=10, export=None, **kwargs):
        if self.emb_model == "dino":
            self.run_dino()
        elif self.emb_model == "openclip":
            self.run_openclip()
        if method == "hdbscan":
            self.run_hdbscan(**kwargs)
        elif method == "dbscan":
            self.run_dbscan(**kwargs)
        elif method == "kmeans":
            self.run_knn()
        elif method == "kmeans_sil":
            self.run_knn_km_version(**kwargs)

        filtered_frames = self.filter_frames()
        if export is not None:
            self.export_frames(filtered_frames, export)

        np.save("last_embeddings.npy", self.all_emb)
        np.save(self.save_path, self.data_array)

        if eval:
            self.eval_iso(include_outliers=True)
            self.eval_tightness()
            self.evaluate(n=eval4_n)

        return (
            filtered_frames,
            self.n_clusters,
            self.n_samples_per_cluster,
            self.all_indices,
        )


# data = np.load("data/surgical_1fps.npy")
# embs = np.load("data/dino_embs.npy", allow_pickle=True)

# fvi_model = DiversitySampling(
#     data_array=None  ,
#     optim_clusters=True,
#     dino_model_string="dinov2_vits14",
#     n_samples_per_cluster=2,
#     viz_clusters=True,
#     emb_prev=None,
#     plot_chosen_frames=False,
#     openclip_model_string="ViT-B-32",
#     openclip_pretrained="laion2b_s34b_b79k",
#     emb_model="openclip"
# )
# fvi_model.get_frames_from_mp4("data/mvd_whole_thing.mp4", to_rgb=True)
# fvi_model.forward(eval=True, method="kmeans", export="mvd_knn_openclip", min_k=5, reduce_dims=True)
