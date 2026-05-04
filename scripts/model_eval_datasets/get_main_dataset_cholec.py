import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from diversity_sampling_main import DiversitySampling
import numpy as np

### getting cohorts
N_VAL = 1000

# frames = np.load("cholec8080_frames.npy")
# masks = np.load("cholec8080_masks.npy")

# n_total = frames.shape[0]
# n_train = n_total - N_VAL

# all_indices = np.arange(n_total)
# train_indices = np.random.choice(n_total, size=n_train, replace=False)
# val_indices = np.setdiff1d(all_indices, train_indices)

# train_frames = frames[train_indices]
# val_frames = frames[val_indices]
# train_masks = masks[train_indices]
# val_masks = masks[val_indices]


# # print("All frames: ", len(all_indices))
# # print("Train frames: ", len(train_indices))
# # print("Val frames: ", len(val_indices))

# print("Train frames: ", train_frames.shape)
# print("Val frames: ", val_frames.shape)
# print("Train masks: ", train_masks.shape)
# print("Val masks: ", val_masks.shape)

# np.save("train_frames.npy", train_frames)
# np.save("train_masks.npy", train_masks)
# np.save("val_frames.npy", val_frames)
# np.save("val_masks.npy", val_masks)

train_frames = np.load("scripts/model_eval_datasets/data/input/train_frames.npy")
train_masks = np.load("scripts/model_eval_datasets/data/input/train_masks.npy")
val_frames = np.load("scripts/model_eval_datasets/data/input/val_frames.npy")
val_masks = np.load("scripts/model_eval_datasets/data/input/val_masks.npy")

n_train = train_frames.shape[0]

emb_prev = np.load("scripts/model_eval_datasets/data/input/last_embeddings.npy")

### sample frames
sampler = DiversitySampling(
    data_array=train_frames,
    optim_clusters=True,
    dino_model_string="dinov2_vits14",
    n_samples_per_cluster=250,
    viz_clusters=True,
    emb_prev=emb_prev,
    plot_chosen_frames=False,
    openclip_model_string="ViT-B-32",
    openclip_pretrained="laion2b_s34b_b79k",
    emb_model="openclip",
    save_path="cholec_diverse_sampled_frames.npy" # also saves embeddings as last_embeddings.npy implicitly
)
    
frames, n_clusters, n_per_cluster, all_indices = sampler.forward(eval=True, method="kmeans", export="scripts/model_eval_datasets/data/output", min_k=5, reduce_dims=True)
np.save("scripts/model_eval_datasets/data/output/all_chosen_indices_diverse.npy", all_indices)        

n_frames_sampled = n_clusters * n_per_cluster

random_sample = np.random.choice(n_train, size=n_frames_sampled, replace=False)
random_frames = train_frames[random_sample]
random_masks = train_masks[random_sample]

np.save("scripts/model_eval_datasets/data/output/random_sample_frames.npy", random_frames)
np.save("scripts/model_eval_datasets/data/output/random_sample_masks.npy", random_masks)
np.save("scripts/model_eval_datasets/data/output/diverse_sampled_masks.npy", train_masks[all_indices])
np.save("scripts/model_eval_datasets/data/output/diverse_sampled_frames.npy", train_frames[all_indices])

