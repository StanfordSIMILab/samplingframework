# Diversity Sampling Project

This repository contains code and resources for implementing and evaluating diversity sampling techniques, including Frame Variation Index (FVI) and entropy-based sampling, to optimize dataset selection for AI model training.

## Project Goals
- Reduce data annotation costs by selecting representative and informative samples.
- Experiment with FVI, entropy metrics, and hybrid approaches.
- Compare diversity sampling methods with random sampling.

Ultimately, we want to develop a model (that integrates into SMI-SAMNet) that performs well in surgical scene segmentation with minimal annotated data. We are using diversity sampling techniques to overcome the challenge that manual annotation of surgical videos is expensive and time-consuming; we want to select the most _informative frames_ for annotaiton, which creates a smaller but highly representative ground truth. 

## Workflow

The input is a dataset of _raw video files_ or image frames from surgical procedures (e.g. dAVF, MVD, EndoVis18), and we first:

1. _Extract the frames_: Convert videos into individual frames at a standardized frame rate (e.g. 10fps), and save frames as a sequence of images.
2. _Preprocess frames_: Resize frames (e.g. 224 x 224) and normalize pixels if needed

Once we have a directory of preprocessed frames ready for sampling, we can begin the diversity sampling process. The objective of diversity sampling is to _select a subset of frames_ from the dataset that represents the diversity and variability of the entire video sequence, reducing the number of frames requiring annotation whilst maximizing coverage of unique surgical scenarios.

There are three techniques that we can explore/deploy and use a combination of. The first is _Frame variation index_ where we compute the difference between consecutive frames to identify those with the most significant visual changes, using high FVI frames as annotation candidates. The second is _entropy metrics_ where we can use a pretrained model (e.g. SAM) to make predictions on all frames, and calculate the entropy of prediction to identify frames where the model is most uncertain (high entropy frames as annotation candidates). Lastly, we use _clustering_ where we apply dimensionality reduction (e.g. UMAP) and clustering (e.g. k-means) to group frames by visual similarity, and sample a representative frame from each cluster. 

Hence we obtain a subset of frames selected for annotation, optimized for diversity and informativeness. 

Now once we have the sampled, annotated frames, from the ground truth; we can train our model (in this case SOLOv2) on the annotated frames, loading the pretrained weights and fine-tuning the annotated frames. Then we evaluate the effectiveness of this model, comparing the performance on the sampled ground truth versus the random ground truth. 

## Features
- Frame Variation Index (FVI) computation.
- Entropy-based sampling.
- Experimental pipelines for testing sampling methods.
- Integration-ready scripts for deep learning models.

## Repository Structure
- `data/`: Example datasets and instructions on data preparation.
- `notebooks/`: Jupyter notebooks for exploratory analysis and experiments.
- `scripts/`: Scripts for sampling, utility functions, and processing pipelines.
- `models/`: Pretrained weights and model training scripts.
- `tests/`: Unit tests for the implemented algorithms.
- `docs/`: Detailed documentation of methods and usage.

## Getting Started
1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/diversity-sampling-project.git
   cd diversity-sampling-project

pip install -r requirements.txt

python scripts/run_sampling.py --method fvi --data_path data/examples/example_dataset1.csv

