# Diversity Sampling Project

This repository contains code and resources for implementing and evaluating diversity sampling techniques, including Frame Variation Index (FVI) and entropy-based sampling, to optimize dataset selection for AI model training.

## Project Goals
- Reduce data annotation costs by selecting representative and informative samples.
- Experiment with FVI, entropy metrics, and hybrid approaches.
- Compare diversity sampling methods with random sampling.

Ultimately, we want to develop a model (that integrates into SMI-SAMNet) that performs well in surgical scene segmentation with minimal annotated data. We are using diversity sampling techniques to overcome the challenge that manual annotation of surgical videos is expensive and time-consuming; we want to select the most _informative frames_ for annotaiton, which creates a smaller but highly representative ground truth. 

The input is a dataset of _raw video files_ or image frames from surgical procedures (e.g. dAVF, MVD, EndoVis18), and we first:

1. _Extract the frames_: Convert videos into individual frames at a standardized frame rate (e.g. 10fps), and save frames as a sequence of images.
2. _Preprocess frames_: Resize frames (e.g. 224 x 224) and normalize pixels if needed

Once we have a directory of preprocessed frames ready for sampling, we can begin the diversity sampling process. 

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

