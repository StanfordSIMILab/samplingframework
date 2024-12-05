# Diversity Sampling Project

This repository contains code and resources for implementing and evaluating diversity sampling techniques, including Frame Variation Index (FVI) and entropy-based sampling, to optimize dataset selection for AI model training.

## Project Goals
- Reduce data annotation costs by selecting representative and informative samples.
- Experiment with FVI, entropy metrics, and hybrid approaches.
- Compare diversity sampling methods with random sampling.

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

