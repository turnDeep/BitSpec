# EI-MS-Predictor

This project provides a machine learning model to predict Electron Ionization (EI) mass spectra from chemical structures provided in MOL file format.

## Overview

The model is built using PyTorch and PyTorch Geometric. It leverages a Graph Transformer architecture to learn from molecular graphs and predict the corresponding mass spectrum. The architecture incorporates a Physics-Informed Neural Network (PINN) layer to integrate physical constraints, such as the mass conservation law.

## Project Structure

- `config/`: Configuration files for training and inference.
- `data/`: Directory for input data (MOL files) and spectrum libraries (MSP).
- `src/`: Main source code for the project.
  - `models/`: Model architecture definitions (Graph Transformer, etc.).
  - `data/`: Data processing, parsing, and loading modules.
  - `training/`: Training loops, loss functions, and optimizers.
  - `evaluation/`: Evaluation metrics and visualization tools.
  - `utils/`: Utility modules like device management and logging.
- `scripts/`: High-level scripts for training, prediction, and evaluation.
- `tests/`: Unit and integration tests.
- `checkpoints/`: Saved model checkpoints.
- `results/`: Output files, such as predicted spectra.
- `notebooks/`: Jupyter notebooks for exploration and analysis.
