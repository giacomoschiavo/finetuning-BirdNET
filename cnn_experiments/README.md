# CNN Baseline and Hyperparameter Search

## Overview

This repository contains the implementation of a vanilla Convolutional Neural Network (CNN) trained from scratch on the same dataset used to fine-tune BirdNET, serving as a baseline model.

## Hyperparameter Search

To explore the CNN design space, we performed a random search over 200 configurations. The hyperparameters were sampled from the following search space:

- Number of convolutional layers: {1, 2, 3, 4}  
- Kernel sizes: non-decreasing sequences from {2, 3, 4, 5, 6}  
- Number of channels: {16, 32, 64, 128}  
- Dropout rate: {0.0, 0.5}  
- Dense layer size: {32, 64, 128}  
- Batch size: {32, 64, 128}  

## Model Selection

Performance was evaluated on a separate validation set using micro average, samples average, and weighted average F1-scores. The best configuration was selected based on the average of these three metrics.

## Best Model

- 4 convolutional layers  
- Kernel sizes: [4, 5, 6, 6]  
- Number of channels: [16, 32, 64, 128]  
- Dropout: none (0.0)  
- Dense layer size: 128 units  
- Batch size: 128  

## Training Details

- Optimizer: Adam  
- Max epochs: 200  
- Early stopping patience: 15 epochs  

---

## Repository Structure

cnn_experiments/
├── configs/ # Hyperparameter configuration files
├── train.py # Training script
├── test.py # Testing and evaluation script
├── configs_generation.ipynb # Notebook for generating and managing configs
└── README.md # This file

---

Feel free to explore the configurations and scripts to reproduce or extend the hyperparameter search.
