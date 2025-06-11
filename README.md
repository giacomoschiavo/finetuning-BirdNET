# Fine-tuning BirdNET for Species Classification

This repository contains the code and data pipeline used in the paper _"Fine-tuning BirdNET for Automatic Ecoacoustic Monitoring of Bird"_.

## Requirements

All the required libraries and dependencies are listed in the requirements.txt file. Just run:

pip install -r requirements.txt

to get everything set up.


## Pipeline Overview

1. `01_segments_extraction.ipynb` – Extracts raw audio segments
2. `02_data_preprocessing.ipynb` – Converts segments to spectrograms
3. `03_data_augmentation.ipynb` – Applies data augmentation
4. `04_custom_model_training.ipynb` – Trains CNN model
5. `05_custom_model_testing.ipynb` – Evaluates models on test data
6. `05_birdnet_model_testing.ipynb` – Train and evaluate fine-tuned BirdNET models on test data
7. `06_model_evaluation_metrics.ipynb` – Generates metrics and plots
8. `07_best_model_selection.ipynb` – Compares models and picks best


