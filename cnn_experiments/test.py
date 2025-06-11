import torch
import json
import os
from birdlib import utils, test_utils
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import classification_report
import numpy as np
import pandas as pd
import argparse


parser = argparse.ArgumentParser(description="Training script with configuration")
parser.add_argument('--config', type=int,
                    help='Config ID')
args = parser.parse_args()

VM_ID = args.config
INPUT_SHAPE = (256, 256)

config_file = f'/home/giacomoschiavo/finetuning-BirdNET/configs/configs_{VM_ID}.json'
with open(config_file) as f:
    configs = json.load(f)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DATASET_NAME = "dataset"
MODEL_NAME = 'CustomCNN'
DATASET_VAR = 'custom'

DATASET_PATH = f'/home/giacomoschiavo/segments/{DATASET_NAME}'
TRAIN_PATH = f"{DATASET_PATH}/train"
TEST_PATH = f"{DATASET_PATH}/test"
MODEL_PATH = f'/home/giacomoschiavo/models/{MODEL_NAME}'

with open(f"../utils/{DATASET_NAME}/dataset_config_{DATASET_VAR}_1.json") as f:
    dataset_config = json.load(f)

mappings = dataset_config["mappings"]
inverse_mappings = {value: key for key, value in mappings.items()}

print("🔃 Loading thresholds and validation set...")
tresh_loader = utils.get_dataloader(dataset_config, split="thresh", batch_size=1)
valid_loader = utils.get_dataloader(dataset_config, split="valid", batch_size=1)
print("✌ Loaded!")

model_class = utils.load_model_class(MODEL_NAME)
results_summary = []

sorted_configs = sorted(configs, key=lambda x: x['batch_size'])
test_species_list = os.listdir(TEST_PATH)
mlb = MultiLabelBinarizer()
mlb.fit([test_species_list])

for i, config in enumerate(sorted_configs):

    print(f"🧪 Testing model #{i}...")
    model = model_class(INPUT_SHAPE, config, len(mappings))
    model.to(device)
    saving_path = f'{MODEL_PATH}/config_{VM_ID}/{i}/checkpoint.pth'
    checkpoint = torch.load(saving_path)
    model.load_state_dict(checkpoint['model_state_dict'])

    conf_scores = test_utils.calculate_conf_scores(tresh_loader, model, dataset_config["mappings"])
    best_thresholds = test_utils.compute_best_thresholds(conf_scores)

    avg_loss, test_pred_segments = test_utils.test_model(model, dataset_config, valid_loader, inverse_mappings, thresholds=best_thresholds)
    true_segments = test_utils.get_true_segments(TEST_PATH)
    pred_segments, pred_proba = test_utils.get_pred_proba_segments(test_pred_segments)
    pred_segments, pred_proba = test_utils.fill_pred_segments(true_segments, pred_segments, pred_proba)
    
    y_true, y_pred, y_pred_proba = test_utils.binarize_test_segments(mlb, true_segments, pred_segments, pred_proba)

    np.savez(f'{MODEL_PATH}/config_{VM_ID}/{i}/results.npz', y_true=y_true, y_pred=y_pred, y_pred_proba=y_pred_proba, class_names=mlb.classes_)
    report = classification_report(y_true, y_pred, target_names=mlb.classes_, zero_division=0, output_dict=True)
    torch.cuda.empty_cache()

    with open(f"{MODEL_PATH}/config_{VM_ID}/{i}/test_pred_segments.json", "w") as f:
        json.dump(test_pred_segments, f)

    micro_f1 = report['micro avg']['f1-score']
    weighted_f1 = report['weighted avg']['f1-score']
    samples_f1 = report['samples avg']['f1-score']

    print("Micro avg: ", micro_f1)
    print("Weighted avg: ", weighted_f1)
    print("Samples avg: ", samples_f1)

    results_summary.append({
        "model_id": i,
        "config": config,
        "micro_f1": micro_f1,
        "weighted_f1": weighted_f1,
        "samples_f1": samples_f1,
        "mean_f1": (micro_f1 + weighted_f1 + samples_f1) / 3
    })

results_summary = sorted(results_summary, key=lambda x: x['mean_f1'], reverse=True)
with open(f'{MODEL_PATH}/model_ranking_config_{VM_ID}.json', 'w') as f:
    json.dump(results_summary, f, indent=4)

print("Saved model rankings!")