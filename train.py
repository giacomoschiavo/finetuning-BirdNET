import torch
import numpy as np
import json
import os
from birdlib import utils
from collections import defaultdict


VM_ID = 1
INPUT_SHAPE = (256, 256)

config_file = f'/home/giacomoschiavo/finetuning-BirdNET/configs/configs_{VM_ID}.json'
with open(config_file) as f:
    configs = json.load(f)

DATASET_NAME = "dataset"
MODEL_NAME = 'CustomCNN'
DATASET_VAR = 'custom'

DATASET_PATH = f'../segments/{DATASET_NAME}'
TRAIN_PATH = f"{DATASET_PATH}/train"
VALID_PATH = f"{DATASET_PATH}/valid"
TEST_PATH = f"{DATASET_PATH}/test"
MODEL_PATH = f'./models/{MODEL_NAME}'

dataset_config = utils.create_dataset_config(TRAIN_PATH, VALID_PATH, TEST_PATH, DATASET_NAME, f'dataset_config_{DATASET_VAR}.json')
mappings = dataset_config["mappings"]

SPECS_TRAIN_PATH = f"{DATASET_PATH}/train_specs"
SPECS_VALID_PATH = f"{DATASET_PATH}/valid_specs"
SPECS_TEST_PATH = f"{DATASET_PATH}/test_specs"
os.makedirs(SPECS_TRAIN_PATH, exist_ok=True)
os.makedirs(SPECS_VALID_PATH, exist_ok=True)
os.makedirs(SPECS_TEST_PATH, exist_ok=True)
utils.specs_generation(TRAIN_PATH, SPECS_TRAIN_PATH, dataset_config['mappings'])
utils.specs_generation(VALID_PATH, SPECS_VALID_PATH, dataset_config['mappings'])
utils.specs_generation(TEST_PATH, SPECS_TEST_PATH, dataset_config['mappings'])

prev_batch_size = None
train_loader = valid_loader = test_loader = None

model_class = utils.load_model_class(MODEL_NAME)
sorted_configs = sorted(configs, key=lambda x: x['batch_size'])
for i, config in enumerate(sorted_configs):
    print(f"\n🔁 Processing config {i + 1}/{len(sorted_configs)}")
    print(f"Config: {config}")
    os.makedirs(f'models/{MODEL_NAME}/{i}', exist_ok=True)
    batch_size = config['batch_size']
    model = model_class(INPUT_SHAPE, config, len(mappings))

    if batch_size != prev_batch_size:
        print(f"\n🔁 Switching to batch_size={batch_size}")

        del train_loader, valid_loader, test_loader
        torch.cuda.empty_cache()  
        import gc
        gc.collect()  

        train_loader = utils.get_dataloader(dataset_config, split="train", batch_size=batch_size)
        valid_loader = utils.get_dataloader(dataset_config, split="valid", batch_size=batch_size)
        test_loader = utils.get_dataloader(dataset_config, split="test", batch_size=1)

        prev_batch_size = batch_size

    print(f"\n🔁 Training with config: {config}")
    trained_model = utils.train_model(train_loader, valid_loader, model, MODEL_NAME, i)
