#### TESTING
import os
import torch
import torch.nn as nn
import numpy as np
from collections import defaultdict
from sklearn.metrics import f1_score

def calculate_conf_scores(valid_loader, model, mappings):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    model.to(device)
    
    conf_scores = defaultdict(list)

    with torch.no_grad():
        for mel_spec, _, file_path in valid_loader:
            mel_spec = mel_spec.to(device)

            # Estraggo la specie corretta dal path
            correct_species = file_path[0].split("/")[-2]
            outputs = model(mel_spec)
            probs = torch.sigmoid(outputs)[0].cpu().numpy()

            for i, prob in enumerate(probs):
                species_name = list(mappings.keys())[i]
                is_correct = species_name == correct_species
                conf_scores[species_name].append((prob, is_correct))

    return conf_scores

def compute_best_thresholds(conf_scores, num_thresholds=200, min_thresh=0.01, max_thresh=0.95):
    thresholds = {}

    for species, values in conf_scores.items():
        probs, truths = zip(*values)
        probs = np.array(probs)
        truths = np.array(truths).astype(int)

        best_thresh = 0.15
        best_f1 = 0.0

        for thresh in np.linspace(min_thresh, max_thresh, num_thresholds):
            preds = (probs >= thresh).astype(int)
            f1 = f1_score(truths, preds, zero_division=0)
            if f1 > best_f1:
                best_f1 = f1
                best_thresh = thresh

        thresholds[species] = best_thresh

    return thresholds

import os

def test_model(model, dataset_config, test_loader, inverse_mappings, thresholds=0.2):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n🧬 Advanced testing on: {device}")
    test_pred_segments = {}

    model.eval()
    criterion = nn.BCEWithLogitsLoss()
    class_names = list(dataset_config['mappings'].keys())
    total_loss = 0.0

    use_custom_threshold = isinstance(thresholds, dict)

    with torch.no_grad():
        for mel_spec, labels, file_path in test_loader:
            basename = os.path.splitext(file_path[0].split("/")[-1])[0]
            date, time, segm1, segm2 = basename.split("_")
            audio_name = "_".join([date, time]) + ".WAV"
            segm = "_".join([segm1, segm2])
            test_pred_segments.setdefault(audio_name, {})

            mel_spec = mel_spec.to(device)
            labels = labels.to(device)

            outputs = model(mel_spec)
            loss = criterion(outputs, labels)
            total_loss += loss.item()

            probs = torch.sigmoid(outputs)

            if use_custom_threshold:
                batch_preds = torch.zeros_like(probs)
                for i, class_name in enumerate(class_names):
                    thresh = thresholds.get(class_name, 0.5)
                    batch_preds[:, i] = (probs[:, i] > thresh).float()
            else:
                batch_preds = (probs > thresholds).float()

            correct_probs = probs * batch_preds
            if segm not in test_pred_segments:
                test_pred_segments[audio_name][segm] = {}
                
            conf_scores = {
                inverse_mappings[i]: correct_probs[0, i].item()
                for i in range(correct_probs.size(1))
                if correct_probs[0, i].item() != 0
            }
            test_pred_segments[audio_name][segm].update(conf_scores)

    avg_loss = total_loss / len(test_loader)
    return avg_loss, test_pred_segments

from collections import defaultdict
import os

def get_true_segments(test_path):
    test_species_list = os.listdir(test_path)
    true_segments = defaultdict(dict)
    for species in test_species_list:
        for audio in os.listdir(os.path.join(test_path, species)):
            audio = audio.split('.')[0]
            date, time, segm1, segm2 = audio.split('_')
            audio_name = '_'.join([date, time]) + '.WAV'
            segm = '_'.join([segm1, segm2])
            if segm not in true_segments[audio_name]:
                true_segments[audio_name][segm] = []
            true_segments[audio_name][segm].extend([species])
    return true_segments

def get_pred_proba_segments(test_pred_segments):
    pred_segments = {}
    pred_proba = {}

    for audio, segments in test_pred_segments.items():
        pred_segments.setdefault(audio, {})
        pred_proba.setdefault(audio, {})
        for segm, labels in segments.items():
            pred_segments[audio].setdefault(segm, {})
            pred_segments[audio][segm] = list(labels.keys())
            pred_proba[audio].setdefault(segm, {})
            pred_proba[audio][segm] = list(labels.values())
    return pred_segments, pred_proba

def fill_pred_segments(true_segments, pred_segments, pred_proba):
    for audio in true_segments.keys():
        if audio in pred_segments:
            for segm in true_segments[audio].keys():
                if segm not in pred_segments[audio]:
                    pred_segments[audio][segm] = {}
                    pred_proba[audio][segm] = {}

    return pred_segments, pred_proba

def binarize_test_segments(mlb, true_segments, pred_segments, pred_proba):
    y_pred = []
    y_true = []
    y_pred_proba = []

    for audio in pred_segments:
        for segment in sorted(pred_segments[audio].keys()):
            true_labels = true_segments[audio].get(segment, [])
            pred_labels = pred_segments[audio].get(segment, [])
            proba_values = pred_proba[audio].get(segment, [])

            y_true_vec = mlb.transform([true_labels])[0]  # 1D array
            y_pred_vec = mlb.transform([pred_labels])[0]  # 1D array

            proba_vec = np.zeros(len(mlb.classes_))
            for label, score in zip(pred_labels, proba_values):
                if label in mlb.classes_:
                    idx = list(mlb.classes_).index(label)
                    proba_vec[idx] = score

            y_true.append(y_true_vec)
            y_pred.append(y_pred_vec)
            y_pred_proba.append(proba_vec)

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_pred_proba = np.array(y_pred_proba)

    return y_true, y_pred, y_pred_proba
