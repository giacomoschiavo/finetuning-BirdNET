#!/usr/bin/env python3
"""
BirdNET Model Finetuning & Testing Pipeline
===========================================
Production framework for BirdNET model finetuning, inference, and evaluation.

Automates the complete workflow: finetuning pre-trained BirdNET models,
running inference on validation/test sets, computing optimal thresholds,
and generating comprehensive performance metrics for ornithological research.
"""

import os
import csv
import json
import click
import subprocess
import numpy as np
from pathlib import Path
from collections import defaultdict
from sklearn.metrics import classification_report, f1_score
from sklearn.preprocessing import MultiLabelBinarizer
import pandas as pd
import sys
from birdlib import utils

@click.group()
def cli():
    """BirdNET Finetuning & Testing - Pipeline for model optimization."""
    pass


@cli.command()
@click.option(
    '--dataset-name',
    default='dataset',
    help='Dataset directory name.',
    show_default=True
)
@click.option(
    '--dataset-variant',
    default='test',
    help='Model variant identifier (e.g., augm_final, base_final).',
    show_default=True
)
@click.option(
    '--segments-base',
    default='segments',
    type=click.Path(),
    help='Base directory for audio segments.',
    show_default=True
)
@click.option(
    '--output-base',
    default='models/BirdNET_tuned',
    type=click.Path(),
    help='Base directory for finetuning outputs.',
    show_default=True
)
@click.option(
    '--batch-size',
    default=64,
    type=int,
    help='Training batch size.',
    show_default=True
)
@click.option(
    '--threads',
    default=1,
    type=int,
    help='Number of CPU threads.',
    show_default=True
)
@click.option(
    '--val-split',
    default=0.01,
    type=float,
    help='Validation split ratio during training.',
    show_default=True
)
@click.option(
    '--epochs',
    default=150,
    type=int,
    help='Number of training epochs.',
    show_default=True
)
@click.option(
    '--mixup/--no-mixup',
    default=True,
    help='Enable mixup data augmentation.',
    show_default=True
)
@click.option(
    '--cache-mode',
    type=click.Choice(['none', 'load', 'save']),
    default='save',
    help='Cache mode for training data.',
    show_default=True
)
@click.option(
    '--dry-run',
    is_flag=True,
    help='Print command without executing.'
)
def finetune(dataset_name, dataset_variant, segments_base, output_base,
             batch_size, threads, val_split, epochs, mixup, cache_mode, dry_run):
    """
    Finetune BirdNET pre-trained model on custom dataset.

    Executes BirdNET analyzer training with optimized hyperparameters.
    Generates TFLite model and training cache for efficient inference.
    """
    click.echo("=" * 60)
    click.echo("BIRDNET MODEL FINETUNING")
    click.echo("=" * 60)

    train_path = Path(segments_base) / 'train'
    output_dir = Path(output_base) / dataset_variant
    output_dir.mkdir(parents=True, exist_ok=True)

    model_path = output_dir / f'{dataset_variant}.tflite'
    cache_path = output_dir / f'{dataset_variant}.npz'

    click.echo(f"\n📁 Configuration:")
    click.echo(f"   Training data: {train_path}")
    click.echo(f"   Output model: {model_path}")
    click.echo(f"   Cache file: {cache_path}")
    click.echo(f"   Batch size: {batch_size}")
    click.echo(f"   Epochs: {epochs}")
    click.echo(f"   Threads: {threads}")
    click.echo(f"   Mixup: {mixup}")

    # Build command
    cmd = [
        sys.executable, '-m', 'birdnet_analyzer.train',
        str(train_path.absolute()),
        '--output', str(model_path.absolute()),
        '--batch_size', str(batch_size),
        '--threads', str(threads),
        '--val_split', str(val_split),
        '--epochs', str(epochs)
    ]

    if mixup:
        cmd.append('--mixup')

    if cache_mode != 'none':
        cmd.extend(['--cache_mode', cache_mode])
        cmd.extend(['--cache_file', str(cache_path)])

    # Display command
    cmd_str = ' '.join(cmd)
    click.echo(f"\n🚀 Executing command:")
    click.echo(f"   {cmd_str}")
    if dry_run:
        click.echo("\n⚠️  DRY RUN - Command not executed")
        return

    # Execute training
    click.echo("\n" + "=" * 60)
    click.echo("TRAINING OUTPUT")
    click.echo("=" * 60 + "\n")

    try:
        birdnet_dir = Path(__file__).parent / "BirdNET-Analyzer"
        result = subprocess.run(cmd, cwd=(birdnet_dir), check=True, text=True)
        click.echo(f"\n✅ Finetuning completed successfully!")
        click.echo(f"   Model saved: {model_path}")
        if cache_mode == 'save':
            click.echo(f"   Cache saved: {cache_path}")
    except subprocess.CalledProcessError as e:
        click.echo(f"\n❌ Finetuning failed with exit code {e.returncode}")
        raise click.Abort()


@cli.command()
@click.option(
    '--dataset-name',
    default='dataset',
    help='Dataset directory name.',
    show_default=True
)
@click.option(
    '--dataset-variant',
    default='test',
    help='Model variant identifier.',
    show_default=True
)
@click.option(
    '--segments-base',
    default='segments',
    type=click.Path(),
    help='Base directory for audio segments.',
    show_default=True
)
@click.option(
    '--model-base',
    default='models/BirdNET_tuned',
    type=click.Path(),
    help='Base directory for BirdNET models.',
    show_default=True
)
# @click.option(
#     '--species-list',
#     default='models/BirdNET_tuned/Labels.txt',
#     type=click.Path(exists=True),
#     help='Species list file for BirdNET.',
#     show_default=True
# )
@click.option(
    '--split',
    type=click.Choice(['valid', 'test', 'both']),
    default='both',
    help='Dataset split to analyze.',
    show_default=True
)
@click.option(
    '--min-conf',
    default=0.05,
    type=float,
    help='Minimum confidence threshold.',
    show_default=True
)
@click.option(
    '--sensitivity',
    default=1.0,
    type=float,
    help='BirdNET sensitivity (0.5-1.5).',
    show_default=True
)
@click.option(
    '--threads',
    default=1,
    type=int,
    help='Number of CPU threads.',
    show_default=True
)
@click.option(
    '--dry-run',
    is_flag=True,
    help='Print commands without executing.'
)
def analyze(dataset_name, dataset_variant, segments_base, model_base,
            split, min_conf, sensitivity, threads, dry_run):
    """
    Run BirdNET inference on validation/test datasets.

    Executes BirdNET analyzer to generate predictions with the finetuned model.
    Creates selection tables with confidence scores for threshold optimization.
    """
    click.echo("=" * 60)
    click.echo("BIRDNET INFERENCE ANALYSIS")
    click.echo("=" * 60)

    model_path = Path(model_base) / dataset_variant / f'{dataset_variant}.tflite'
    species_list = Path(model_base) / dataset_variant /  f'{dataset_variant}_Labels.txt'

    if not model_path.exists():
        click.echo(f"\n❌ Error: Model not found: {model_path}")
        click.echo("   Run 'finetune' command first")
        raise click.Abort()

    click.echo(f"\n📊 Configuration:")
    click.echo(f"   Model: {model_path}")
    click.echo(f"   Species list: {species_list}")
    click.echo(f"   Min confidence: {min_conf}")
    click.echo(f"   Sensitivity: {sensitivity}")
    click.echo(f"   Threads: {threads}")

    splits_to_process = []
    if split in ['valid', 'both']:
        splits_to_process.append('valid')
    if split in ['test', 'both']:
        splits_to_process.append('test')

    for split_name in splits_to_process:
        click.echo(f"\n{'='*60}")
        click.echo(f"Processing {split_name.upper()} split")
        click.echo(f"{'='*60}")

        input_path = Path(segments_base) / split_name
        output_path = Path(model_base) / dataset_variant / split_name
        output_path.mkdir(parents=True, exist_ok=True)

        cmd = [
            sys.executable, '-m', 'birdnet_analyzer.analyze',
            str(input_path.absolute()),
            '--output', str(output_path.absolute()),
            '--slist', str(species_list.absolute()),
            '--threads', str(threads),
            '--combine_results',
            '--min_conf', str(min_conf),
            '--classifier', str(model_path.absolute()),
            '--sensitivity', str(sensitivity)
        ]

        cmd_str = ' '.join(cmd)
        click.echo(f"\n🚀 Command:")
        click.echo(f"   {cmd_str}")

        if dry_run:
            click.echo(f"   ⚠️  DRY RUN - Skipping execution")
            continue

        click.echo(f"\n🔍 Analyzing {split_name} data...")
        try:
            birdnet_dir = Path(__file__).parent / "BirdNET-Analyzer"
            result = subprocess.run(cmd, cwd=birdnet_dir, check=True, text=True)
            click.echo(f"   ✅ Analysis completed for {split_name}")

            # Check for selection table
            selection_table = output_path / 'BirdNET_SelectionTable.txt'
            if selection_table.exists():
                # Count predictions
                with open(selection_table) as f:
                    line_count = sum(1 for _ in f) - 1  # Exclude header
                click.echo(f"   📄 Selection table: {line_count} predictions")

        except subprocess.CalledProcessError as e:
            click.echo(f"\n❌ Analysis failed for {split_name}")
            click.echo(f"   Error: {e.stderr}")
            raise click.Abort()

    click.echo(f"\n✅ Inference analysis completed!")


@cli.command()
@click.option(
    '--dataset-name',
    default='dataset',
    help='Dataset directory name.',
    show_default=True
)
@click.option(
    '--dataset-variant',
    default='test',
    help='Model variant identifier.',
    show_default=True
)
@click.option(
    '--model-base',
    default='models/BirdNET_tuned',
    type=click.Path(),
    help='Base directory for BirdNET models.',
    show_default=True
)
@click.option(
    '--utils-dir',
    default='utils',
    type=click.Path(),
    help='Directory containing dataset config.',
    show_default=True
)
# @click.option(
#     '--species-mapping',
#     default='utils/species_dict_map.json',
#     type=click.Path(exists=True),
#     help='Species name mapping file.',
#     show_default=True
# )
@click.option(
    '--num-thresholds',
    default=200,
    type=int,
    help='Number of thresholds to evaluate.',
    show_default=True
)
@click.option(
    '--min-thresh',
    default=0.01,
    type=float,
    help='Minimum threshold value.',
    show_default=True
)
@click.option(
    '--max-thresh',
    default=0.95,
    type=float,
    help='Maximum threshold value.',
    show_default=True
)
@click.option(
    '--default-thresh',
    default=0.15,
    type=float,
    help='Default threshold for species without optimization.',
    show_default=True
)
def optimize_thresholds(dataset_name, dataset_variant, model_base, utils_dir, num_thresholds, min_thresh, max_thresh,
                       default_thresh):
    """
    Compute optimal confidence thresholds per species.

    Analyzes validation set predictions to find species-specific thresholds
    that maximize F1-score. Critical for multi-label classification performance.
    """
    click.echo("=" * 60)
    click.echo("THRESHOLD OPTIMIZATION")
    click.echo("=" * 60)

    results_path = Path(model_base) / dataset_variant
    valid_table = results_path / 'valid' / 'BirdNET_SelectionTable.txt'

    if not valid_table.exists():
        click.echo(f"\n❌ Error: Validation results not found: {valid_table}")
        click.echo("   Run 'analyze --split valid' first")
        raise click.Abort()

    click.echo(f"\n📊 Loading validation predictions...")
    click.echo(f"   Selection table: {valid_table}")

    # Load dataset config
    config_path = Path(utils_dir) / dataset_name / 'dataset_config_custom.json'
    with open(config_path) as f:
        dataset_config = json.load(f)
    class_names = list(dataset_config['mappings'].keys())

    # Load species mapping
    labels_path = results_path / f'{dataset_variant}_Labels.txt'
    species_dict = utils.get_species_dict(labels_path)
    inv_species_dict = {v: k for k, v in species_dict.items()}

    click.echo(f"   Classes: {len(class_names)}")

    # Collect confidence scores
    click.echo(f"\n🔍 Collecting confidence scores...")
    conf_scores = {}

    with open(valid_table) as f:
        reader = csv.DictReader(f, delimiter='\t')
        for row in reader:
            file_path = row['Begin Path']
            pred_species = row['Common Name']

            # Map species name if needed
            if pred_species in inv_species_dict:
                pred_species = '_'.join([inv_species_dict[pred_species], pred_species])

            true_species = file_path.split('/')[-2]
            confidence = float(row['Confidence'])

            if pred_species not in conf_scores:
                conf_scores[pred_species] = []

            is_correct = (pred_species == true_species)
            conf_scores[pred_species].append((confidence, is_correct))

    click.echo(f"   ✓ Processed predictions for {len(conf_scores)} species")

    # Optimize thresholds
    click.echo(f"\n⚙️  Optimizing thresholds (grid size: {num_thresholds})...")
    click.echo(f"   Range: [{min_thresh}, {max_thresh}]")
    click.echo("")

    best_thresholds = {}
    threshold_results = []

    for species, values in conf_scores.items():
        probs, truths = zip(*values)
        probs = np.array(probs)
        truths = np.array(truths).astype(int)

        best_thresh = default_thresh
        best_f1 = 0.0

        # Grid search
        for thresh in np.linspace(min_thresh, max_thresh, num_thresholds):
            preds = (probs >= thresh).astype(int)
            f1 = f1_score(truths, preds, zero_division=0)
            if f1 > best_f1:
                best_f1 = f1
                best_thresh = thresh

        best_thresholds[species] = float(best_thresh)
        threshold_results.append({
            'species': species,
            'threshold': best_thresh,
            'f1_score': best_f1,
            'num_samples': len(values)
        })

        click.echo(f"   📊 {species:<50} → {best_thresh:.3f} (F1: {best_f1:.3f})")

    # Save thresholds
    thresholds_path = results_path / 'optimized_thresholds.json'
    with open(thresholds_path, 'w') as f:
        json.dump(best_thresholds, f, indent=2)

    # Save detailed results
    results_df = pd.DataFrame(threshold_results)
    results_df = results_df.sort_values('f1_score', ascending=False)
    results_csv = results_path / 'threshold_optimization_results.csv'
    results_df.to_csv(results_csv, index=False)

    click.echo(f"\n💾 Results saved:")
    click.echo(f"   Thresholds: {thresholds_path}")
    click.echo(f"   Details: {results_csv}")

    # Summary statistics
    click.echo(f"\n📈 Summary:")
    click.echo(f"   Mean F1: {results_df['f1_score'].mean():.3f}")
    click.echo(f"   Median F1: {results_df['f1_score'].median():.3f}")
    click.echo(f"   Mean threshold: {results_df['threshold'].mean():.3f}")

    click.echo(f"\n✅ Threshold optimization completed!")


@cli.command()
@click.option(
    '--dataset-name',
    default='dataset',
    help='Dataset directory name.',
    show_default=True
)
@click.option(
    '--dataset-variant',
    default='test',
    help='Model variant identifier.',
    show_default=True
)
@click.option(
    '--segments-base',
    default='segments',
    type=click.Path(),
    help='Base directory for audio segments.',
    show_default=True
)
@click.option(
    '--model-base',
    default='models/BirdNET_tuned',
    type=click.Path(),
    help='Base directory for BirdNET models.',
    show_default=True
)
@click.option(
    '--utils-dir',
    default='utils',
    type=click.Path(),
    help='Directory containing dataset config.',
    show_default=True
)
# @click.option(
#     '--species-mapping',
#     default='utils/species_dict_map.json',
#     type=click.Path(exists=True),
#     help='Species name mapping file.',
#     show_default=True
# )
@click.option(
    '--default-thresh',
    default=0.15,
    type=float,
    help='Default threshold for unmapped species.',
    show_default=True
)
def evaluate(dataset_name, dataset_variant, segments_base, model_base, utils_dir, default_thresh):
    """
    Evaluate model performance on test set with optimized thresholds.

    Generates comprehensive metrics: precision, recall, F1 (micro/macro/weighted/samples)
    with per-class breakdown. Uses optimized thresholds from validation set.
    """
    click.echo("=" * 60)
    click.echo("MODEL EVALUATION")
    click.echo("=" * 60)

    results_path = Path(model_base) / dataset_variant
    test_table = results_path / 'test' / 'BirdNET_SelectionTable.txt'
    test_path = Path(segments_base) / 'test'

    if not test_table.exists():
        click.echo(f"\n❌ Error: Test results not found: {test_table}")
        click.echo("   Run 'analyze --split test' first")
        raise click.Abort()
    
    labels_path = results_path / f'{dataset_variant}_Labels.txt'
    species_dict = utils.get_species_dict(labels_path)
    inv_species_dict = {v: k for k, v in species_dict.items()}

    # Load optimized thresholds
    thresholds_path = results_path / 'optimized_thresholds.json'
    if thresholds_path.exists():
        with open(thresholds_path) as f:
            best_thresholds = json.load(f)
        click.echo(f"\n📂 Using optimized thresholds: {thresholds_path}")
    else:
        click.echo(f"\n⚠️  No optimized thresholds found, using default: {default_thresh}")
        best_thresholds = {}

    # Load dataset config
    config_path = Path(utils_dir) / dataset_name / 'dataset_config_custom.json'
    with open(config_path) as f:
        dataset_config = json.load(f)

    # Get test species
    test_species_list = os.listdir(test_path)
    mlb = MultiLabelBinarizer()
    mlb.fit([test_species_list])

    click.echo(f"\n📊 Configuration:")
    click.echo(f"   Test classes: {len(test_species_list)}")
    click.echo(f"   Selection table: {test_table}")

    # Build predicted segments
    click.echo(f"\n🔍 Processing predictions...")
    pred_segments_proba = {}

    with open(test_table) as f:
        reader = csv.DictReader(f, delimiter='\t')
        for row in reader:
            file_path = row['Begin Path']
            audio_name = os.path.basename(file_path)
            only_audio_name = '_'.join(audio_name.split('_')[:2]) + '.WAV'
            segm = '_'.join(audio_name.split('.')[0].split('_')[-2:])

            pred_species = row['Common Name']
            if pred_species in inv_species_dict:
                pred_species = '_'.join([inv_species_dict[pred_species], pred_species])

            confidence = float(row['Confidence'])

            pred_segments_proba.setdefault(only_audio_name, {})
            threshold = best_thresholds.get(pred_species, default_thresh)

            if confidence >= threshold:
                pred_segments_proba[only_audio_name].setdefault(segm, {})

                if 'None' in pred_segments_proba[only_audio_name][segm]:
                    continue

                if pred_species == 'None':
                    pred_segments_proba[only_audio_name][segm] = {'None': confidence}
                else:
                    pred_segments_proba[only_audio_name][segm].update({pred_species: confidence})

    # Build ground truth segments
    click.echo(f"   Building ground truth...")
    true_segments = defaultdict(dict)

    for species in os.listdir(test_path):
        if species not in test_species_list:
            continue

        species_path = Path(test_path) / species
        for audio_file in os.listdir(species_path):
            audio = audio_file.split('.')[0]
            parts = audio.split('_')
            date, time, segm1, segm2 = parts[0], parts[1], parts[2], parts[3]
            audio_name = f'{date}_{time}.WAV'
            segm = f'{segm1}_{segm2}'

            if segm not in true_segments[audio_name]:
                true_segments[audio_name][segm] = []
            true_segments[audio_name][segm].append(species)

    # Ensure all ground truth segments have prediction entries
    for audio in true_segments.keys():
        pred_segments_proba.setdefault(audio, {})
        for segm in true_segments[audio].keys():
            pred_segments_proba[audio].setdefault(segm, {})

    # Extract labels and probabilities
    click.echo(f"   Preparing evaluation data...")
    pred_segments = {}
    pred_proba = {}

    for audio, segments in pred_segments_proba.items():
        pred_segments.setdefault(audio, {})
        pred_proba.setdefault(audio, {})
        for segm, labels in segments.items():
            pred_segments[audio][segm] = list(labels.keys())
            pred_proba[audio][segm] = list(labels.values())

    # Build arrays for sklearn
    y_pred = []
    y_true = []
    y_pred_proba = []

    for audio in sorted(pred_segments.keys()):
        for segment in sorted(pred_segments[audio].keys()):
            true_labels = true_segments[audio].get(segment, [])
            pred_labels = pred_segments[audio].get(segment, [])
            proba_values = pred_proba[audio].get(segment, [])

            y_true_vec = mlb.transform([true_labels])[0]
            y_pred_vec = mlb.transform([pred_labels])[0]

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

    # Compute metrics
    click.echo(f"\n📊 Computing evaluation metrics...")

    report = classification_report(
        y_true, y_pred,
        target_names=list(mlb.classes_),
        output_dict=True,
        zero_division=0
    )

    # Display results
    click.echo(f"\n{'='*80}")
    click.echo("CLASSIFICATION REPORT")
    click.echo(f"{'='*80}")

    # Per-class results
    click.echo(f"\n{'Species':<50} {'Precision':>10} {'Recall':>10} {'F1-Score':>10} {'Support':>10}")
    click.echo("-" * 90)

    for species in mlb.classes_:
        if species in report:
            metrics = report[species]
            click.echo(f"{species:<50} {metrics['precision']:>10.3f} {metrics['recall']:>10.3f} "
                      f"{metrics['f1-score']:>10.3f} {int(metrics['support']):>10}")

    # Aggregate metrics
    click.echo("-" * 90)
    for avg_type in ['micro avg', 'macro avg', 'weighted avg', 'samples avg']:
        if avg_type in report:
            metrics = report[avg_type]
            click.echo(f"{avg_type:<50} {metrics['precision']:>10.3f} {metrics['recall']:>10.3f} "
                      f"{metrics['f1-score']:>10.3f} {int(metrics['support']):>10}")

    # Save detailed report
    report_df = pd.DataFrame(report).T
    report_csv = results_path / 'test_evaluation_report.csv'
    report_df.to_csv(report_csv)

    # Save predictions
    pred_json = results_path / 'test_pred_segments.json'
    with open(pred_json, 'w') as f:
        json.dump(pred_segments_proba, f, indent=2)

    click.echo(f"\n💾 Results saved:")
    click.echo(f"   Classification report: {report_csv}")
    click.echo(f"   Predictions: {pred_json}")

    # Summary
    click.echo(f"\n📈 Summary:")
    click.echo(f"   Micro F1: {report['micro avg']['f1-score']:.4f}")
    click.echo(f"   Macro F1: {report['macro avg']['f1-score']:.4f}")
    click.echo(f"   Weighted F1: {report['weighted avg']['f1-score']:.4f}")
    click.echo(f"   Total samples: {int(report['micro avg']['support'])}")

    click.echo(f"\n✅ Evaluation completed!")


@cli.command()
@click.option(
    '--dataset-name',
    default='dataset',
    help='Dataset directory name.',
    show_default=True
)
@click.option(
    '--dataset-variant',
    default='test',
    help='Model variant identifier.',
    show_default=True
)
@click.option(
    '--segments-base',
    default='segments',
    type=click.Path(),
    help='Base directory for audio segments.',
    show_default=True
)
@click.option(
    '--model-base',
    default='models/BirdNET_tuned',
    type=click.Path(),
    help='Base directory for BirdNET models.',
    show_default=True
)
# @click.option(
#     '--species-list',
#     default='models/BirdNET_tuned/Labels.txt',
#     type=click.Path(exists=True),
#     help='Species list file.',
#     show_default=True
# )
@click.option(
    '--utils-dir',
    default='utils',
    type=click.Path(),
    help='Directory containing dataset config.',
    show_default=True
)
@click.option(
    '--species-mapping',
    default='utils/species_dict_map.json',
    type=click.Path(exists=True),
    help='Species name mapping file.',
    show_default=True
)
@click.option(
    '--batch-size',
    default=64,
    type=int,
    help='Training batch size.',
    show_default=True
)
@click.option(
    '--epochs',
    default=150,
    type=int,
    help='Number of training epochs.',
    show_default=True
)
@click.option(
    '--threads',
    default=16,
    type=int,
    help='Number of CPU threads.',
    show_default=True
)
@click.option(
    '--min-conf',
    default=0.05,
    type=float,
    help='Minimum confidence threshold for analysis.',
    show_default=True
)
@click.option(
    '--sensitivity',
    default=1.0,
    type=float,
    help='BirdNET sensitivity.',
    show_default=True
)
def full_pipeline(dataset_name, dataset_variant, segments_base, model_base,
                 utils_dir, species_mapping, batch_size, epochs, threads, min_conf, sensitivity):
    """
    Execute complete pipeline: finetune → analyze → optimize → evaluate.

    Automated end-to-end workflow for production model development.
    Ideal for batch experiments and systematic hyperparameter exploration.
    """
    click.echo("=" * 60)
    click.echo("FULL PIPELINE EXECUTION")
    click.echo("=" * 60)
    click.echo(f"\n🚀 Running complete workflow for variant: {dataset_variant}")
    click.echo(f"   Dataset: {dataset_name}")
    click.echo("")

    ctx = click.get_current_context()

    # Step 1: Finetune
    click.echo("\n" + "🔵" * 30)
    click.echo("STEP 1/4: FINETUNING")
    click.echo("🔵" * 30)
    ctx.invoke(finetune,
               dataset_name=dataset_name,
               dataset_variant=dataset_variant,
               segments_base=segments_base,
               output_base=model_base,
               batch_size=batch_size,
               epochs=epochs,
               threads=threads)

    # Step 2: Analyze
    species_list = Path(model_base) / dataset_variant /  f'{dataset_variant}_Labels.txt'
    click.echo("\n" + "🔵" * 30)
    click.echo("STEP 2/4: INFERENCE ANALYSIS")
    click.echo("🔵" * 30)
    ctx.invoke(analyze,
               dataset_name=dataset_name,
               dataset_variant=dataset_variant,
               segments_base=segments_base,
               model_base=model_base,
               species_list=species_list,
               split='both',
               min_conf=min_conf,
               sensitivity=sensitivity,
               threads=threads)

    # Step 3: Optimize thresholds
    click.echo("\n" + "🔵" * 30)
    click.echo("STEP 3/4: THRESHOLD OPTIMIZATION")
    click.echo("🔵" * 30)
    ctx.invoke(optimize_thresholds,
               dataset_name=dataset_name,
               dataset_variant=dataset_variant,
               model_base=model_base,
               utils_dir=utils_dir,
               species_mapping=species_mapping)

    # Step 4: Evaluate
    click.echo("\n" + "🔵" * 30)
    click.echo("STEP 4/4: MODEL EVALUATION")
    click.echo("🔵" * 30)
    ctx.invoke(evaluate,
               dataset_name=dataset_name,
               dataset_variant=dataset_variant,
               segments_base=segments_base,
               model_base=model_base,
               utils_dir=utils_dir,
               species_mapping=species_mapping)

    click.echo("\n" + "=" * 60)
    click.echo("✅ FULL PIPELINE COMPLETED SUCCESSFULLY!")
    click.echo("=" * 60)

    # Summary
    results_path = Path(model_base) / dataset_variant
    click.echo(f"\n📁 Output directory: {results_path}")
    click.echo(f"\n📄 Generated files:")
    click.echo(f"   - {dataset_variant}.tflite - Finetuned model")
    click.echo(f"   - optimized_thresholds.json - Per-species thresholds")
    click.echo(f"   - test_evaluation_report.csv - Performance metrics")
    click.echo(f"   - test_pred_segments.json - Detailed predictions")


if __name__ == '__main__':
    cli()
