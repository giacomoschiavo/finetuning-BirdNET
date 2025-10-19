#!/usr/bin/env python3
"""
BirdNET Custom Model Training Pipeline
======================================
Training framework for custom CNN architectures.

Implements complete training workflow including dataset configuration,
spectrogram generation, model training with early stopping, and
comprehensive evaluation metrics for ornithological research.
"""

import json
import click
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
from birdlib import utils


@click.group()
def cli():
    """BirdNET Custom Model Training - ML pipeline for ornithological research."""
    pass


@cli.command()
@click.option(
    '--dataset-name',
    default='dataset',
    help='Name of the dataset directory.',
    show_default=True
)
@click.option(
    '--model-name',
    default='CustomCNN',
    help='Model architecture name.',
    show_default=True
)
@click.option(
    '--dataset-variant',
    default='custom',
    help='Dataset variant identifier.',
    show_default=True
)
@click.option(
    '--config-file',
    type=click.Path(exists=True),
    help='Model configuration JSON file (overrides auto-config).'
)
@click.option(
    '--num-conv-layers',
    default=4,
    type=int,
    help='Number of convolutional layers.',
    show_default=True
)
@click.option(
    '--channels',
    default='16,32,64,128',
    help='Comma-separated channel sizes.',
    show_default=True
)
@click.option(
    '--kernel-sizes',
    default='4,5,6,6',
    help='Comma-separated kernel sizes.',
    show_default=True
)
@click.option(
    '--dense-hidden',
    default=128,
    type=int,
    help='Dense layer hidden units.',
    show_default=True
)
@click.option(
    '--dropout',
    default=0.0,
    type=float,
    help='Dropout probability.',
    show_default=True
)
@click.option(
    '--batch-size',
    default=128,
    type=int,
    help='Training batch size.',
    show_default=True
)
@click.option(
    '--epochs',
    default=200,
    type=int,
    help='Maximum training epochs.',
    show_default=True
)
@click.option(
    '--learning-rate',
    default=1e-4,
    type=float,
    help='Initial learning rate.',
    show_default=True
)
@click.option(
    '--lr-patience',
    default=3,
    type=int,
    help='LR scheduler patience (epochs).',
    show_default=True
)
@click.option(
    '--early-stop-patience',
    default=10,
    type=int,
    help='Early stopping patience (epochs).',
    show_default=True
)
@click.option(
    '--print-freq',
    default=100,
    type=int,
    help='Batch logging frequency.',
    show_default=True
)
@click.option(
    '--resume/--no-resume',
    default=False,
    help='Resume from checkpoint if exists.',
    show_default=True
)
@click.option(
    '--gpu-id',
    default=0,
    type=int,
    help='GPU device ID (-1 for CPU).',
    show_default=True
)
def train(dataset_name, model_name, dataset_variant, config_file, num_conv_layers,
          channels, kernel_sizes, dense_hidden, dropout, batch_size, epochs,
          learning_rate, lr_patience, early_stop_patience, print_freq, resume, gpu_id):
    """
    Train custom CNN model on prepared dataset.

    Implements complete training loop with validation, early stopping,
    learning rate scheduling, and checkpoint management.
    """
    click.echo("=" * 60)
    click.echo("MODEL TRAINING PIPELINE")
    click.echo("=" * 60)

    # Setup device
    if gpu_id >= 0 and torch.cuda.is_available():
        device = torch.device(f"cuda:{gpu_id}")
        click.echo(f"\n🖥️  Using GPU: {torch.cuda.get_device_name(gpu_id)}")
    else:
        device = torch.device("cpu")
        click.echo(f"\n🖥️  Using CPU")

    # Load dataset config
    dataset_config_path = Path('utils') / dataset_name / f'dataset_config_{dataset_variant}.json'
    if not dataset_config_path.exists():
        click.echo(f"\n❌ Error: Dataset config not found: {dataset_config_path}")
        return

    with open(dataset_config_path) as f:
        dataset_config = json.load(f)

    num_classes = len(dataset_config['mappings'])
    click.echo(f"\n📊 Dataset: {dataset_name} (variant: {dataset_variant})")
    click.echo(f"   Classes: {num_classes}")
    click.echo(f"   Training samples: {len([s for s in dataset_config['samples'] if s['split'] == 'train'])}")

    # Build model config
    if config_file:
        click.echo(f"\n⚙️  Loading model config from: {config_file}")
        with open(config_file) as f:
            configs = json.load(f)
        model_config = configs[0]['config']
    else:
        model_config = {
            'num_conv_layers': num_conv_layers,
            'channels': [int(c) for c in channels.split(',')],
            'kernel_sizes': [int(k) for k in kernel_sizes.split(',')],
            'dense_hidden': dense_hidden,
            'dropout': dropout,
            'batch_size': batch_size
        }

    click.echo(f"\n🏗️  Model architecture: {model_name}")
    click.echo(f"   Convolutional layers: {model_config['num_conv_layers']}")
    click.echo(f"   Channels: {model_config['channels']}")
    click.echo(f"   Kernel sizes: {model_config['kernel_sizes']}")
    click.echo(f"   Dense hidden: {model_config['dense_hidden']}")
    click.echo(f"   Dropout: {model_config['dropout']}")

    # Load model class
    click.echo(f"\n🔧 Building model...")
    model_class = utils.load_model_class(model_name)
    model = model_class((256, 256), model_config, num_classes)
    model.to(device)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    click.echo(f"   Total parameters: {total_params:,}")
    click.echo(f"   Trainable parameters: {trainable_params:,}")

    # Setup training
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=lr_patience
    )

    # Model save path
    model_dir = Path('models') / model_name / dataset_variant
    model_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = model_dir / 'checkpoint.pth'

    # Initialize training state
    history_loss = []
    history_valid_loss = []
    best_loss = float('inf')
    start_epoch = 0

    # Resume from checkpoint
    if resume and checkpoint_path.exists():
        click.echo(f"\n📂 Loading checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        history_loss = checkpoint['history_loss']
        history_valid_loss = checkpoint['history_valid_loss']
        best_loss = checkpoint['best_loss']
        start_epoch = len(history_loss)
        click.echo(f"   Resumed from epoch {start_epoch}")
        click.echo(f"   Best validation loss: {best_loss:.5f}")

    # Load data
    click.echo(f"\n📦 Loading training data...")
    train_loader, valid_loader = utils.get_dataloader(
        dataset_config,
        split='train',
        batch_size=model_config['batch_size'],
        split_ratio=0.15
    )
    click.echo(f"   Training batches: {len(train_loader)}")
    click.echo(f"   Validation batches: {len(valid_loader)}")

    # Training loop
    click.echo(f"\n🚀 Starting training...")
    click.echo(f"   Epochs: {epochs}")
    click.echo(f"   Batch size: {model_config['batch_size']}")
    click.echo(f"   Learning rate: {learning_rate:.1e}")
    click.echo(f"   Early stop patience: {early_stop_patience}")
    click.echo("")

    early_stop_counter = 0

    for epoch in range(start_epoch, epochs):
        # Training phase
        model.train()
        running_loss = 0.0

        click.echo(f"{'='*60}")
        click.echo(f"🎯 Epoch {epoch + 1}/{epochs}")
        click.echo(f"{'='*60}")

        with tqdm(train_loader, desc="Training", unit="batch") as pbar:
            for batch_idx, (mel_spec, labels, _) in enumerate(pbar):
                mel_spec = mel_spec.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()
                outputs = model(mel_spec)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()

                pbar.set_postfix({'loss': f'{loss.item():.4f}'})

                if batch_idx % print_freq == 0 and batch_idx > 0:
                    avg_loss = running_loss / (batch_idx + 1)
                    click.echo(f"   Batch [{batch_idx}/{len(train_loader)}] - Loss: {loss.item():.5f}")

        train_loss = running_loss / len(train_loader)

        # Validation phase
        model.eval()
        valid_loss = 0.0

        with torch.no_grad():
            for mel_spec, labels, _ in tqdm(valid_loader, desc="Validation", unit="batch"):
                mel_spec = mel_spec.to(device)
                labels = labels.to(device)
                outputs = model(mel_spec)
                loss = criterion(outputs, labels)
                valid_loss += loss.item()

        valid_loss /= len(valid_loader)

        # Update history
        history_loss.append(train_loss)
        history_valid_loss.append(valid_loss)

        # Learning rate scheduling
        scheduler.step(valid_loss)
        current_lr = optimizer.param_groups[0]['lr']

        # Save history
        np.save(model_dir / 'history_loss.npy', history_loss)
        np.save(model_dir / 'history_valid_loss.npy', history_valid_loss)

        # Check improvement
        if valid_loss < best_loss:
            best_loss = valid_loss
            early_stop_counter = 0

            click.echo(f"\n💾 Saving improved model (valid loss: {valid_loss:.5f})")
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'history_loss': history_loss,
                'history_valid_loss': history_valid_loss,
                'best_loss': best_loss,
                'epoch': epoch + 1,
                'model_config': model_config
            }, checkpoint_path)
        else:
            early_stop_counter += 1
            click.echo(f"\n🛑 No improvement - patience: {early_stop_counter}/{early_stop_patience}")

        # Summary
        click.echo(f"\n📊 Epoch {epoch + 1} Summary:")
        click.echo(f"   Train loss: {train_loss:.5f}")
        click.echo(f"   Valid loss: {valid_loss:.5f}")
        click.echo(f"   Learning rate: {current_lr:.1e}")
        click.echo(f"   Best valid loss: {best_loss:.5f}")
        click.echo("")

        # Early stopping
        if early_stop_counter >= early_stop_patience:
            click.echo(f"\n🚨 Early stopping triggered after {early_stop_patience} epochs without improvement")
            break

    # Save final weights
    final_path = model_dir / 'final_weights.pth'
    torch.save(model.state_dict(), final_path)
    click.echo(f"\n💾 Final weights saved: {final_path}")

    # Plot training history
    click.echo(f"\n📈 Generating training curves...")
    plot_training_history(history_loss, history_valid_loss, model_dir)

    click.echo(f"\n✅ Training completed!")
    click.echo(f"   Best validation loss: {best_loss:.5f}")
    click.echo(f"   Total epochs: {len(history_loss)}")
    click.echo(f"   Model directory: {model_dir}")


@cli.command()
@click.option(
    '--dataset-name',
    default='dataset',
    help='Name of the dataset directory.',
    show_default=True
)
@click.option(
    '--model-name',
    default='CustomCNN',
    help='Model architecture name.',
    show_default=True
)
@click.option(
    '--dataset-variant',
    default='custom',
    help='Dataset variant identifier.',
    show_default=True
)
@click.option(
    '--batch-size',
    default=128,
    type=int,
    help='Evaluation batch size.',
    show_default=True
)
@click.option(
    '--gpu-id',
    default=0,
    type=int,
    help='GPU device ID (-1 for CPU).',
    show_default=True
)
def evaluate(dataset_name, model_name, dataset_variant, batch_size, gpu_id):
    """
    Evaluate trained model on test set.

    Computes comprehensive metrics including micro/macro/weighted F1,
    precision, recall, and per-class performance analysis.
    """
    click.echo("=" * 60)
    click.echo("MODEL EVALUATION")
    click.echo("=" * 60)

    # Setup device
    if gpu_id >= 0 and torch.cuda.is_available():
        device = torch.device(f"cuda:{gpu_id}")
    else:
        device = torch.device("cpu")

    click.echo(f"\n🖥️  Device: {device}")

    # Load dataset config
    dataset_config_path = Path('utils') / dataset_name / f'dataset_config_{dataset_variant}.json'
    if not dataset_config_path.exists():
        click.echo(f"\n❌ Error: Dataset config not found")
        return

    with open(dataset_config_path) as f:
        dataset_config = json.load(f)

    num_classes = len(dataset_config['mappings'])
    click.echo(f"\n📊 Dataset: {dataset_name}")
    click.echo(f"   Classes: {num_classes}")

    # Load model
    model_dir = Path('models') / model_name / dataset_variant
    checkpoint_path = model_dir / 'checkpoint.pth'

    if not checkpoint_path.exists():
        click.echo(f"\n❌ Error: Checkpoint not found: {checkpoint_path}")
        return

    click.echo(f"\n📂 Loading model from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model_config = checkpoint['model_config']

    # Build model
    model_class = utils.load_model_class(model_name)
    model = model_class((256, 256), model_config, num_classes)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()

    click.echo(f"   ✓ Model loaded (best valid loss: {checkpoint['best_loss']:.5f})")

    # Load test data
    click.echo(f"\n📦 Loading test data...")
    test_loader = utils.get_dataloader(
        dataset_config,
        split='test',
        batch_size=batch_size,
        shuffle=False
    )
    click.echo(f"   Test batches: {len(test_loader)}")

    # Evaluation
    click.echo(f"\n🔍 Evaluating model...")

    all_predictions = []
    all_labels = []

    with torch.no_grad():
        for mel_spec, labels, _ in tqdm(test_loader, desc="Evaluating", unit="batch"):
            mel_spec = mel_spec.to(device)
            outputs = model(mel_spec)
            predictions = torch.sigmoid(outputs).cpu().numpy()

            all_predictions.append(predictions)
            all_labels.append(labels.numpy())

    all_predictions = np.vstack(all_predictions)
    all_labels = np.vstack(all_labels)

    # Compute metrics
    from sklearn.metrics import f1_score, precision_score, recall_score, classification_report

    # Convert to binary predictions (threshold=0.5)
    binary_predictions = (all_predictions > 0.5).astype(int)

    # Global metrics
    micro_f1 = f1_score(all_labels, binary_predictions, average='micro')
    macro_f1 = f1_score(all_labels, binary_predictions, average='macro')
    weighted_f1 = f1_score(all_labels, binary_predictions, average='weighted')
    samples_f1 = f1_score(all_labels, binary_predictions, average='samples')

    micro_precision = precision_score(all_labels, binary_predictions, average='micro')
    micro_recall = recall_score(all_labels, binary_predictions, average='micro')

    # Display results
    click.echo(f"\n{'='*60}")
    click.echo("EVALUATION RESULTS")
    click.echo(f"{'='*60}")
    click.echo(f"\n📈 Global Metrics:")
    click.echo(f"   Micro F1:    {micro_f1:.4f}")
    click.echo(f"   Macro F1:    {macro_f1:.4f}")
    click.echo(f"   Weighted F1: {weighted_f1:.4f}")
    click.echo(f"   Samples F1:  {samples_f1:.4f}")
    click.echo(f"\n   Precision:   {micro_precision:.4f}")
    click.echo(f"   Recall:      {micro_recall:.4f}")

    # Per-class metrics
    click.echo(f"\n📊 Per-Class F1 Scores (Top 20):")
    per_class_f1 = f1_score(all_labels, binary_predictions, average=None)

    # Sort by F1 score
    species_names = list(dataset_config['mappings'].keys())
    class_scores = [(species_names[i], per_class_f1[i]) for i in range(len(per_class_f1))]
    class_scores.sort(key=lambda x: x[1], reverse=True)

    click.echo(f"\n{'Species':<50} {'F1 Score':>10}")
    click.echo("-" * 62)
    for species, score in class_scores[:20]:
        click.echo(f"{species:<50} {score:>10.4f}")

    # Save results
    results = {
        'micro_f1': float(micro_f1),
        'macro_f1': float(macro_f1),
        'weighted_f1': float(weighted_f1),
        'samples_f1': float(samples_f1),
        'micro_precision': float(micro_precision),
        'micro_recall': float(micro_recall),
        'per_class_f1': {species: float(score) for species, score in class_scores}
    }

    results_path = model_dir / 'evaluation_results.json'
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)

    click.echo(f"\n💾 Results saved: {results_path}")
    click.echo(f"\n✅ Evaluation completed!")


def plot_training_history(train_loss, valid_loss, save_dir):
    """Generate and save training history plots."""
    plt.figure(figsize=(10, 6))
    plt.plot(train_loss, label='Training Loss', linewidth=2)
    plt.plot(valid_loss, label='Validation Loss', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training History')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    plot_path = Path(save_dir) / 'training_history.png'
    plt.savefig(plot_path, dpi=150)
    plt.close()

    print(f"   Training plot saved: {plot_path}")


if __name__ == '__main__':
    cli()
