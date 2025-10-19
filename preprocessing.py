#!/usr/bin/env python3
"""
BirdNET Dataset Pipeline
========================
Integrated pipeline for ornithological research dataset preparation.
Combines preprocessing, segment integration, and data augmentation workflows.

This script processes audio annotations, generates labeled segments, 
and applies data augmentation techniques for machine learning training.
"""

import os
import json
import random
import click
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
import scipy.io
from birdlib import utils
from audiomentations import Compose, PitchShift, TimeStretch, AddBackgroundNoise, Gain
import soundfile as sf
import librosa

@click.group()
def cli():
    """BirdNET Dataset Processing Pipeline"""
    pass


@cli.command()
@click.option(
    '--dataset-name',
    default='dataset',
    help='Name of the dataset directory.',
    show_default=True
)
@click.option(
    '--audio-source',
    default='Tovanella',
    type=click.Path(exists=True),
    help='Path to source audio files.',
    show_default=True
)
@click.option(
    '--segments-base',
    default='segments',
    type=click.Path(),
    help='Base directory for segment storage.',
    show_default=True
)
@click.option(
    '--annotation-file',
    default='Birds_tags_Train_2.mat',
    type=click.Path(exists=True),
    help='MATLAB annotation file path.',
    show_default=True
)
@click.option(
    '--labels-file',
    default='BirdNET_GLOBAL_6K_V2.4_Labels_en_uk.txt',
    type=click.Path(exists=True),
    help='BirdNET species labels file.',
    show_default=True
)
@click.option(
    '--utils-dir',
    default='utils',
    type=click.Path(),
    help='Directory for utility files and cache.',
    show_default=True
)
@click.option(
    '--generate-none/--no-generate-none',
    default=True,
    help='Generate negative samples (None class).',
    show_default=True
)
def integrate_segments(dataset_name, audio_source, segments_base, annotation_file, 
                      labels_file, utils_dir, generate_none):
    """
    Integrate annotated audio segments into dataset structure.

    Processes MATLAB annotations and generates train/valid/test splits
    with properly labeled audio segments for model training.
    """
    click.echo("=" * 60)
    click.echo("SEGMENT INTEGRATION PIPELINE")
    click.echo("=" * 60)

    # Setup paths
    dataset_path = Path(segments_base)
    integration_path = dataset_path / 'integration'
    integration_path.mkdir(parents=True, exist_ok=True)

    click.echo(f"\n📁 Dataset: {dataset_path}")
    click.echo(f"📁 Audio Source: {audio_source}")

    # Load species dictionary
    click.echo(f"\n📋 Loading species dictionary from {labels_file}...")
    species_dict = utils.get_species_dict(labels_file)
    click.echo(f"   ✓ Loaded {len(species_dict)} species")

    # Load annotations
    click.echo(f"\n📋 Processing annotations from {annotation_file}...")
    category_annots, audio_annots, missing = utils.get_audio_category_annots(
        annotation_file, audio_source, species_dict
    )
    species_list = list(category_annots.keys())
    click.echo(f"   ✓ Found {len(species_list)} species categories")
    click.echo(f"   ✓ Processed {len(audio_annots)} audio files")
    if missing:
        click.echo(f"   ⚠ Missing files: {len(missing)}")

    # Generate audio info cache
    info_cache = 'audio_info_2.json'
    click.echo(f"\n⚙️  Generating audio metadata...")
    audio_info = utils.load_or_generate_info(
        str(info_cache), audio_annots, audio_source, utils_dir
    )
    click.echo(f"   ✓ Audio info cached to {info_cache}")

    # Generate true segments
    click.echo(f"\n🎯 Generating ground truth segments...")
    true_segments = utils.generate_true_segments(audio_annots, audio_info)

    segments_cache = Path(utils_dir) / dataset_name / 'true_segments_2.json'
    segments_cache.parent.mkdir(parents=True, exist_ok=True)
    with open(segments_cache, 'w') as f:
        json.dump(true_segments, f)
    click.echo(f"   ✓ True segments saved to {segments_cache}")

    # Generate segment files
    click.echo(f"\n🔊 Extracting audio segments...")
    click.echo(f"   Generate negative samples: {generate_none}")
    utils.generate_segments(
        audio_source_path=audio_source,
        target_path=str(integration_path),
        true_segments=true_segments,
        audio_info=audio_info,
        generate_None=generate_none
    )

    click.echo(f"\n✅ Segment integration completed!")
    click.echo(f"   Output: {integration_path}")


@cli.command()
@click.option(
    '--dataset-name',
    default='dataset',
    help='Name of the dataset directory.',
    show_default=True
)
@click.option(
    '--segments-base',
    default='segments',
    type=click.Path(),
    help='Base directory for segment storage.',
    show_default=True
)
@click.option(
    '--train-split',
    default=0.7,
    type=float,
    help='Training set proportion (0.0-1.0).',
    show_default=True
)
@click.option(
    '--valid-split',
    default=0.15,
    type=float,
    help='Validation set proportion (0.0-1.0).',
    show_default=True
)
@click.option(
    '--exclude-species',
    multiple=True,
    default=['Engine_Engine', 'Cuculus canorus_Common Cuckoo'],
    help='Species to exclude from processing.',
    show_default=True
)
@click.option(
    '--seed',
    default=42,
    type=int,
    help='Random seed for reproducibility.',
    show_default=True
)
def preprocess(dataset_name, segments_base, train_split, valid_split, 
               exclude_species, seed):
    """
    Preprocess dataset and create train/validation/test splits.

    Organizes audio segments into structured directories with balanced
    splits for training, validation, and testing phases.
    """
    click.echo("=" * 60)
    click.echo("DATA PREPROCESSING PIPELINE")
    click.echo("=" * 60)

    random.seed(seed)
    np.random.seed(seed)

    dataset_path = Path(segments_base)
    train_path = dataset_path / 'train'
    valid_path = dataset_path / 'valid'
    test_path = dataset_path / 'test'

    click.echo(f"\n📁 Dataset: {dataset_path}")
    click.echo(f"📊 Split ratios: Train={train_split:.2f}, Valid={valid_split:.2f}")

    # Get species list
    if not test_path.exists() or not train_path.exists():
        click.echo(f"\n❌ Error: Dataset directories not found!")
        click.echo(f"   Expected: {train_path} and {test_path}")
        return

    species_list = set(os.listdir(test_path)).intersection(set(os.listdir(train_path)))

    # Exclude species
    for species in exclude_species:
        if species in species_list:
            species_list.remove(species)
            click.echo(f"Excluded: {species}")

    click.echo(f"\n🔍 Found {len(species_list)} species to process")

    # Move species not in species list to the "removed" folder
    click.echo(f"\n🗑️  Removing unwanted species...")
    removed_path = dataset_path / 'removed'
    removed_path.mkdir(parents=True, exist_ok=True)
    for species_dir in train_path.iterdir():
        if species_dir.name not in species_list:
            target_dir = removed_path / species_dir.name
            target_dir.parent.mkdir(parents=True, exist_ok=True)
            species_dir.rename(target_dir)
            click.echo(f"Moved {species_dir.name} to removed folder")

    # Process splits
    click.echo(f"\n⚙️  Performing dataset splits...")

    # Check if files aleready present in valid folder
    if any(valid_path.glob('*/*.wav')):
        click.echo(f"\n❌ Error: Validation directory is not empty!")
        click.echo(f"   Please clear {valid_path} before proceeding.")
        return

    for species in tqdm(species_list, desc="Processing species"):
        species_train = train_path / species
        species_valid = valid_path / species
        species_test = test_path / species

        species_valid.mkdir(parents=True, exist_ok=True)

        if species_train.exists():
            files = list(species_train.glob('*.wav'))

            n_train = int(len(files) * train_split)
            n_valid = int(len(files) * valid_split)

            valid_files = files[n_train:n_train + n_valid]

            for file in valid_files:
                file.rename(species_valid / file.name)

    # Generate statistics
    click.echo(f"\n📊 Generating dataset statistics...")
    stats = generate_dataset_stats(dataset_path, species_list)

    # Display summary
    click.echo(f"\n{'Species':<50} {'Train':>8} {'Valid':>8} {'Test':>8}")
    click.echo("-" * 78)
    for species, counts in list(stats.items())[:10]:  # Top 10
        click.echo(f"{species:<50} {counts['train']:>8} {counts['valid']:>8} {counts['test']:>8}")

    if len(stats) > 10:
        click.echo(f"\n   ... and {len(stats) - 10} more species")

    total_train = sum(s['train'] for s in stats.values())
    total_valid = sum(s['valid'] for s in stats.values())
    total_test = sum(s['test'] for s in stats.values())

    click.echo("-" * 78)
    click.echo(f"{'TOTAL':<50} {total_train:>8} {total_valid:>8} {total_test:>8}")

    click.echo(f"\n✅ Preprocessing completed!")


@cli.command()
@click.option(
    '--dataset-name',
    default='dataset',
    help='Name of the dataset directory.',
    show_default=True
)
@click.option(
    '--segments-base',
    default='segments',
    type=click.Path(),
    help='Base directory for segment storage.',
    show_default=True
)
@click.option(
    '--target-count',
    default=1000,
    type=int,
    help='Target sample count for augmentation.',
    show_default=True
)
@click.option(
    '--max-none-samples',
    default=5000,
    type=int,
    help='Maximum negative samples to keep.',
    show_default=True
)
@click.option(
    '--augmentation-methods',
    multiple=True,
    default=['time_stretch', 'pitch_shift', 'add_noise', 'time_shift'],
    help='Augmentation techniques to apply.',
    show_default=True
)
@click.option(
    '--exclude-species',
    multiple=True,
    default=['Engine_Engine', 'Cuculus canorus_Common Cuckoo'],
    help='Species to exclude from augmentation.',
    show_default=True
)
@click.option(
    '--seed',
    default=42,
    type=int,
    help='Random seed for reproducibility.',
    show_default=True
)
def augment(dataset_name, segments_base, target_count, max_none_samples,
            augmentation_methods, exclude_species, seed):
    """
    Apply data augmentation to balance dataset classes.

    Generates synthetic training samples using audio transformation
    techniques to address class imbalance and improve model generalization.
    """
    click.echo("=" * 60)
    click.echo("DATA AUGMENTATION PIPELINE")
    click.echo("=" * 60)

    random.seed(seed)
    np.random.seed(seed)

    dataset_path = Path(segments_base)
    train_path = dataset_path / 'train'
    test_path = dataset_path / 'test'

    click.echo(f"\n📁 Dataset: {dataset_path}")
    click.echo(f"🎯 Target samples per class: {target_count}")
    click.echo(f"🔧 Augmentation methods: {', '.join(augmentation_methods)}")

    # Get species list
    species_list = set(os.listdir(test_path)).intersection(set(os.listdir(train_path)))

    for species in exclude_species:
        if species in species_list:
            species_list.remove(species)

    click.echo(f"\n🔍 Processing {len(species_list)} species classes")

    # Handle None class separately
    if 'None' in species_list:
        none_path = train_path / 'None'
        files = list(none_path.glob('*.wav'))

        if len(files) > max_none_samples:
            click.echo(f"\n⚖️  Balancing 'None' class: {len(files)} -> {max_none_samples} samples")
            files_to_keep = random.sample(files, max_none_samples)
            files_to_remove = set(files) - set(files_to_keep)

            for file in files_to_remove:
                file.unlink()

    # Augmentation pipeline
    click.echo(f"\n🔄 Applying augmentation...")

    bg_noise_path = train_path / "None"
    bg_noises = os.listdir(bg_noise_path)

    augmentation_funcs = {
        "psA": Compose([PitchShift(min_semitones=-3, max_semitones=-1, p=0.75)]),
        "psB": Compose([PitchShift(min_semitones=1, max_semitones=3, p=0.75)]),
        "ts": Compose([TimeStretch(min_rate=0.9, max_rate=1.1, p=0.75)]),
        "gain": Compose([Gain(min_gain_db=-5, max_gain_db=5, p=0.5)]),
        "bn": Compose([
            AddBackgroundNoise(sounds_path=os.path.join(bg_noise_path, random.choice(bg_noises)), p=0.8),
            Gain(min_gain_db=-5, max_gain_db=5, p=0.5) 
        ]),
    }

    stats_before = generate_dataset_stats(dataset_path, species_list)

    for species in tqdm(species_list, desc="Augmenting species"):
        species_path = train_path / species

        if not species_path.exists():
            continue

        files = list(species_path.glob('*.wav'))
        current_count = len(files)

        if current_count >= target_count or species == 'None':
            continue
        # Apply augmentation
        for file in files:
            if "aug" in str(file):
                continue
            for aug_name, aug_method in augmentation_funcs.items():
                output_file = species_path / f"{file.stem}_aug_{aug_name}{file.suffix}"
                try:
                    audio, sr = librosa.load(file)
                    audio = aug_method(audio, int(sr))
                    sf.write(output_file, audio, 48000)
                except Exception as e:
                    click.echo(f"\n⚠️  Warning: Failed to augment {file.name}: {e}")

    # Generate final statistics
    click.echo(f"\n📊 Augmentation results:")
    stats_after = generate_dataset_stats(dataset_path, species_list)

    click.echo(f"\n{'Species':<50} {'Before':>10} {'After':>10} {'Change':>10}")
    click.echo("-" * 82)

    for species in list(stats_after.keys())[:15]:
        before = stats_before.get(species, {}).get('train', 0)
        after = stats_after.get(species, {}).get('train', 0)
        change = after - before

        if change > 0:
            click.echo(f"{species:<50} {before:>10} {after:>10} +{change:>9}")

    total_before = sum(s.get('train', 0) for s in stats_before.values())
    total_after = sum(s.get('train', 0) for s in stats_after.values())

    click.echo("-" * 82)
    click.echo(f"{'TOTAL':<50} {total_before:>10} {total_after:>10} +{total_after-total_before:>9}")

    click.echo(f"\n✅ Augmentation completed!")


@cli.command()
@click.option(
    '--dataset-name',
    default='dataset',
    help='Name of the dataset directory.',
    show_default=True
)
@click.option(
    '--segments-base',
    default='segments',
    type=click.Path(),
    help='Base directory for segment storage.',
    show_default=True
)
@click.option(
    '--exclude-species',
    multiple=True,
    default=['Engine_Engine', 'Cuculus canorus_Common Cuckoo'],
    help='Species to exclude from statistics.',
    show_default=True
)
@click.option(
    '--output-csv',
    type=click.Path(),
    help='Export statistics to CSV file.',
)
def stats(dataset_name, segments_base, exclude_species, output_csv):
    """
    Generate comprehensive dataset statistics report.

    Analyzes current dataset composition, class distribution, and
    provides insights for training optimization.
    """
    click.echo("=" * 60)
    click.echo("DATASET STATISTICS")
    click.echo("=" * 60)

    dataset_path = Path(segments_base)
    train_path = dataset_path / 'train'
    valid_path = dataset_path / 'valid'
    test_path = dataset_path / 'test'

    click.echo(f"\n📁 Dataset: {dataset_path}")

    # Get species list
    species_list = set(os.listdir(test_path)).intersection(set(os.listdir(train_path)))

    for species in exclude_species:
        if species in species_list:
            species_list.remove(species)

    # Generate statistics
    stats_dict = generate_dataset_stats(dataset_path, species_list)

    # Convert to DataFrame
    df = pd.DataFrame.from_dict(stats_dict, orient='index')
    df.index.name = 'Species'
    df = df.sort_values(by=['train'], ascending=False)

    # Display table
    click.echo(f"\n{df.to_string()}")

    # Summary statistics
    click.echo(f"\n{'='*60}")
    click.echo("SUMMARY")
    click.echo("=" * 60)
    click.echo(f"Total species: {len(stats_dict)}")
    click.echo(f"Total train samples: {df['train'].sum()}")
    click.echo(f"Total valid samples: {df['valid'].sum()}")
    click.echo(f"Total test samples: {df['test'].sum()}")

    # Export if requested
    if output_csv:
        df.to_csv(output_csv)
        click.echo(f"\n💾 Statistics exported to: {output_csv}")

def generate_dataset_stats(dataset_path, species_list):
    """
    Generate dataset statistics for all splits.

    Args:
        dataset_path: Path to dataset root directory
        species_list: List of species to include

    Returns:
        Dictionary with counts per species and split
    """
    train_folder = dataset_path / "train"
    valid_folder = dataset_path / "valid"
    test_folder = dataset_path / "test"

    dataset_count = {}

    for species in species_list:
        train_count = 0
        valid_count = 0
        test_count = 0

        if (train_folder / species).exists():
            train_count = len(list((train_folder / species).glob('*.wav')))

        if (valid_folder / species).exists():
            valid_count = len(list((valid_folder / species).glob('*.wav')))

        if (test_folder / species).exists():
            test_count = len(list((test_folder / species).glob('*.wav')))

        dataset_count[species] = {
            "train": train_count,
            "valid": valid_count,
            "test": test_count
        }

    return dataset_count

@cli.command()
@click.option(
    '--dataset-name',
    default='dataset',
    help='Name of the dataset directory.',
    show_default=True
)
@click.option(
    '--segments-base',
    default='segments',
    type=click.Path(),
    help='Base directory for segments.',
    show_default=True
)
@click.option(
    '--dataset-variant',
    default='custom',
    help='Dataset variant identifier.',
    show_default=True
)
@click.option(
    '--force-regenerate',
    is_flag=True,
    help='Force regeneration of dataset config.'
)
def prepare_dataset(dataset_name, segments_base, dataset_variant, force_regenerate):
    """
    Prepare dataset configuration and mappings.

    Creates JSON configuration file containing species mappings and
    sample paths for train/valid/test splits. Required before training.
    """
    click.echo("=" * 60)
    click.echo("DATASET CONFIGURATION")
    click.echo("=" * 60)

    dataset_path = Path(segments_base)
    train_path = dataset_path / 'train'
    valid_path = dataset_path / 'valid'
    test_path = dataset_path / 'test'

    # Verify paths exist
    for path in [train_path, valid_path, test_path]:
        if not path.exists():
            click.echo(f"\n❌ Error: Required path does not exist: {path}")
            return

    click.echo(f"\n📁 Dataset: {dataset_path}")
    click.echo(f"🏷️  Variant: {dataset_variant}")

    # Create config
    config_file = f'dataset_config_{dataset_variant}.json'
    saving_path = Path('utils') / dataset_name / config_file
    saving_path.parent.mkdir(parents=True, exist_ok=True)

    if saving_path.exists() and not force_regenerate:
        click.echo(f"\n✅ Dataset config already exists: {saving_path}")
        click.echo("   Use --force-regenerate to recreate")
        return

    click.echo(f"\n⚙️  Generating dataset configuration...")

    # Get species mappings
    click.echo("   → Building species mappings...")
    mappings = utils.get_mappings(str(train_path))
    click.echo(f"   ✓ Found {len(mappings)} species classes")

    # Collect samples
    click.echo("   → Collecting sample paths...")
    samples_train, samples_valid, samples_test = utils.collect_samples(
        str(train_path),
        str(valid_path),
        str(test_path),
        mappings
    )

    samples = []
    samples.extend(samples_train.values())
    samples.extend(samples_valid.values())
    samples.extend(samples_test.values())
    dataset_config = {
        "mappings": mappings,
        "samples": samples
    }

    # Save configuration
    with open(saving_path, 'w') as f:
        json.dump(dataset_config, f, indent=2)

    click.echo(f"\n💾 Configuration saved: {saving_path}")
    click.echo(f"\n📊 Dataset summary:")
    click.echo(f"   Species count: {len(mappings)}")
    click.echo(f"   Training samples: {len(samples_train)}")
    click.echo(f"   Validation samples: {len(samples_valid)}")
    click.echo(f"   Test samples: {len(samples_test)}")

    # Display first few species
    click.echo(f"\n🏷️  Species classes (first 10):")
    for species, idx in list(mappings.items())[:10]:
        click.echo(f"   [{idx:3d}] {species}")

    if len(mappings) > 10:
        click.echo(f"   ... and {len(mappings) - 10} more")

    click.echo(f"\n✅ Dataset configuration completed!")


@cli.command()
@click.option(
    '--dataset-name',
    default='dataset',
    help='Name of the dataset directory.',
    show_default=True
)
@click.option(
    '--segments-base',
    default='segments',
    type=click.Path(),
    help='Base directory for segments.',
    show_default=True
)
@click.option(
    '--dataset-variant',
    default='custom',
    help='Dataset variant identifier.',
    show_default=True
)
@click.option(
    '--force-regenerate',
    is_flag=True,
    help='Force regeneration of spectrograms.'
)
def generate_spectrograms(dataset_name, segments_base, dataset_variant, force_regenerate):
    """
    Generate mel-spectrograms from audio segments.

    Converts audio files to mel-spectrogram representations for
    CNN input. Processes all splits (train/valid/test).
    """
    click.echo("=" * 60)
    click.echo("SPECTROGRAM GENERATION")
    click.echo("=" * 60)

    dataset_path = Path(segments_base)

    # Load dataset config
    config_file = Path('utils') / dataset_name / f'dataset_config_{dataset_variant}.json'
    if not config_file.exists():
        click.echo(f"\n❌ Error: Dataset config not found: {config_file}")
        click.echo("   Run 'prepare-dataset' first")
        return

    with open(config_file) as f:
        dataset_config = json.load(f)

    click.echo(f"\n📁 Dataset: {dataset_path}")
    click.echo(f"🔊 Processing {len(dataset_config['mappings'])} species")

    # Define spectrogram paths
    specs_train = dataset_path / 'train_specs'
    specs_valid = dataset_path / 'valid_specs'
    specs_test = dataset_path / 'test_specs'

    # Create directories
    specs_train.mkdir(parents=True, exist_ok=True)
    specs_valid.mkdir(parents=True, exist_ok=True)
    specs_test.mkdir(parents=True, exist_ok=True)

    click.echo(f"\n⚙️  Generating mel-spectrograms...")

    # Process each split
    splits = [
        ('train', dataset_path / 'train', specs_train),
        ('valid', dataset_path / 'valid', specs_valid),
        ('test', dataset_path / 'test', specs_test)
    ]

    for split_name, audio_path, spec_path in splits:
        click.echo(f"\n🔄 Processing {split_name} split...")
        utils.specs_generation(
            str(audio_path),
            str(spec_path),
            dataset_config['mappings']
        )

        # Count generated specs
        spec_count = sum(1 for _ in spec_path.rglob('*.npy'))
        click.echo(f"   ✓ Generated {spec_count} spectrograms")

    click.echo(f"\n✅ Spectrogram generation completed!")



if __name__ == '__main__':
    cli()
