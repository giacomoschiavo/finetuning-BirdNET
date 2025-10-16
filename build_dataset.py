#!/usr/bin/env python3
"""
BirdNET Audio Segments Processing Pipeline
Modular pipeline for extraction and preprocessing of ornithological audio segments
"""

import json
import click
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from pydub import AudioSegment

# Assuming birdlib and utils are available
try:
    from birdlib import utils
except ImportError:
    print("Error: birdlib module not found. Make sure it is installed.")
    exit(1)


class BirdNETProcessor:
    """
    Main class for processing BirdNET audio segments
    Handles segment extraction, preprocessing and dataset distribution analysis
    """
    
    def __init__(self, audio_source: str, dataset_path: str, dataset_name: str = "dataset"):
        self.audio_source = Path(audio_source)
        self.dataset_path = Path(dataset_path)
        self.dataset_name = dataset_name
        self.train_path = self.dataset_path / "train"
        self.valid_path = self.dataset_path / "valid" 
        self.test_path = self.dataset_path / "test"
        
        # Create directories if they do not exist
        for path in [self.dataset_path, self.train_path, self.valid_path, self.test_path]:
            path.mkdir(parents=True, exist_ok=True)
    
    def convert_mp3_to_wav(self) -> None:
        """Convert all MP3 files in the source directory to WAV format"""
        click.echo(f"🔄 Converting MP3 → WAV in {self.audio_source}")
        
        converted_count = 0
        for filepath in self.audio_source.glob("*.mp3"):
            try:
                audio = AudioSegment.from_mp3(filepath)
                wav_path = filepath.with_suffix(".WAV")
                audio.export(wav_path, format="wav")
                click.echo(f"✅ Converted: {filepath.name} → {wav_path.name}")
                converted_count += 1
            except Exception as e:
                click.echo(f"❌ Conversion error {filepath.name}: {e}")

        click.echo(f"📊 Conversion completed: {converted_count} files processed")
    
    def normalize_filenames(self) -> None:
        """Normalize filenames from 'XC123456 - Species - Scientific.wav' → 'XC123456_0.WAV'"""
        click.echo("🔧 Normalizing filenames...")
        
        renamed_count = 0
        for filepath in self.audio_source.glob("*.WAV"):
            if " - " in filepath.name:
                code = filepath.name.split(" - ")[0]
                new_path = filepath.parent / f"{code}_0.WAV"
                filepath.rename(new_path)
                click.echo(f"📝 Renamed: {filepath.name} → {new_path.name}")
                renamed_count += 1
        
        click.echo(f"📊 Normalization completed: {renamed_count} files renamed")
    
    def extract_annotations(self, bird_tags_files: str, 
                          species_dict_file: str) -> Tuple[Dict, Dict]:
        """
        Extract annotations from .mat files and build category/audio dictionaries

        Args:
            bird_tags_files: List of .mat annotation files
            species_dict_file: BirdNET species dictionary file

        Returns:
            Tuple of (category_annots, audio_annots)
        """
        click.echo("🏷️  Extracting annotations from .mat files...")

        # Load species dictionary
        species_dict = utils.get_species_dict(species_dict_file)
        
        category_annots = {}
        audio_annots = {}

        click.echo(f"📂 Processing {bird_tags_files}...")
        category_annots, audio_annots, _ = utils.get_audio_category_annots(
            bird_tags_files, str(self.audio_source), species_dict
        )

        click.echo(f"✅ Annotations extracted: {len(category_annots)} categories, {len(audio_annots)} audio files")
        return category_annots, audio_annots
    
    def generate_audio_segments(self, audio_annots: Dict, split: str = "train",
                                generate_none: bool = True) -> None:
        """
        Generate 3s audio segments with 50% overlap (1.5s shift)

        Args:
            audio_annots: Audio annotations dictionary
            split: Dataset split (train/valid/test)
            generate_none: Include unannotated segments as class "None"
        """
        click.echo(f"🎵 Generating audio segments for split: {split}")

        # Generate audio info
        utils_dir = Path("utils")
        utils_dir.mkdir(exist_ok=True)

        audio_info = utils.load_or_generate_info(
            f'audio_info_{split}.json', audio_annots, str(self.audio_source), 'utils'
        )

        # Generate true segments
        true_segments = utils.generate_true_segments(audio_annots, audio_info)

        # Save true segments
        dataset_utils_dir = utils_dir / self.dataset_name
        dataset_utils_dir.mkdir(exist_ok=True)

        with open(dataset_utils_dir / f'true_segments_{split}.json', 'w') as f:
            json.dump(true_segments, f)

        # Generate physical audio segments
        target_path = self.dataset_path / split
        utils.generate_segments(
            audio_source_path=str(self.audio_source),
            target_path=str(target_path),
            true_segments=true_segments,
            audio_info=audio_info,
            generate_None=generate_none
        )

        click.echo(f"✅ Segments generated at: {target_path}")
    
    def analyze_dataset_distribution(self) -> pd.DataFrame:
        """
        Analyze and display dataset distribution

        Returns:
            DataFrame with counts per split
        """
        click.echo("📊 Analyzing dataset distribution...")
        
        # Get species list from intersection of train and test
        if self.test_path.exists() and self.train_path.exists():
            train_species = set(d.name for d in self.train_path.iterdir() if d.is_dir())
            test_species = set(d.name for d in self.test_path.iterdir() if d.is_dir())
            species_list = train_species.intersection(test_species)
            
            # Remove problematic species
            species_list.discard('Engine_Engine')
            species_list.discard('Cuculus canorus_Common Cuckoo')
        else:
            species_list = set()
        
        dataset_count = {}
        for species in species_list:
            dataset_count[species] = {
                "train": len(list((self.train_path / species).iterdir())) if (self.train_path / species).exists() else 0,
                "valid": len(list((self.valid_path / species).iterdir())) if (self.valid_path / species).exists() else 0,
                "test": len(list((self.test_path / species).iterdir())) if (self.test_path / species).exists() else 0
            }
        
        df = pd.DataFrame.from_dict(dataset_count, orient='index')
        df.index.name = 'Species'
        df_sorted = df.sort_values(by=["train"], ascending=False)
        
        click.echo(f"📈 Dataset distribution (Top 10):")
        click.echo(df_sorted.head(10).to_string())

        # Save full report
        report_path = self.dataset_path / "dataset_distribution.csv"
        df_sorted.to_csv(report_path)
        click.echo(f"📄 Full report saved: {report_path}")

        return df_sorted
    
    def process_complete_pipeline(self, bird_tags_files: List[str],
                                species_dict_file: str, generate_none: bool = True) -> None:
        """
        Run the complete processing pipeline

        Args:
            bird_tags_files: .mat annotation files
            species_dict_file: Species dictionary file
            generate_none: Include "None" segments
        """
        click.echo("🚀 Starting full BirdNET pipeline...")

        # File preprocessing
        self.convert_mp3_to_wav()
        self.normalize_filenames()

        # Extract annotations
        category_annots_train, audio_annots_train = self.extract_annotations(
            "Bird_tags_Train.mat", species_dict_file
        )
        category_annots_test, audio_annots_test = self.extract_annotations(
            "Bird_tags_Test.mat", species_dict_file
        )

        # Generate segments for all splits
        if audio_annots_train and audio_annots_test:
            self.generate_audio_segments(audio_annots_train, split="train", generate_none=generate_none)
            self.generate_audio_segments(audio_annots_test, split="test", generate_none=False)
            # If there are separate test annotations, handle them here

        self.analyze_dataset_distribution()
        click.echo("🎉 Pipeline completed successfully!")


@click.command()
@click.version_option(version="1.0.0", prog_name="BirdNET Pipeline")
@click.option('--audio-source', '-s', default="Tovanella",
              help='Path to source audio files directory')
@click.option('--dataset-path', '-d', default="segments",
              help='Destination path for the dataset')
@click.option('--bird-tags-files', '-b', multiple=True, 
              default=["Bird_tags_Train.mat", "Bird_tags_Test.mat"],
              help='Annotation .mat files (can be specified multiple times)')
@click.option('--species-dict', '-dict', default="BirdNET_GLOBAL_6K_V2.4_Labels_en_uk.txt",
              help='BirdNET species dictionary file')
@click.option('--generate-none/--no-generate-none', default=True,
              help='Include unannotated segments as class "None"')
@click.option('--dataset-name', '-n', default='dataset',
              help='Dataset name')
def main(audio_source: str, dataset_path: str, bird_tags_files: tuple,
         species_dict: str, generate_none: bool, dataset_name: str):
    """
    Modular BirdNET pipeline for extraction and preprocessing of ornithological audio segments.

    Scientific-technical tool for ornithological research with focus on performance and scalability.
    Runs the full pipeline: conversion + extraction + analysis.
    """
    processor = BirdNETProcessor(audio_source, dataset_path, dataset_name)
    processor.process_complete_pipeline(
        list(bird_tags_files), species_dict, generate_none
    )


if __name__ == "__main__":
    main()
