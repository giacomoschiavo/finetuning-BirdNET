#!/usr/bin/env python3
"""
BirdNET Audio Segments Processing Pipeline
Pipeline modulare per estrazione e preprocessing di segmenti audio ornitologici
"""

import os
import json
import click
import pandas as pd
import scipy.io
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from pydub import AudioSegment
from tqdm import tqdm
import copy

# Assumendo che birdlib e utils siano disponibili
try:
    from birdlib import utils
except ImportError:
    print("Errore: modulo birdlib non trovato. Assicurarsi che sia installato.")
    exit(1)


class BirdNETProcessor:
    """
    Classe principale per il processing di segmenti audio BirdNET
    Gestisce estrazione segmenti, preprocessing e analisi distribuzione dataset
    """
    
    def __init__(self, audio_source: str, dataset_path: str, dataset_name: str = "dataset"):
        self.audio_source = Path(audio_source)
        self.dataset_path = Path(dataset_path)
        self.dataset_name = dataset_name
        self.train_path = self.dataset_path / "train"
        self.valid_path = self.dataset_path / "valid" 
        self.test_path = self.dataset_path / "test"
        
        # Crea directory se non esistono
        for path in [self.dataset_path, self.train_path, self.valid_path, self.test_path]:
            path.mkdir(parents=True, exist_ok=True)
    
    def convert_mp3_to_wav(self) -> None:
        """Converte tutti i file MP3 nella directory sorgente in formato WAV"""
        click.echo(f"🔄 Conversione MP3 → WAV in {self.audio_source}")
        
        converted_count = 0
        for filepath in self.audio_source.glob("*.mp3"):
            try:
                audio = AudioSegment.from_mp3(filepath)
                wav_path = filepath.with_suffix(".WAV")
                audio.export(wav_path, format="wav")
                click.echo(f"✅ Convertito: {filepath.name} → {wav_path.name}")
                converted_count += 1
            except Exception as e:
                click.echo(f"❌ Errore conversione {filepath.name}: {e}")
        
        click.echo(f"📊 Conversione completata: {converted_count} file processati")
    
    def normalize_filenames(self) -> None:
        """Normalizza nomi file formato 'XC123456 - Species - Scientific.wav' → 'XC123456_0.WAV'"""
        click.echo("🔧 Normalizzazione nomi file...")
        
        renamed_count = 0
        for filepath in self.audio_source.glob("*.WAV"):
            if " - " in filepath.name:
                code = filepath.name.split(" - ")[0]
                new_path = filepath.parent / f"{code}_0.WAV"
                filepath.rename(new_path)
                click.echo(f"📝 Rinominato: {filepath.name} → {new_path.name}")
                renamed_count += 1
        
        click.echo(f"📊 Normalizzazione completata: {renamed_count} file rinominati")
    
    def extract_annotations(self, bird_tags_files: str, 
                          species_dict_file: str) -> Tuple[Dict, Dict]:
        """
        Estrae annotazioni da file .mat e crea dizionari categoria/audio
        
        Args:
            bird_tags_files: Lista file .mat con annotazioni
            species_dict_file: File dizionario specie BirdNET
            
        Returns:
            Tuple di (category_annots, audio_annots)
        """
        click.echo("🏷️  Estrazione annotazioni da file .mat...")
        
        # Carica dizionario specie
        species_dict = utils.get_species_dict(species_dict_file)
        
        category_annots = {}
        audio_annots = {}
        
        click.echo(f"📂 Processando {bird_tags_files}...")
        category_annots, audio_annots, _ = utils.get_audio_category_annots(
            bird_tags_files, str(self.audio_source), species_dict
        )
        
    
        click.echo(f"✅ Annotazioni estratte: {len(category_annots)} categorie, {len(audio_annots)} file audio")
        return category_annots, audio_annots
    
    def generate_audio_segments(self, audio_annots: Dict, split: str = "train",
                              generate_none: bool = True) -> None:
        """
        Genera segmenti audio 3s con overlap 50% (shift 1.5s)
        
        Args:
            audio_annots: Dizionario annotazioni audio
            split: Split dataset (train/valid/test)
            generate_none: Include segmenti non annotati come classe "None"
        """
        click.echo(f"🎵 Generazione segmenti audio per split: {split}")
        
        # Genera info audio
        utils_dir = Path("utils")
        utils_dir.mkdir(exist_ok=True)
        
        audio_info = utils.load_or_generate_info(
            f'audio_info_{split}.json', audio_annots, str(self.audio_source), 'utils'
        )
        
        # Genera true segments
        true_segments = utils.generate_true_segments(audio_annots, audio_info)
        
        # Salva true segments
        dataset_utils_dir = utils_dir / self.dataset_name
        dataset_utils_dir.mkdir(exist_ok=True)
        
        with open(dataset_utils_dir / f'true_segments_{split}.json', 'w') as f:
            json.dump(true_segments, f)
        
        # Genera segmenti fisici
        target_path = self.dataset_path / split
        utils.generate_segments(
            audio_source_path=str(self.audio_source),
            target_path=str(target_path),
            true_segments=true_segments,
            audio_info=audio_info,
            generate_None=generate_none
        )
        
        click.echo(f"✅ Segmenti generati in: {target_path}")
    
    def analyze_dataset_distribution(self) -> pd.DataFrame:
        """
        Analizza e visualizza distribuzione del dataset
        
        Returns:
            DataFrame con conteggi per split
        """
        click.echo("📊 Analisi distribuzione dataset...")
        
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
        
        click.echo(f"📈 Distribuzione dataset (Top 10):")
        click.echo(df_sorted.head(10).to_string())
        
        # Salva report completo
        report_path = self.dataset_path / "dataset_distribution.csv"
        df_sorted.to_csv(report_path)
        click.echo(f"📄 Report completo salvato: {report_path}")
        
        return df_sorted
    
    def process_complete_pipeline(self, bird_tags_files: List[str],
                                species_dict_file: str, generate_none: bool = True) -> None:
        """
        Esegue pipeline completa di processing
        
        Args:
            bird_tags_files: File annotazioni .mat
            species_dict_file: File dizionario specie
            generate_none: Include segmenti "None"
        """
        click.echo("🚀 Avvio pipeline completa BirdNET...")
        
        # 1. Preprocessing file
        self.convert_mp3_to_wav()
        self.normalize_filenames()
        
        # 2. Estrazione annotazioni
        category_annots_train, audio_annots_train = self.extract_annotations(
            "Bird_tags_Train.mat", species_dict_file
        )
        category_annots_test, audio_annots_test = self.extract_annotations(
            "Bird_tags_Test.mat", species_dict_file
        )
        
        # 3. Split train/test (assumendo che i file train/test siano separati)
        # Qui si può implementare logica di split custom se necessario
        
        # 4. Generazione segmenti per tutti gli split
        if audio_annots_train and audio_annots_test:
            self.generate_audio_segments(audio_annots_train, split="train", generate_none=generate_none)
            self.generate_audio_segments(audio_annots_test, split="test", generate_none=False)
            # Se ci sono annotazioni test separate, gestirle qui
        
        # 5. Analisi finale
        self.analyze_dataset_distribution()
        
        click.echo("🎉 Pipeline completata con successo!")


@click.command()
@click.version_option(version="1.0.0", prog_name="BirdNET Pipeline")
@click.option('--audio-source', '-s', default="Tovanella",
              help='Path directory file audio sorgente')
@click.option('--dataset-path', '-d', default="segments",
              help='Path destinazione dataset')
@click.option('--bird-tags-files', '-b', multiple=True, 
              default=["Bird_tags_Train.mat", "Bird_tags_Test.mat"],
              help='File annotazioni .mat (può essere specificato più volte)')
@click.option('--species-dict', '-dict', default="BirdNET_GLOBAL_6K_V2.4_Labels_en_uk.txt",
              help='File dizionario specie BirdNET')
@click.option('--generate-none/--no-generate-none', default=True,
              help='Includi segmenti non annotati come classe "None"')
@click.option('--dataset-name', '-n', default='dataset',
              help='Nome dataset')
def main(audio_source: str, dataset_path: str, bird_tags_files: tuple,
         species_dict: str, generate_none: bool, dataset_name: str):
    """
    Pipeline modulare BirdNET per estrazione e preprocessing segmenti audio ornitologici.
    
    Strumento tecnico-scientifico per ricerca ornitologica con focus su performance e scalabilità.
    Esegue pipeline completa: conversione + estrazione + analisi.
    """
    processor = BirdNETProcessor(audio_source, dataset_path, dataset_name)
    processor.process_complete_pipeline(
        list(bird_tags_files), species_dict, generate_none
    )


if __name__ == "__main__":
    main()
