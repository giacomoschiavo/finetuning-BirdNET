import os
from pydub import AudioSegment
import numpy as np
from tqdm import tqdm

class SegmentCreator():

    def __init__(self, source_audio_path, target_path, audio_info, generate=False):
        self.source_audio_path = source_audio_path
        self.target_path = target_path
        self.audio_info = audio_info
        self.generate = generate
        self.true_segments = {}
        

    def generate_true_segments(self): 
        self.true_segments = {}
        audio_list = self.audio_info.keys()
        audios = list(audio_list)
        segment_length = 3.0  # Lunghezza di ogni segmento (s)
        step_size = 1.5  # Overlap tra segmenti (s)

        for j, audio in enumerate(audios):
            all_annotations = self.audio_info[audio]  
            
            # Carica la durata dell'audio (ipotizziamo sia nota, se no devi calcolarla)
            audio_duration = 600  # Esempio di audio di 10 minuti (600s), devi sostituirlo con il valore reale

            # Inizializza il dizionario per questo audio
            self.true_segments[audio] = {}

            # Genera tutti i segmenti vuoti ogni 1.5 secondi
            segment_start = 0.0
            while segment_start + segment_length <= audio_duration:
                segm_id = f"{int(segment_start)}_{int((segment_start % 1) * 10)}"
                self.true_segments[audio][segm_id] = []  # Lista vuota che poi riempiamo con le annotazioni
                segment_start += step_size  # Sposta il segmento in avanti di 1.5s

            # Ora assegniamo le annotazioni ai segmenti corrispondenti
            for annotation in all_annotations:
                start_time = annotation["start_time"]
                duration = annotation["duration"]
                species = annotation["label"]
                if species.split("_")[1] == "":     # ex. Wind_
                    species = species.split("_")[0]     # save without the underscore

                # Trova l'intervallo di tempo in cui questa annotazione è presente
                annotation_end = start_time + duration

                # Controlla quali segmenti la contengono almeno parzialmente
                for segment_start in self.true_segments[audio].keys():
                    segment_start_time = float(segment_start.replace("_", "."))  # Converte da stringa a float
                    segment_end_time = segment_start_time + segment_length

                    # Se l'annotazione cade almeno in parte in questo segmento, la aggiungiamo
                    if not (annotation_end <= segment_start_time or start_time >= segment_end_time):
                        if species not in self.true_segments[audio][segment_start]:
                            self.true_segments[audio][segment_start].append(species)
        return self.true_segments
    
    def _generate_species_segment(self, segment_audio, species_name, target_path, basename, segm_id):
        os.makedirs(os.path.join(target_path, species_name), exist_ok=True)
        export_path = os.path.join(
            target_path,
            species_name, 
            f"{basename}_{segm_id}.wav"
        )
        if os.path.exists(export_path):
            return
        segment_audio.export(export_path, format="wav")

    # generate the audio from the true_segments_audio
    def generate_segments(self):
        progress_bar_audio = tqdm(total=len(self.true_segments_audio.keys()), colour='blue')
        for audio_path, segms in self.true_segments_audio.items():
            basename = os.path.splitext(audio_path)[0]
            progress_bar = tqdm(total=len(segms), colour='red')
            print(f"Elaborating audio {audio_path}...")
            for segm_id, species in segms.items():
                audio = AudioSegment.from_file(
                        os.path.join(self.audio_source_path, audio_path), 
                        format="wav"
                    )
                segment_start_time = float(segm_id.replace("_", "."))
                segment_audio = audio[segment_start_time*1000:segment_start_time*1000 + 3000]
                if not species:
                    self._generate_species_segment(segment_audio, "None", self.target_path, basename, segm_id)
                for sp in species:
                    self._generate_species_segment(segment_audio, sp, self.target_path, basename, segm_id)
                progress_bar.update(1)
            progress_bar_audio.update(1)
