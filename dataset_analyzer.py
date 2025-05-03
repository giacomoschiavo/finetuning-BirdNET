import os
import json
from birdnetlib.analyzer import Analyzer
from birdnetlib.batch import DirectoryAnalyzer
import copy

class BirdAnalyzer:
    def __init__(self, model_name, dataset_path, model_folder_path, clf_name="CustomClassifier", min_conf=0.1):
        self.model_name = model_name
        self.dataset_path = dataset_path
        self.min_conf = min_conf
        self.complete_pred_segments = {}
        
        # DA MODIFICARE QUI SOTTO
        self.model_path = f"{model_folder_path}/{model_name}/{clf_name}.tflite"
        self.labels_path = f"{model_folder_path}/{model_name}/{clf_name}_Labels.txt"
        
        self.test_path = os.path.join(dataset_path, 'test')
        self.valid_path = os.path.join(dataset_path, 'valid')
        
        self.analyzer = Analyzer(
            classifier_labels_path=self.labels_path, 
            classifier_model_path=self.model_path
        )

        # Original BirdNET
        # self.analyzer = Analyzer(
        #     custom_species_list_path=self.labels_path, 
        # )
    
    def _on_analyze_complete(self, recording):
        audio_name = recording.path.split('\\')[-1]
        from_wabad = len(audio_name.split('_')) == 5

        if from_wabad:
            site, date, hour, segm_sec1, segm_sec2 = audio_name.split('_')
            audio_name = "_".join([site, date, hour]) + ".WAV"
        else:
            date, hour, segm_sec1, segm_sec2 = audio_name.split('_')
            audio_name = "_".join([date, hour]) + ".WAV"

        segm_sec2 = segm_sec2.split('.')[0]
        segm_id = "_".join([segm_sec1, segm_sec2])
        
        if audio_name not in self.complete_pred_segments:
            self.complete_pred_segments[audio_name] = {}
        
        self.complete_pred_segments[audio_name][segm_id] = {
            detection["label"]: detection["confidence"] for detection in recording.detections
        }
        
        print("Analyzing ", recording.path)
    
    def on_error(self, recording, error):
        print(f"An exception occurred: {error}")
        print(recording.path)
    
    def process_data_set(self, data_set_type="valid"):
        if data_set_type not in ["valid", "test"]:
            raise ValueError("data_set_type must be 'valid' or 'test'")

        if data_set_type == "valid":
            data_path = self.valid_path
        else:
            data_path = self.test_path

        self.complete_pred_segments = {}
        
        for folder in os.listdir(data_path):
            directory = os.path.join(data_path, folder)
            print(f"Starting Watcher for {data_set_type} set, folder: {folder}")

            batch = DirectoryAnalyzer(
                directory,
                analyzers=[self.analyzer],
                min_conf=self.min_conf,
            )
            batch.on_analyze_complete = self._on_analyze_complete
            batch.on_error = self.on_error
            batch.process()

        pred_segments = copy.deepcopy(self.complete_pred_segments)

        return pred_segments
