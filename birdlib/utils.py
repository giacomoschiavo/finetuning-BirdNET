import os
import torchaudio
import torch
import torch.nn.functional as F
import random
from torch.utils.data import DataLoader, Dataset
import importlib
import json
import scipy
import numpy as np
import librosa
from pydub import AudioSegment
from tqdm import tqdm

def get_mappings(train_path, include_None=False):
    train_species = os.listdir(train_path) 
    with open("utils/category_annots.json") as f:
        category_annots = json.load(f)

    filtered_species = [species for species in category_annots.keys() if species in train_species]
    mappings = {species: i for i, species in enumerate(filtered_species)}
    if include_None:
        mappings['None'] = len(filtered_species)
    return mappings

def load_model_class(model_name):
    model_module = importlib.import_module(f"models.{model_name}.model")
    model_class = getattr(model_module, model_name)
    return model_class

def collect_samples(train_path, test_path, final_test_path, mappings, seed=42):
    random.seed(seed)  
    samples_train = {}
    for species in os.listdir(train_path):
        if species not in mappings:
            continue
        audio_list = os.listdir(os.path.join(train_path, species))
        valid_samples = random.sample(audio_list, len(audio_list) // 10)
        for audio in audio_list:
            if audio in samples_train:      # same audio in different folders, save the other species
                samples_train[audio]["labels"].append(mappings[species])
                continue
            samples_train[audio] = {
                "file_path": os.path.join(train_path, species, audio),
                "split": "valid" if audio in valid_samples else "train",
                "labels": [mappings[species]]
            }

    samples_test = {}
    for species in os.listdir(test_path):
        if species not in mappings:
            continue
        for audio in os.listdir(os.path.join(test_path, species)):
            labels = [mappings[species]]
            updated = False
            if audio in samples_test:       # adds label
                samples_test[audio]["labels"].extend(labels)
                updated = True
            if audio in samples_train:      # considers training labels
                samples_train[audio]["labels"].extend(labels)       # add test label
                labels = samples_train[audio]["labels"]             # save new labels
            if updated:
                continue
            samples_test[audio] = {
                "file_path": os.path.join(test_path, species, audio),
                "split": "test",
                "labels": labels
            }

    samples_final_test = {}
    for species in os.listdir(final_test_path):
        if species not in mappings:
            continue
        for audio in os.listdir(os.path.join(final_test_path, species)):
            labels = [mappings[species]]
            samples_final_test[audio] = {
                "file_path": os.path.join(final_test_path, species, audio),
                "split": "final_test",
                "labels": labels
            }

    samples = []
    for _, props in samples_train.items():
        samples.append(props)
    for _, props in samples_test.items():
        samples.append(props)
    for _, props in samples_final_test.items():
        samples.append(props)
    return samples

def wav_to_spec(audio_path):
    split = audio_path.split("/")[-3]
    audio = audio_path.replace("wav", "pt").replace(split, f"{split}_specs")
    return audio

class CachedAudioDataset(Dataset):
    def __init__(self, dataset_config, split="train"):
        self.samples = [s for s in dataset_config["samples"] if s["split"] == split]
        self.num_classes = len(dataset_config["mappings"])
        
        # Carica tutto in RAM
        self.cache = []
        for sample in self.samples:
            spec = torch.load(wav_to_spec(sample["file_path"]))  # spettrogramma
            label_tensor = torch.zeros(self.num_classes)
            for label in sample["labels"]:
                label_tensor[label] = 1.0
            self.cache.append((spec, label_tensor, sample["file_path"]))

    def __len__(self):
        return len(self.cache)

    def __getitem__(self, idx):
        return self.cache[idx]


def get_dataloader(dataset_config, split="train", batch_size=100, shuffle=True):
    dataset = CachedAudioDataset(dataset_config=dataset_config, split=split)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def generate_spectrogram(waveform, sample_rate=32000):
    n_fft = 1024
    hop_length = 256
    win_length = 1024
    target_samples = sample_rate * 3

    if waveform.shape[1] < target_samples:
        pad_len = target_samples - waveform.shape[1]
        waveform = F.pad(waveform, (0, pad_len))
    else:
        waveform = waveform[:, :target_samples]

    waveform = waveform[0:1, :]  # mono

    window = torch.hann_window(n_fft, device=waveform.device)
    stft = torch.stft(
        waveform,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        window=window,
        return_complex=True
    )
    spectrogram = torch.abs(stft)
    spectrogram = torch.log1p(spectrogram)

    spectrogram = spectrogram.unsqueeze(0)  # (1, 1, freq, time)
    spectrogram = F.interpolate(spectrogram, size=(256, 256), mode="bilinear", align_corners=False)
    return spectrogram.squeeze(0).squeeze(0)  # [256, 256]


def specs_generation(input_path, output_path, mappings):
    sample_rate = 32000

    for species in os.listdir(input_path):
        if species not in mappings:
            continue
        species_path = os.path.join(input_path, species)
        output_species_path = os.path.join(output_path, species)
        os.makedirs(output_species_path, exist_ok=True)

        print(f"Processing: {species}")

        for audio in os.listdir(species_path):
            audio_name = os.path.splitext(audio)[0]
            save_path = os.path.join(output_species_path, f"{audio_name}.pt")
            if os.path.exists(save_path):
                continue

            audio_path = os.path.join(species_path, audio)
            waveform, sr = torchaudio.load(audio_path)

            if sr != sample_rate:
                resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=sample_rate)
                waveform = resampler(waveform)

            spec = generate_spectrogram(waveform, sample_rate)
            torch.save(spec, save_path)

    return "✅ Spettrogrammi generati e salvati."


def get_species_dict(labels_path):
    from pathlib import Path

    all_species = Path(labels_path).read_text(encoding="utf-8").splitlines()
    # maps every scientific name to its common name
    species_dict = {}
    for specie in all_species:
        scientific_name, common_name = specie.split("_")    # <Abroscopus albogularis>_<Rufous-faced Warbler>
        species_dict[scientific_name] = common_name

    return species_dict

# estae audio e category annots
def get_audio_category_annots(bird_tags_filepath, audio_source_path, species_dict):
    bird_tags = scipy.io.loadmat(bird_tags_filepath)["Bird_tags"]
    category_annots = {}      # detections grouped by category
    audio_annots = {}         # detections grouped by audio
    for elem in bird_tags:
        tag = elem[0][0][0][0][0]
        scientific_name = tag.replace("_", " ")                 # Fringilla_coelebs -> Fringilla coelebs
        common_name = species_dict.get(scientific_name, "")     # Fringilla coelebs -> Common Chaffinch
        label = "_".join([scientific_name, common_name])        # Fringilla coelebs_Common Chaffinch

        if not common_name:             # this happens only for non-species class, like "Wind_" and "Vegetation_"
            label = scientific_name     # as they don't have a common name, we use the scientific name as label

        file_name = elem[0][0][0][1][0]                         
        file_path = os.path.join(audio_source_path, file_name)   # path to the audio file

        start_time, end_time = np.array(elem[0][0][0][2]).flatten()[-2:]
        duration = end_time - start_time
        
        if not os.path.exists(file_path):       # do not store if file does not exist 
            continue
        if label not in category_annots:
            category_annots[label] = []
        if file_name not in audio_annots:
            audio_annots[file_name] = []

        category_annots[label].append({ "file_name": file_name, "start_time": start_time, "duration": duration, "label": label  })
        audio_annots[file_name].append({ "scientific_name": scientific_name, "common_name": common_name, "start_time": start_time, "duration": duration, "label": label })

    return category_annots, audio_annots

# store info about duration and sampling rate of the given audio
def generate_audio_info(source_audio_path, audio_annots):
    audio_info = {}
    audios = list(audio_annots.keys())
    for audio in audios:
        y, sr = librosa.load(os.path.join(source_audio_path, audio), sr=None)
        duration = librosa.get_duration(y=y, sr=sr)
        audio_info[audio] = {"duration": duration, "sampling_rate": sr}
    return audio_info

# generates the true segments for each audio file
def generate_true_segments(audio_annots, audio_info):
    true_segments = {}
    audios = list(audio_annots.keys())
    segment_length = 3.0    # length of each segment 
    step_size = 1.5         # overlap between segments 

    for audio in audios:
        # load annotations for this audio
        all_annotations = audio_annots[audio]               
        audio_duration = audio_info[audio]["duration"]
        true_segments[audio] = {}

        # generate all empty segments every 1.5 seconds
        segment_start = 0.0
        while segment_start + segment_length <= audio_duration:
            segm_id = f"{int(segment_start)}_{int((segment_start % 1) * 10)}"
            true_segments[audio][segm_id] = ["None"]      # empty list that will be filled with annotations
            segment_start += step_size              # move the segment forward by 1.5s

        # assign the annotations to the corresponding segments
        for annotation in all_annotations:
            start_time = annotation["start_time"]
            duration = annotation["duration"]
            if duration < 0.5:
                continue
            species = annotation["label"]

            # find the time interval where this annotation is present
            annotation_end = start_time + duration

            # check which segments contain it at least partially
            for segment_start_str in true_segments[audio].keys():
                segment_start_time = float(segment_start_str.replace("_", "."))  # convert from string to float
                segment_end_time = segment_start_time + segment_length

                # if the annotation falls at least partially within this segment, add it
                if not (annotation_end <= segment_start_time or start_time >= segment_end_time):
                    if species not in true_segments[audio][segment_start_str]:
                        true_segments[audio][segment_start_str].append(species)
                        if "None" in true_segments[audio][segment_start_str]:
                            true_segments[audio][segment_start_str].remove("None")
    return true_segments

# create segment file given a species name
def generate_species_segment(segment_audio, species_name, target_path, basename, segm_id):
    os.makedirs(os.path.join(target_path, species_name), exist_ok=True)
    export_path = os.path.join(
        target_path,
        species_name, 
        f"{basename}_{segm_id}.wav"
    )
    if os.path.exists(export_path):
        return
    segment_audio.export(export_path, format="wav")

# generate the audio from the true_segments
def generate_segments(audio_source_path, target_path, true_segments, audio_info, generate_None=False):
    os.makedirs(target_path, exist_ok=True)
    for audio_path, segms in true_segments.items():     # { <audio_path>.wav: { <segm_id>: [<species>] } }
        basename = os.path.splitext(audio_path)[0]      # removes ".wav"
        progress_bar = tqdm(total=len(segms), colour='red', desc=f"Processing segments for {audio_path}...")
        # loads the audio
        audio = AudioSegment.from_file(                 
                os.path.join(audio_source_path, audio_path),    
                format="wav",
                frame_rate=audio_info[audio_path]["sampling_rate"]
            )
        for segm_id, species in segms.items():          # <segm_id>: [<species>]
            segment_start_time = float(segm_id.replace("_", "."))
            segment_audio = audio[segment_start_time*1000:segment_start_time*1000 + 3000]
            if not species and generate_None:           # if the segment is empty, generate a None segment
                generate_species_segment(segment_audio, "None", target_path, basename, segm_id)
            for sp in species:
                generate_species_segment(segment_audio, sp, target_path, basename, segm_id)
            progress_bar.update(1)
        progress_bar.close()