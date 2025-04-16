import os
import torchaudio
import torch
import torch.nn.functional as F
import random
from torch.utils.data import DataLoader, Dataset
import importlib
import json

def get_mappings(test_path):
    test_species = os.listdir(test_path) 
    with open("utils/category_annots.json") as f:
        category_annots = json.load(f)

    filtered_species = [species for species in category_annots.keys() if len(species.split("_")) > 1 and species in test_species]
    mappings = {species: i for i, species in enumerate(filtered_species)}
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


def specs_generation(input_path, output_path, mappings):
    sample_rate = 32000
    target_samples = sample_rate * 3
    n_fft = 1024
    hop_length = 256
    win_length = 1024

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

            if waveform.shape[1] < target_samples:
                pad_len = target_samples - waveform.shape[1]
                waveform = F.pad(waveform, (0, pad_len))
            else:
                waveform = waveform[:, :target_samples]

            waveform = waveform[0:1, :]  # keep solo il primo canale

            window = torch.hann_window(n_fft, device=waveform.device)

            stft = torch.stft(
                waveform,
                n_fft=n_fft,
                hop_length=hop_length,
                win_length=win_length,
                window=window,
                return_complex=True
            )
            spectrogram = torch.abs(stft)  # shape: (freq, time)

            # Optional log scaling or 
            spectrogram = torch.log1p(spectrogram)
            # spectrogram = (spectrogram - spectrogram.min()) / (spectrogram.max() - spectrogram.min())

            # Resize to 256x256
            spectrogram = spectrogram.unsqueeze(0) # (1, 1, freq, time)
            spectrogram = F.interpolate(spectrogram, size=(256, 256), mode="bilinear", align_corners=False)
            spectrogram = spectrogram.squeeze(0).squeeze(0)  # torna a (256, 256)

            # Save the spectrogram tensor
            torch.save(spectrogram, save_path)