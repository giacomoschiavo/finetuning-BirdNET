import os
import torchaudio
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import torch.optim as optim
import importlib
import json
import scipy
import numpy as np
import librosa
from pydub import AudioSegment
from tqdm import tqdm

def get_mappings(train_path):
    train_species = os.listdir(train_path) 
    filtered_species = [species for species in train_species]
    mappings = {species: i for i, species in enumerate(filtered_species)}
    return mappings

def load_or_generate_info(filename, annots, audio_source, save_path):
    full_path = os.path.join(save_path, filename)
    info = generate_audio_info(audio_source, annots)
    with open(full_path, 'w') as f:
        json.dump(info, f)
    return info

def load_model_class(model_name):
    model_module = importlib.import_module(f"models.{model_name}.model")
    model_class = getattr(model_module, model_name)
    return model_class

def collect_samples(train_path, valid_path, test_path, mappings):
    # {
    #     "file_path": "Aeroplane/20190621_070000_472_5.wav",
    #     "split": "train",
    #     "labels": [
    #         16,
    #         2
    #     ]
    # },
    samples_train = {}
    for species in os.listdir(train_path):
        if species not in mappings:
            continue
        audio_list = os.listdir(os.path.join(train_path, species))
        for audio in audio_list:
            if audio in samples_train:      # same audio in different folders, save the other species
                samples_train[audio]["labels"].append(mappings[species])
                continue                    # do not save another sample of the same audio
            samples_train[audio] = {
                "file_path": os.path.join(train_path, species, audio),
                "split": "train",
                "labels": [mappings[species]]
            }


    samples_valid = {}
    for species in os.listdir(valid_path):
        if species not in mappings:
            continue
        for audio in os.listdir(os.path.join(valid_path, species)):
            labels = [mappings[species]]
            updated = False
            if audio in samples_valid:       # adds label
                samples_valid[audio]["labels"].extend(labels)
                updated = True
            if audio in samples_train:      # considers training labels
                samples_train[audio]["labels"].extend(labels)       # add test label
                labels = samples_train[audio]["labels"]             # save new labels
            if updated:
                continue
            samples_valid[audio] = {
                "file_path": os.path.join(valid_path, species, audio),
                "split": "valid",
                "labels": labels
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
            if audio in samples_valid:      # considers training labels
                samples_valid[audio]["labels"].extend(labels)       # add test label
                labels = samples_valid[audio]["labels"]             # save new labels
            if updated:
                continue
            samples_test[audio] = {
                "file_path": os.path.join(test_path, species, audio),
                "split": "test",
                "labels": labels
            }

    samples = []
    for _, props in samples_train.items():
        samples.append(props)
    for _, props in samples_valid.items():
        samples.append(props)
    for _, props in samples_test.items():
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
    missing_file_name = set()
    for elem in bird_tags:
        tag = elem[0][0][0][0][0]
        scientific_name = tag.replace("_", " ")                 # Fringilla_coelebs -> Fringilla coelebs
        common_name = species_dict.get(scientific_name, "")     # Fringilla coelebs -> Common Chaffinch
        label = "_".join([scientific_name, common_name])        # Fringilla coelebs_Common Chaffinch

        if not common_name:             # this happens only for non-species class, like "Wind_" and "Vegetation_"
            label = scientific_name     # as they don't have a common name, we use the scientific name as label

        file_name = elem[0][0][0][1][0]                         
        custom_name = file_name
        if '-' in file_name:
            file_name = file_name.split(' - ')[0] + '_0.WAV'
        file_path = os.path.join(audio_source_path, file_name)   # path to the audio file
        if not os.path.exists(file_path):       # do not store if file does not exist
            missing_file_name.add(custom_name) 
            continue

        start_time, end_time = np.array(elem[0][0][0][2]).flatten()[-2:]
        duration = end_time - start_time
        
        if label not in category_annots:
            category_annots[label] = []
        if file_name not in audio_annots:
            audio_annots[file_name] = []
        category_annots[label].append({ "file_name": file_name, "start_time": start_time, "duration": duration, "label": label  })
        audio_annots[file_name].append({ "scientific_name": scientific_name, "common_name": common_name, "start_time": start_time, "duration": duration, "label": label })

    return category_annots, audio_annots, missing_file_name

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
            if duration < 0.25:          # <----------  FILTERS AUDIO LENGTH
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

def get_date_count(path, species_list):
    dates_count = {}
    for species in species_list:
        species_audio = os.listdir(os.path.join(path, species))
        dates_count.setdefault(species, {})
        for audio in species_audio:
            date = audio.split('_')[0]
            if date not in dates_count[species]:
                dates_count[species][date] = 0
            dates_count[species][date] = dates_count[species][date] + 1
    return dates_count

def split_dataset(species_data, folder_source_path, test_ratio=0.2, random_seed=42):
    """
    Split the dataset into training and testing sets by moving 
    exactly up to 20% of total examples to the test set.
    
    Args:
    - species_data (dict): Dictionary with species as keys and 
      day-level audio counts as values
    - test_ratio (float): Proportion of examples to move to test set (default 0.2)
    - random_seed (int): Random seed for reproducibility
    
    Returns:
    - tuple: (train_dataset, test_dataset) with the same structure as input
    """
    # Prepare output dictionaries
    train_dataset = {}
    test_dataset = {}
    
    for species, day_counts in species_data.items():
        # If only one day or very few examples, keep all in training
        total_examples = sum(day_counts.values())
        if len(day_counts) == 1:
            train_dataset[species] = day_counts.copy()
            test_dataset[species] = {}
            continue
        
        # Calculate maximum number of examples to move to test set
        max_test_examples = int(total_examples * test_ratio)
        
        # Prepare a list of days with their counts, sorted by count in descending order
        # This helps us preferentially select days with more examples first
        day_list = sorted(day_counts.items(), key=lambda x: x[1], reverse=True)
        
        # Initialize tracking variables
        train_dataset[species] = {}
        test_dataset[species] = {}
        test_examples_count = len(os.listdir(os.path.join(folder_source_path, species)))
        
        # Distribute examples
        for day, count in day_list:
            # If adding this day would exceed max test examples, keep it in training
            if test_examples_count + count > max_test_examples:
                train_dataset[species][day] = count
            else:
                # Add to test set
                test_dataset[species][day] = count
                test_examples_count += count
        
        # If no examples were moved to test set, move some from the largest day
        if test_examples_count == 0 and total_examples > 10:
            smallest_day = list(day_counts.keys())[-1]
            smallest_count = day_counts[smallest_day]
            partial_count = max(1, int(smallest_count * test_ratio))
            
            test_dataset[species][smallest_day] = partial_count
            train_dataset[species][smallest_day] = smallest_count - partial_count
        
        # Sanity check to ensure we've distributed all examples
        assert sum(train_dataset[species].values()) + sum(test_dataset[species].values()) == total_examples
    
    return train_dataset, test_dataset

def move_by_date(dates_division, source_path, dest_path):
    for species in dates_division:
        for audio in os.listdir(os.path.join(source_path, species)):
            day = audio.split('_')[0]
            if day in dates_division[species]:
                print(audio)
                os.rename(
                    os.path.join(source_path, species, audio),
                    os.path.join(dest_path, species, audio)
                )

def create_dataset_config(train_path, valid_path, test_path, dataset_name, config_file_name='dataset_config.json'):
    saving_path = f"utils/{dataset_name}/{config_file_name}"
    if os.path.exists(saving_path):
        print("Dataset config already created!")
        with open(saving_path) as f:
            return json.load(f)

    mappings = get_mappings(train_path)
    samples = collect_samples(train_path, valid_path, test_path, mappings)

    dataset_config = {
        "mappings": mappings,
        "samples": samples
    }
    with open(saving_path, "w") as f:
        json.dump(dataset_config, f)
    print("Saved new dataset config")
    return dataset_config

def train_model(train_loader, valid_loader, model, model_name, dataset_var, epochs=200, lr=1e-4, patience=3, early_stop_patience=15, load_weights=False):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=patience
    )
    history_loss = []
    history_valid_loss = []
    best_loss = float("inf")

    saving_path = f'models/{model_name}/{dataset_var}/checkpoint.pth'
    if load_weights:
        if not os.path.exists(saving_path):
            print("No weights found!")
            return None
        checkpoint = torch.load(saving_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        history_loss = checkpoint['history_loss']
        history_valid_loss = checkpoint['history_valid_loss']
        best_loss = checkpoint['best_loss']
        print(f"Model Loaded!")
            
    print(f"Training #{dataset_var} started!")
    model.train()
    early_stop_counter = 0
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0

        for batch_index, (mel_spec, labels, _) in enumerate(train_loader):
            mel_spec = mel_spec.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = model(mel_spec)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            
        train_loss = running_loss / len(train_loader)
        print(f"Epoch {epoch + 1} - Train Loss: {train_loss:.5f}")

        # Validation phase
        model.eval()
        with torch.no_grad():
            valid_loss = 0.0
            for mel_spec, labels, _ in valid_loader:
                mel_spec = mel_spec.to(device)
                labels = labels.to(device)
                outputs = model(mel_spec)
                loss = criterion(outputs, labels)
                valid_loss += loss.item()

            valid_loss /= len(valid_loader)
        scheduler.step(valid_loss)
        
        if valid_loss < best_loss:
            best_loss = valid_loss
            early_stop_counter = 0
            history_loss.append(train_loss)
            history_valid_loss.append(valid_loss)
            print(f"💾 Saving improved model at epoch {epoch+1} with Valid loss={valid_loss:.5f}")
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'history_valid_loss': history_valid_loss,
                'history_loss': history_loss,
                'best_loss': best_loss,
            }, saving_path)
        else:
            early_stop_counter += 1
            print(f"🛑 No improvement — early stop counter: {early_stop_counter}/{early_stop_patience}")

        print(f"🔁 Epoch {epoch+1} completed - valid loss: {valid_loss:.7f} - LR: {optimizer.param_groups[0]['lr']:.1e}")

        if early_stop_counter >= early_stop_patience:
            print(f"\n🚨 Early stopping triggered after {early_stop_patience} epochs without improvement.")
            break

        np.save(f'models/{model_name}/{dataset_var}/history_loss.npy', history_loss)
        np.save(f'models/{model_name}/{dataset_var}/history_valid_loss.npy', history_valid_loss)

    torch.save(model.state_dict(), f"models/{model_name}/{dataset_var}/final_weights.pth")
    print("✅ Training completed")


    return model
