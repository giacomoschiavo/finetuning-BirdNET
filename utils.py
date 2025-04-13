import os
import torchaudio
import torch
import torch.nn.functional as F
import random

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
            spectrogram = spectrogram.unsqueeze(0).unsqueeze(0)  # (1, 1, freq, time)
            print(audio)
            spectrogram = F.interpolate(spectrogram, size=(256, 256), mode="bilinear", align_corners=False)
            spectrogram = spectrogram.squeeze(0).squeeze(0)  # torna a (256, 256)

            # Save the spectrogram tensor
            torch.save(spectrogram, save_path)

def collect_samples(train_path, test_path, mappings):
    samples_train = {}
    for species in os.listdir(train_path):
        if species not in os.listdir(test_path):
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
    samples = []
    for _, props in samples_train.items():
        samples.append(props)
    for _, props in samples_test.items():
        samples.append(props)
    return samples
