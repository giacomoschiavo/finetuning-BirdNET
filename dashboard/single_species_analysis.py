import os
import json
import streamlit as st
import pandas as pd
import torchaudio.transforms as T
import torchaudio
import librosa
import matplotlib.pyplot as plt
import numpy as np

DATASET_NAME = "NEW_DATASET"
DATASET_PATH = f"E:/Giacomo/Tovanella/{DATASET_NAME}/"

removed_species = os.listdir(f"{DATASET_PATH}/removed")

st.title("Single Species Analysis")
with open("../utils/category_info.json") as f:
    category_info = json.load(f)

# load all species names
species_list = [species for species in category_info.keys()]
species_list = sorted(species_list)
selected_species = st.selectbox("Select a species:", species_list)
if selected_species in removed_species:
    st.write("⚠ Removed from the dataset!")

# load dataframe with metrics
avg_metrics_df = pd.read_csv("../utils/avg_metrics.json", index_col=0)
st.dataframe(avg_metrics_df[avg_metrics_df.index == selected_species])

# count division by species
species_count_sf = pd.read_csv(f"../utils/{DATASET_NAME}/species_count_df.csv", index_col=0)
st.dataframe(species_count_sf[species_count_sf.index == selected_species])

# load dataset division 
with open(f"../utils/{DATASET_NAME}/species_split.json") as f:
    species_split = json.load(f)

selected_split = species_split[selected_species]
col1, col2 = st.columns([0.5, 0.5])

def show_audio(split_name):
    chosen_audio_path = ""
    if len(selected_split[split_name]) > 0:
        selected_split[split_name] = sorted(selected_split[split_name])
        selected_audio = st.selectbox("Select a segment:", selected_split[split_name])
        file_audio_path = os.path.join(DATASET_PATH, split_name, selected_species, selected_audio)
        removed_file_audio_path = os.path.join(DATASET_PATH, "removed", selected_species, selected_audio)
        if os.path.exists(file_audio_path):
            st.audio(file_audio_path)
            chosen_audio_path = file_audio_path
        elif os.path.exists(removed_file_audio_path):
            st.audio(removed_file_audio_path)
            chosen_audio_path = removed_file_audio_path
        else:
            st.markdown("File not found :(")
    else:
        st.markdown("No audio files here :(")
    return chosen_audio_path

def show_mel_spec(chosen_audio_path):
    min_db = -80 
    max_db = 10   
    mel_transform = T.MelSpectrogram(
      n_mels=96,
      hop_length=512,
      f_min=500,
      f_max=15000,
      n_fft=1024,
      sample_rate=48000
    )
    waveform, sr = torchaudio.load(chosen_audio_path)
    mel_spec = mel_transform(waveform).squeeze(0)
    mel_spec = librosa.power_to_db(mel_spec, ref=np.max)
    fig, ax = plt.subplots(figsize=(6, 4))
    img = librosa.display.specshow(mel_spec, sr=sr, x_axis='time', y_axis='mel', vmin=min_db, vmax=max_db)
    ax.set(title='Mel Spectrogram (dB)')
    fig.colorbar(img, ax=ax, format='%+2.0f dB')
    fig.tight_layout()
    st.pyplot(fig)

def show_true_label(chosen_audio_path):
    only_audio_path = os.path.basename(chosen_audio_path)
    date, time, segm = os.path.splitext(only_audio_path)[0].split("_")
    true_labels = true_segments["_".join([date, time]) + ".WAV"][segm]
    st.markdown("True labels: " + ", ".join([label for label in true_labels]))

with open(f"../utils/{DATASET_NAME}/true_segments.json") as f:
    true_segments = json.load(f)

with col1:
    st.subheader("Training Audio Analysis")
    chosen_audio_path = show_audio("train")
    if chosen_audio_path != "":
        show_mel_spec(chosen_audio_path)
        show_true_label(chosen_audio_path)
with col2:
    st.subheader("Test Audio Analysis")
    chosen_audio_path = show_audio("test")
    if chosen_audio_path != "":
        show_mel_spec(chosen_audio_path)
        show_true_label(chosen_audio_path)

st.header("Test Audio - Deep Analysis")
st.markdown(f"Selected species: **{selected_species}**")

test_dataset = species_split[selected_species]["test"]      # TODO: fix "removed" folder case
if len(test_dataset) <= 0:
    st.write("No audio in test")
    pass

if selected_species not in removed_species:
    with open("../utils/mean_conf_scores.json") as f:
        mean_conf_scores = json.load(f)

    custom_mean_conf_scores = {}
    for audio in test_dataset:
        # 20190608_070000_63.wav
        date, hour, segm = os.path.splitext(audio)[0].split("_")
        audio_path = f"{date}_{hour}.WAV"
        custom_mean_conf_scores[audio] = mean_conf_scores.get(audio_path, {}).get(segm, {}).get(selected_species, 0)

    st.dataframe(custom_mean_conf_scores)

    test_dataset = sorted(test_dataset)
    selected_audio_test = st.selectbox("Select a segment:", test_dataset, key="test_audio_deep_analysis")

    date, hour, segm = os.path.splitext(selected_audio_test)[0].split("_")
    audio_path = f"{date}_{hour}.WAV"
    st.write(mean_conf_scores.get(audio_path, {}).get(segm, {}))
    full_audio_path = os.path.join(DATASET_PATH, "test", selected_species, selected_audio_test)
    show_true_label(chosen_audio_path)

    st.audio(full_audio_path)
    show_mel_spec(full_audio_path)