# This page is intended to investigate audio in a given dataset (given path)

import streamlit as st
import os
import torchaudio
import librosa
import matplotlib.pyplot as plt
import numpy as np
import json

selected_dataset = st.selectbox("Select a dataset:", ["NEW_DATASET", "NEW_DATASET_1"])

DATASET_NAME = "NEW_DATASET"
DATASET_PATH = f"E:/Giacomo/Tovanella/{selected_dataset}"

with open(f"../utils/{DATASET_NAME}/true_segments.json") as f:
    true_segments = json.load(f)

species_list = os.listdir(f"{DATASET_PATH}/test")
selected_species = st.selectbox("Select a species:", species_list)

st.header("Train Audio Analysis")
audio_list = os.listdir(f"{DATASET_PATH}/train/{selected_species}")
selected_audio = st.selectbox("Select audio:", audio_list)

def show_mel_spec(chosen_audio_path):
    min_db = -80 
    max_db = 10   
    waveform, sr = torchaudio.load(chosen_audio_path)
    waveform = waveform.numpy().squeeze(0)  # Converti il tensore in numpy
    stft = np.abs(librosa.stft(waveform))
    stft_db = librosa.amplitude_to_db(stft, ref=np.max)
    fig, ax = plt.subplots(figsize=(6, 4))
    img = librosa.display.specshow(stft_db, sr=sr, x_axis='time', y_axis='linear', vmin=min_db, vmax=max_db, cmap='Spectral')
    plt.colorbar(label='dB')
    ax.set(title='Mel Spectrogram (dB)')
    fig.tight_layout()
    st.pyplot(fig)

def show_true_label(chosen_audio_path):
    only_audio_path = os.path.basename(chosen_audio_path)
    audio_name = os.path.splitext(only_audio_path)[0]
    if len(audio_name.split("_")) == 4:
        date, time, segm1, segm2 = audio_name.split("_")
        segm = "_".join([segm1, segm2])
    else:
        date, time, segm = audio_name.split("_")
    true_labels = true_segments["_".join([date, time]) + ".WAV"][segm]
    st.markdown("**True labels**: " + ", ".join([label for label in true_labels]))


full_audio_path = f"{DATASET_PATH}/train/{selected_species}/{selected_audio}"
st.audio(full_audio_path)
show_mel_spec(full_audio_path)
show_true_label(full_audio_path)