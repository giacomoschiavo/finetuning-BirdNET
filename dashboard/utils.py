import streamlit as st
import os
import torchaudio
import librosa
import matplotlib.pyplot as plt
import numpy as np

def count_file_in(path):
    return sum(1 for entry in os.scandir(path) if entry.is_file())

# Returns selected audio
def show_audio(path):
    audio_list = [entry.name for entry in os.scandir(path) if entry.is_file()]
    selected_audio = st.selectbox("Choose audio:", audio_list)
    audio_path = os.path.join(path, selected_audio)
    st.audio(audio_path)
    return audio_path

def show_mel_spec(chosen_audio_path):
    min_db = -80 
    max_db = 10   
    waveform, sr = torchaudio.load(chosen_audio_path)
    waveform = waveform.numpy().squeeze(0)  # Converti il tensore in numpy
    stft = np.abs(librosa.stft(waveform))
    stft_db = librosa.amplitude_to_db(stft, ref=np.max)
    fig, ax = plt.subplots(figsize=(6, 4))
    img = librosa.display.specshow(stft_db, sr=sr, x_axis='time', y_axis='linear', vmin=min_db, vmax=max_db, cmap='magma')
    plt.colorbar(label='dB')
    ax.set(title='Mel Spectrogram (dB)')
    fig.tight_layout()
    st.pyplot(fig)

def show_true_labels(chosen_audio_path, true_segments):
    only_audio_path = os.path.basename(chosen_audio_path)
    date, time, segm1, segm2 = os.path.splitext(only_audio_path)[0].split("_")
    segm = "_".join([segm1, segm2])
    true_labels = true_segments["_".join([date, time]) + ".WAV"][segm]
    st.markdown("##### True Labels")
    bulleted_labels = "\n".join([f"- {label}" for label in true_labels])
    st.markdown(bulleted_labels)
