import os
import json
import streamlit as st
import pandas as pd
import torchaudio.transforms as T
import torchaudio
import librosa
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
from utils import count_file_in, show_audio, show_mel_spec, show_true_labels

DATASET_NAME = "NEW_DATASET_1"
DATASET_PATH = f"E:/Giacomo/Tovanella/{DATASET_NAME}/"
MODEL_NAME = "pasqualo"


st.title("Single Species Analysis")
with open("../utils/category_annots.json") as f:
    category_info = json.load(f)

# load all species names
# species_list = [species for species in category_info.keys()]
species_list = os.listdir(f"{DATASET_PATH}/test")
species_list = sorted(species_list)
selected_species = st.selectbox("Select a species:", species_list)

SPECIES_TRAIN_PATH = f"{DATASET_PATH}/train/{selected_species}"
SPECIES_VALID_PATH = f"{DATASET_PATH}/valid/{selected_species}"
SPECIES_TEST_PATH = f"{DATASET_PATH}/test/{selected_species}"

# load dataframe with metrics
metrics_df = pd.read_json(f"../classifiers/official/{MODEL_NAME}/classification_report.json", orient="index")
st.dataframe(metrics_df[metrics_df.index == selected_species])

# count division by species
species_count_df = pd.read_csv(f"../utils/{DATASET_NAME}/species_count_df.csv", index_col=0)
species_count_df.columns = ["Train sample count", "Test sample counts"]

st.markdown(f"File Count for {selected_species}:")
st.markdown(f"- 📂 Training set: **{count_file_in(SPECIES_TRAIN_PATH)}**")
st.markdown(f"- 📁 Validation set: **{count_file_in(SPECIES_VALID_PATH)}**")
st.markdown(f"- 🗂️ Test set: **{count_file_in(SPECIES_TEST_PATH)}**")


with open(f"../utils/{DATASET_NAME}/true_segments.json") as f:
    true_segments = json.load(f)

st.header("Training Audio Analysis")
chosen_audio_path = show_audio(SPECIES_TRAIN_PATH)
col1, col2 = st.columns([0.5, 0.5])
with col1:
    show_true_labels(chosen_audio_path, true_segments)
with col2:
    show_mel_spec(chosen_audio_path)

st.header("Test Audio Analysis")
st.markdown(f"Selected species: **{selected_species}**")

chosen_audio_test_path = show_audio(SPECIES_TEST_PATH)

with open(f"../classifiers/official/{MODEL_NAME}/test_pred_segments.json") as f:
    test_pred_segments = json.load(f)

# get all the other species predicted in the same segment
audio_name = os.path.basename(os.path.splitext(chosen_audio_test_path)[0])  # 20190608_100000_21_0
if len(audio_name.split("_")) == 4:
    date, hour, segm1, segm2 = audio_name.split("_")
    segm = "_".join([segm1, segm2])
    audio_name = "_".join([date, hour]) + ".WAV"
else:
    st.write("Audio Name error format")
other_predictions = test_pred_segments[audio_name][segm]
other_pred_df = pd.DataFrame(list(other_predictions.items()), columns=["Species", "Confidence Score"])
st.dataframe(other_pred_df)
show_mel_spec(chosen_audio_test_path)
