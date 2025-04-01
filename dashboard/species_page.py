import streamlit as st
import pandas as pd
import os
import json
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from io import StringIO


st.title("Species Analysis")

models_folder_path = '../classifiers/official'
model_names = os.listdir(models_folder_path)
comparable_models = []
for model in model_names:
    if os.path.exists(os.path.join(models_folder_path, model, "classification_report_1.json")):
        comparable_models.append(model)

# load all classification report
classification_reports = {}
for model_name in comparable_models:
    with open(f'{models_folder_path}/{model_name}/classification_report_1.json') as f:
        classification_report = json.load(f)
        classification_reports[model_name] = classification_report

with open("../classifiers/official/original/CustomClassifier_Labels.txt") as f:
    all_species = [line.strip() for line in f.readlines()]

average_metrics_per_species = {}
for species in sorted(list(all_species)):
    precision_values = []
    recall_values = []
    f1_values = []
    support_values = []

    for model_name in comparable_models:
        if species in classification_reports[model_name]:
            metrics = classification_reports[model_name][species]
            precision_values.append(metrics['precision'])
            recall_values.append(metrics['recall'])
            f1_values.append(metrics['f1-score'])
            # support_values.append(metrics['support'])
            
    average_metrics_per_species[species] = {
        'average_precision': np.mean(precision_values),
        'average_recall': np.mean(recall_values),
        'average_f1-score': np.mean(f1_values),
        # 'average_support': np.mean(support_values)
    }

avg_metrics_df = pd.DataFrame(average_metrics_per_species).T
avg_metrics_df = avg_metrics_df.sort_values(by="average_f1-score", ascending=False)

def color_code(series):
    low = series.min()
    high = series.max()
    cmap = plt.cm.get_cmap('RdYlGn') # You can choose other colormaps like 'viridis', 'plasma', etc.
    norm = plt.Normalize(vmin=low, vmax=high)
    c = [matplotlib.colors.rgb2hex(cmap(norm(v))) for v in series]
    return ['background-color: %s' % color for color in c]

avg_metrics_df = avg_metrics_df.style.apply(color_code, subset=['average_precision', 'average_recall', 'average_f1-score'])

st.subheader("Average Metrics per Species")
st.dataframe(avg_metrics_df)

col1, col2 = st.columns([0.5, 0.5])

with col1:
    # BY DURATION
    st.subheader("Dataset by Duration")
    species_by_duration = pd.read_csv("../utils/species_by_duration.csv", index_col=0)
    species_by_duration = species_by_duration.sort_values("Total", ascending=False).reset_index(drop=True)

    st.bar_chart(data=species_by_duration, x="Species Name", y="Count", color="Duration", height=800, horizontal=True)

with col2:
    # BY SPLIT
    st.subheader("Dataset by Dates")
    by_dates_df = pd.read_csv("../utils/by_dates_df.csv", index_col=0)
    by_dates_df["Duration"] = by_dates_df["Duration"].astype('category')

    st.bar_chart(data=by_dates_df, x="Species Name", y="Count", color="Duration", height=800, horizontal=True)
