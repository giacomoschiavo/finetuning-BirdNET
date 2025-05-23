import streamlit as st
import pandas as pd
import os
import json
import matplotlib.pyplot as plt


st.title("Model Analysis")

models_folder_path = '../classifiers/official'
model_names = os.listdir(models_folder_path)
comparable_models = []
for model in model_names:
    if os.path.exists(os.path.join(models_folder_path, model, "classification_report_1.json")):
        comparable_models.append(model)

# load all classification report
model_reports = {}
for model_name in comparable_models:
    with open(f'{models_folder_path}/{model_name}/classification_report_1.json') as f:
        classification_report = json.load(f)
        model_reports[model_name] = classification_report

model = st.selectbox("Seleziona un modello:", list(model_reports.keys()))
df = pd.DataFrame(model_reports[model]).T
df = df.sort_values(by="f1-score", ascending=False)
st.dataframe(df)  

roc_auc_reports = {}
for model_name in comparable_models:
    try:
        f = open(f'{models_folder_path}/{model_name}/roc_auc_1.json')
        roc_auc_values = json.load(f)
        roc_auc_reports[model_name] = roc_auc_values
    except:
        roc_auc_reports[model_name] = {}

st.subheader("ROC AUC Curves per Species")
roc_auc_data = roc_auc_reports[model]
fig, ax = plt.subplots()
for species in roc_auc_data:
    fpr = roc_auc_data[species]['fpr']
    tpr = roc_auc_data[species]['tpr']
    auc = roc_auc_data[species]['auc']
    ax.plot(fpr, tpr, label=f'{species} (AUC = {auc:.2f})')

plt.figure(figsize=(20, 18))  
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.tight_layout()
st.pyplot(fig)

