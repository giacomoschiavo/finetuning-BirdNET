import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
import json 

home_page = st.Page("homepage.py", title="Home", icon="🏠")
models_page = st.Page("models_page.py", title="Models", icon="📊")
species_page = st.Page("species_page.py", title="Species", icon="🔬")
single_species_analysis = st.Page("single_species_analysis.py", title="Single Species Analysis", icon="🔍")
dataset_audio_analysis = st.Page("dataset_audio_analysis.py", title="Dataset Audio Analysis", icon="🔍")

pages = [home_page, models_page, species_page, single_species_analysis, dataset_audio_analysis]
selected_page = st.navigation(pages)
selected_page.run()
