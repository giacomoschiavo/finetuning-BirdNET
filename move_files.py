import os
import json 

WABAD_PATH = "/home/giacomoschiavo/WABAD/audio"
WABAD_AUDIO_SOURCE = "/home/giacomoschiavo/WABAD/all_wabad_audio"

audio_annots_wabad = {}
category_annots_wabad = {}

with open("/home/giacomoschiavo/finetuning-BirdNET/utils/WABAD/audio_annots_wabad.json") as jsonfile:
    audio_annots_wabad = json.load(jsonfile)

with open("/home/giacomoschiavo/finetuning-BirdNET/utils/WABAD/category_annots_wabad.json") as jsonfile:
    category_annots_wabad = json.load(jsonfile)

for folder in os.listdir(WABAD_PATH):
    if not os.path.isdir(os.path.join(WABAD_PATH, folder)):
        continue
    # ...\BAM\BAM\Recordings
    folder_path = os.path.join(WABAD_PATH, folder, folder, "Recordings")
    all_audio = os.listdir(folder_path)
    for audio in all_audio:
        if audio.upper() in audio_annots_wabad.keys():
            os.rename(
                os.path.join(folder_path, audio),
                os.path.join(WABAD_AUDIO_SOURCE, audio)
            )