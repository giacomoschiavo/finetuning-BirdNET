import scipy.io
import numpy as np
import math
import os
from pydub import AudioSegment
import numpy as np
import json
from tqdm import tqdm
import pandas as pd
from segments_creator import SegmentCreator
from pathlib import Path
import argparse
import os
import sys  # For exiting the script gracefully

# In the same folder of the script
#   | Tovanella dataset
#   | Annotations file (Bird_tags_Train.mat)
#   | Species Mapping file (BirdNET_GLOBAL_6K_V2.4_Labels_en_uk.txt) from BirdNET's Github

# DATASET_NAME = 'NEW_DATASET_1'
# DATASET_PATH = f'E:/Giacomo/Tovanella/{DATASET_NAME}'
# AUDIO_SOURCE = 'E:/Giacomo/Tovanella/Tovanella'

# 1_segments_extraction.py

def validate_path(path):
    """Checks if a given path exists."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Error: Path '{path}' does not exist.")
    return path

def validate_directory(path):
    """Checks if a given path is a directory."""
    path = validate_path(path)
    if not os.path.isdir(path):
        raise NotADirectoryError(f"Error: Path '{path}' is not a valid directory.")
    return path

def main():
    parser = argparse.ArgumentParser(description="Populates variables from the terminal")
    parser.add_argument("--dataset_name", required=True, help="Name of the dataset")
    parser.add_argument("--dataset_path", required=True, help="Path to the dataset")
    parser.add_argument("--source_audio_path", required=True, help="Path to the audio files")

    try:
        args = parser.parse_args()

        DATASET_NAME = validate_directory(args.dataset)
        DATASET_PATH = validate_directory(args.path)
        AUDIO_SOURCE = validate_directory(args.audio)

        print(f"Dataset: {DATASET_NAME}")
        print(f"Dataset Path: {DATASET_PATH}")
        print(f"Audio Path: {AUDIO_SOURCE}")

    except argparse.ArgumentError as e:
        print(f"Error in command-line arguments: {e}")
        sys.exit(1)  # Exit with a non-zero status code to indicate an error
    except FileNotFoundError as e:
        print(e)
        sys.exit(1)
    except NotADirectoryError as e:
        print(e)
        sys.exit(1)
    except ValueError as e:
        print(e)
        sys.exit(1)
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()

# all_species = Path("utils/BirdNET_GLOBAL_6K_V2.4_Labels_en_uk.txt").read_text(encoding="utf-8").splitlines()
# species_dict = {}
# for specie in all_species:
#     scientific_name, common_name = specie.split("_")    # <Abroscopus albogularis>_<Rufous-faced Warbler>
#     species_dict[scientific_name] = common_name

# def get_audio_category_info(bird_tags_filepath, audio_source_path):
#     bird_tags = scipy.io.loadmat(bird_tags_filepath)["Bird_tags"]
#     category_info = {}      # detections grouped by category
#     audio_info = {}         # detections grouped by audio
#     for elem in bird_tags:
#         # get <scientific name>_<common name> format 
#         tag = elem[0][0][0][0][0]
#         scientific_name = tag.replace("_", " ")                 # Fringilla coelebs -> Fringilla coelebs
#         common_name = species_dict.get(scientific_name, "")     # Fringilla coelebs -> Common Chaffinch
#         label = "_".join([scientific_name, common_name])        # Fringilla coelebs_Common Chaffinch
#         # get source file
#         file_name = elem[0][0][0][1][0]
#         file_path = os.path.join(audio_source_path, file_name)
#         # duration calculation
#         start_time, end_time = np.array(elem[0][0][0][2]).flatten()[-2:]
#         duration = end_time - start_time
#         # do not store info if file do not exist
#         if not os.path.exists(file_path):   
#             continue
#         # save in dictionaries
#         if label not in category_info:
#             category_info[label] = []
#         if file_name not in audio_info:
#             audio_info[file_name] = []
#         category_info[label].append({ "file_name": file_name, "start_time": start_time, "duration": duration, "label": label  })
#         audio_info[file_name].append({ "scientific_name": scientific_name, "common_name": common_name, "start_time": start_time, "duration": duration, "label": label })
#     return category_info, audio_info

# category_info, audio_info = get_audio_category_info("Bird_tags_Train.mat", AUDIO_SOURCE)

# os.makedirs("utils", exist_ok=True)
# with open("utils/category_info.json", "w") as f:
#     json.dump(category_info, f)
# with open("utils/audio_info.json", "w") as f:
#     json.dump(audio_info, f)
