import json
import os
import argparse
import copy
import sys
sys.path.append('../')
from birdlib import utils

def load_or_generate_info(filename, annots, audio_source, save_path):
    full_path = os.path.join(save_path, filename)
    if os.path.exists(full_path):
        with open(full_path) as f:
            return json.load(f)
    info = utils.generate_audio_info(audio_source, annots)
    with open(full_path, 'w') as f:
        json.dump(info, f)
    return info


def save_json(data, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as f:
        json.dump(data, f)

def parse_args():
    parser = argparse.ArgumentParser(
        description="Builds and prepares dataset segments for training and testing "
                    "from raw audio files and annotation .mat files."
    )
    parser.add_argument('--source', required=True, help="Path to the directory containing raw audio files.")
    parser.add_argument('--dataset-name', required=True, help="Name of the dataset (used as folder name for outputs).")
    parser.add_argument('--train-mat', required=True, help="Path to the .mat file containing training annotations.")
    parser.add_argument('--test-mat', required=True, help="Path to the .mat file containing testing annotations.")
    parser.add_argument('--labels-path', required=True, help="Path to the BirdNET's labels file containing all species names.")
    parser.add_argument('--meta-path', default="../utils",
        help="Directory to save intermediate metadata files like species_dict, annotations, and audio_info. Default is './utils'.")
    parser.add_argument('--output-path', default=None,
        help="Directory where the final segmented dataset will be saved. If not set, uses './<dataset-name>/' by default.")

    return parser.parse_args()

def main(args):
    species_dict = utils.get_species_dict(args.labels_path)
    print("Created Species Dict")
    
    category_annots, audio_annots = utils.get_audio_category_annots(args.train_mat, args.source, species_dict)
    category_annots_test, audio_annots_test = utils.get_audio_category_annots(args.test_mat, args.source, species_dict)
    print("Created Annots")


    utils_path = os.path.join(args.meta_path, args.dataset_name)
    os.makedirs(utils_path, exist_ok=True)
    audio_info = load_or_generate_info("audio_info.json", audio_annots, args.source, utils_path)
    audio_info_test = load_or_generate_info("audio_info_test.json", audio_annots_test, args.source, utils_path)

    true_segments_train = utils.generate_true_segments(audio_annots, audio_info)
    true_segments_test = utils.generate_true_segments(audio_annots_test, audio_info_test)

    true_segments = copy.deepcopy(true_segments_train)
    true_segments.update(true_segments_test)
    print("Created True Segments")

    save_json(true_segments_train, os.path.join(args.meta_path, args.dataset_name, "true_segments_train.json"))
    save_json(true_segments_test, os.path.join(args.meta_path, args.dataset_name, "true_segments_test.json"))
    save_json(true_segments, os.path.join(args.meta_path, args.dataset_name, "true_segments.json"))

    output_path = args.output_path or f"./{args.dataset_name}"

    print("Generating segments...")
    utils.generate_segments(args.source, f"{output_path}/train", true_segments_train, audio_info, generate_None=True)
    utils.generate_segments(args.source, f"{output_path}/test", true_segments_test, audio_info_test, generate_None=True)
    print("Done!")


if __name__ == '__main__':
    args = parse_args()
    main(args)