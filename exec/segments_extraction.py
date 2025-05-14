import json
import os
import argparse
import copy
import sys
sys.path.append('../')
from birdlib import utils

def save_json(data, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as f:
        json.dump(data, f)

def parse_args():
    parser = argparse.ArgumentParser(
        description="Extract segments from an annotation file."
    )
    parser.add_argument('-s', '--source', required=True, help="Path to the directory containing raw audio files.")
    parser.add_argument('--annots_mat', required=True, help="Path to the .mat file containing annotations.")
    parser.add_argument('--labels_path', required=True, help="Path to the BirdNET's labels file containing all species names.")
    parser.add_argument('--utils_path',
        help="Directory to save intermediate metadata files like species_dict, annotations, and audio_info. Default is './utils'.")
    parser.add_argument('-o', '--output',
        help="Directory where the final segmented dataset will be saved.")
    parser.add_argument('--suffix', help="Suffix to append in every generated file, after the underscore (_). Default is 'train'.")

    return parser.parse_args()

def main(args):
    species_dict = utils.get_species_dict(args.labels_path)
    print("Created Species Dict")
    
    category_annots, audio_annots, missing = utils.get_audio_category_annots(args.annots_mat, args.source, species_dict)
    print("Created Annots")

    if len(missing) != 0:
        print(f"These files are missing: {', '.join(missing)}") 

    utils_path = os.path.join(args.utils_path)
    os.makedirs(utils_path, exist_ok=True)
    print("Extracting audio info...")
    # audio_info = utils.load_or_generate_info(f"audio_info_{args.suffix}.json", audio_annots, args.source, utils_path)
    with open(f'/home/giacomoschiavo/finetuning-BirdNET/utils_1/audio_info_{args.suffix}.json') as f:
        audio_info = json.load(f)

    true_segments = utils.generate_true_segments(audio_annots, audio_info)
    save_json(true_segments, os.path.join(args.utils_path, f"true_segments_{args.suffix}.json"))
    print(f"Saved true_segments_{args.suffix}.json to {args.utils_path}")

    print(f"Generating segments in {args.output}")
    utils.generate_segments(args.source, args.output, true_segments, audio_info, generate_None=True)
    print("Done!")


if __name__ == '__main__':
    args = parse_args()
    main(args)