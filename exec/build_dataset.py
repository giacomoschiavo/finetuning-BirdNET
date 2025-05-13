import segments_extraction
import argparse
import os
from argparse import Namespace

def parse_args():
    parser = argparse.ArgumentParser(
        description="Builds and prepares dataset segments for training and testing "
                    "from raw audio files and annotation .mat files."
    )
    parser.add_argument('-s', '--source', required=True, help="Path to the directory containing raw audio files.")
    parser.add_argument('--annots_train', required=True, help="Path to the training annotations .mat file.")
    parser.add_argument('--annots_test', required=True, help="Path to the testing annotations .mat file.")
    parser.add_argument('--labels_path', required=True, help="Path to the BirdNET's labels file containing all species names.")
    parser.add_argument('--utils_path', help="Directory to save intermediate metadata files like species_dict, annotations, and audio_info. Default is './utils'.")
    parser.add_argument('-o', '--output', help="Directory where the final segmented dataset will be saved.")

    return parser.parse_args()

def main(args):
    # Train arguments
    train_args = Namespace(
        source=args.source,
        annots_mat=args.annots_train,
        labels_path=args.labels_path,
        utils_path=args.utils_path,
        output=os.path.join(args.output, "train"),
        suffix="train"
    )
    print("Processing train set...")
    segments_extraction.main(train_args)

    # Test arguments
    test_args = Namespace(
        source=args.source,
        annots_mat=args.annots_test,
        labels_path=args.labels_path,
        utils_path=args.utils_path,
        output=os.path.join(args.output, "test"),
        suffix="test"
    )
    print("Processing test set...")
    segments_extraction.main(test_args)

if __name__ == "__main__":
    args = parse_args()
    main(args)
