import segments_extraction
import argparse
import os
from argparse import Namespace
from birdlib import utils


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
    DATASET_PATH = args.output
    TRAIN_PATH = f'{DATASET_PATH}/train'
    VALID_PATH = f'{DATASET_PATH}/valid'
    TEST_PATH = f'{DATASET_PATH}/test'
    print(TRAIN_PATH, VALID_PATH, TEST_PATH)
    os.makedirs(VALID_PATH, exist_ok=True)
    species_list = set(os.listdir(TEST_PATH)).intersection(set(os.listdir(TRAIN_PATH)))
    if 'Engine_Engine' in species_list:
        species_list.remove('Engine_Engine')     

    # TEST INTEGRATION
    print("Integrating test set...")
    species_to_integrate = []
    for species in species_list:
        species_audio = os.listdir(os.path.join(TEST_PATH, species))
        if len(species_audio) < 100:
            species_to_integrate.append(species)
    
    dates_count = utils.get_date_count(TRAIN_PATH, species_list)
    train_integration, test_integration = utils.split_dataset(dates_count, TEST_PATH, test_ratio=0.2)
    utils.move_by_date(test_integration, TRAIN_PATH, TEST_PATH)

    print("Creating validation set...")
    # VALIDATION
    dates_count_valid = utils.get_date_count(TRAIN_PATH, species_list)
    for species in species_list:
        os.makedirs(os.path.join(VALID_PATH, species), exist_ok=True)

    train_split, valid_split = utils.split_dataset(dates_count_valid, VALID_PATH, test_ratio=0.1)
    utils.move_by_date(valid_split, TRAIN_PATH, VALID_PATH)

    valid_species = os.listdir(VALID_PATH)
    REMOVED_PATH = f'{DATASET_PATH}/removed'
    os.makedirs(REMOVED_PATH, exist_ok=True)
    REMOVED_TRAIN_PATH = f'{REMOVED_PATH}/train'
    os.makedirs(REMOVED_TRAIN_PATH, exist_ok=True)
    for species in os.listdir(TRAIN_PATH):
        if species not in valid_species:
            os.makedirs(os.path.join(REMOVED_TRAIN_PATH, species), exist_ok=True)
            os.rename(
                os.path.join(TRAIN_PATH, species),
                os.path.join(REMOVED_TRAIN_PATH, species)
            )
    valid_species = os.listdir(VALID_PATH)
    REMOVED_TEST_PATH = f'{REMOVED_PATH}/test'
    os.makedirs(REMOVED_TEST_PATH, exist_ok=True)
    for species in os.listdir(TEST_PATH):
        if species not in valid_species:
            os.makedirs(os.path.join(REMOVED_TEST_PATH, species), exist_ok=True)
            os.rename(
                os.path.join(TEST_PATH, species),
                os.path.join(REMOVED_TEST_PATH, species)
            )

if __name__ == "__main__":
    args = parse_args()
    main(args)
