from pathlib import Path

from prediction import predict
from preprocessing import preprocess
from training import train
from random import randint
import argparse


def load_file(filename):
    with open(filename) as f:
        return f.read()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--training', default='no', type=str, help='training')
    args = parser.parse_args()

    input_document = "the_republic_clean.txt"
    output_document = "the_republic_sequences.txt"
    DATA_PATH = Path.cwd() / 'data'

    INPUT_DATASET_PATH = DATA_PATH / input_document
    SEQUENCE_OUTPUT_DATASET_PATH = DATA_PATH / output_document

    preprocess(INPUT_DATASET_PATH, SEQUENCE_OUTPUT_DATASET_PATH)
    doc = load_file(SEQUENCE_OUTPUT_DATASET_PATH)

    # to train a model
    if args.training == "yes":
        train(doc, epochs=1)

    lines = doc.split('\n')
    seed_text = lines[randint(0, len(lines))]
    seq_length = len(lines[0].split()) - 1

    # Generated models are stored in the models directory, which has 43% accuracy
    # If the training pipeline is run it will store the models in root director of the project. To use those models we need specify the path of the models to the root directory of project as:
    # MODELS_PATH = Path.cwd() if re-run of training is desired
    MODELS_PATH = Path.cwd() / 'models'
    result = predict(seed_text, seq_length, model_path=MODELS_PATH)
    print("Text: \n", result.get('text'))
    print("Generated: \n", result.get('generated'))
