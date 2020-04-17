""" 
    This file is for evaluating existing models. 

    usage: evaluate.py [-h] [-k TOP_K] path_model
"""

import argparse
from tensorflow.keras import models

import preprocess
import CNNmodel
from sklearn.model_selection import train_test_split


def get_parser():
    # Parse Arguments
    parser = argparse.ArgumentParser(
        description="A program that predicts name of Pokemon given images.")
    parser.add_argument('path_model',
                        help="Path to the trained model file")
    parser.add_argument('-k', '--top_k', type=int,
                        help="top_k accuracy to be shown when evaluateing model")
    parser.add_argument('-r', '--random_seed', type=int,
                        help="Random seed to be used to split train and test set")
    return parser


def main(arg_list=None):
    # Initialize
    parser = get_parser()
    if arg_list:  # Called from another script
        args = parser.parse_args(arg_list)
    else:  # Called from command line
        args = parser.parse_args()

    path_model = args.path_model
    top_k = args.top_k if args.top_k else None
    random_seed = args.random_seed if args.random_seed else None

    # Load trained model
    model = models.load_model(path_model)
    model.summary()

    # Load image from dataset
    X, y, label_name, num_classes = preprocess.load_from_dataset(
        './dataset3', './label_name.json', size_of_image=128)
    # Preprocess X
    X = preprocess.preprocess(X, size_of_image=128)
    # Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=random_seed, shuffle=True, stratify=y)

    # Evaluate Model
    CNNmodel.evaluate_model(model, X_test,
                            y_test, label_name, top_k=top_k, print_acc_pokemon=True)

########################################################################
if __name__ == "__main__":
    main()
