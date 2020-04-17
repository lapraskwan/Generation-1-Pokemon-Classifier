"""
    This is the program that trains, saves and evaluates a CNN model for classifying Generation 1 Pokemon.

    It can be called in command line (see usage below), 
    or from another script (e.g. train.main(['-e', '5', '-i', '96'])).
    
    usage: train.py [-h] [-e EPOCHS] [-i SIZE_OF_IMAGE] [-j PATH_JSON] [-k TOP_K]
                [-m MODEL_NAME] [-p PATH_MODEL] [-t TEST_SIZE] [-T TITLE]
"""

import tensorflow as tf
from sklearn.model_selection import train_test_split
import sys
import argparse

import preprocess
import CNNmodel
from config import PATH_DATASET, SIZE_OF_IMAGE, GRAYSCALE, TEST_SIZE, RANDOM_SEED, EPOCHS, PATH_CHECKPOINT, MODEL_NAME, PATH_JSON, PATH_MODEL

def get_parser():
    # Parse Arguments
    parser = argparse.ArgumentParser(
        description="A program that trains a CNN model.")
    parser.add_argument('-e', '--epochs', type=int,
                        help="Number of epochs for model training")
    parser.add_argument('-i', '--size_of_image', type=int,
                        help="Size of image after resizing")
    parser.add_argument('-j', '--path_json',
                        help="The path to save the dictionary of integer labels mapping to Pokemon names")
    parser.add_argument('-k', '--top_k', type=int,
                        help="top_k accuracy to be shown when evaluateing model")
    parser.add_argument('-m', '--model_name',
                        help="The model to be trained")
    parser.add_argument('-p', '--path_model',
                        help="The path to save the trained model")
    parser.add_argument('-t', '--test_size', type=float,
                        help="Portion of dataset/Exact number of sample to be included into the test set")
    parser.add_argument('-T', '--title',
                        help="Title for the epoch-accuracy graph")
    return parser


def main(arg_list=None):
    """ Initialize """
    parser = get_parser()
    if arg_list: # Called from another script
        args = parser.parse_args(arg_list)
    else: # Called from command line
        args = parser.parse_args()

    # Set variables (See get_parser() for meanings of variables)
    epochs = args.epochs if args.epochs else EPOCHS
    size_of_image = args.size_of_image if args.size_of_image else SIZE_OF_IMAGE
    path_json = args.path_json if args.path_json else PATH_JSON
    top_k = args.top_k if args.top_k else None
    model_name = args.model_name if args.model_name else MODEL_NAME
    path_model = args.path_model if args.path_model else PATH_MODEL
    test_size = args.test_size if args.test_size else TEST_SIZE
    title = args.title if args.title else None

    """ Load Images and Data Preprocessing """
    # Load image from dataset
    X, y, y_map, num_classes = preprocess.load_from_dataset(
        PATH_DATASET, path_json, size_of_image)
    # Preprocess X
    X = preprocess.preprocess(X, size_of_image=size_of_image,
                              grayscale=GRAYSCALE)
    # Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=RANDOM_SEED, shuffle=True, stratify=y)

    """ Data Augmentation """
    data_gen = tf.keras.preprocessing.image.ImageDataGenerator(
        rotation_range=45,
        width_shift_range=.15,
        height_shift_range=.15,
        horizontal_flip=True,
        zoom_range=0.5
    )
    # data_gen.fit(X_train)

    """ Train CNN Model """
    model, history = CNNmodel.train_model(X_train, y_train, num_classes, path_model, model_name=model_name, epochs=epochs,
                                          validation_data=(
                                              X_test, y_test), size_of_image=size_of_image, data_gen=data_gen,
                                          path_checkpoint=PATH_CHECKPOINT)

    """ Evaluate Model """
    CNNmodel.evaluate_model(model, X_test,
                            y_test, y_map, history=history, top_k=top_k, title=title)

#########################################################################################
if __name__ == "__main__":
    main()
