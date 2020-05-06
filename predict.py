"""
    This file is used to predict the name of Pokemon given images.

    usage: predict.py [-h] [-i SIZE_OF_IMAGE] [-j PATH_JSON] [-k TOP_K]
                  path_model path_image
"""

import argparse
import os
import json
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras import models
from sklearn.model_selection import train_test_split

import preprocess
import CNNmodel
from config import SIZE_OF_IMAGE, PATH_JSON

def get_parser():
    # Parse Arguments
    parser = argparse.ArgumentParser(
        description="A program that predicts name of Pokemon given images.")
    parser.add_argument('path_model',
                        help="Path to the trained model file")
    parser.add_argument('path_image',
                        help="Path to the input image(s)")
    parser.add_argument('-i', '--size_of_image', type=int,
                        help="Size of image after resizing")
    parser.add_argument('-j', '--path_json',
                        help="Path to the .json file that maps integer labels to Pokemon names")
    parser.add_argument('-k', '--top_k', type=int,
                        help="top_k accuracy to be shown when evaluateing model")
    return parser

def main(arg_list=None):
    # Initialize
    parser = get_parser()
    if arg_list:  # Called from another script
        args = parser.parse_args(arg_list)
    else:  # Called from command line
        args = parser.parse_args()
    
    path_image = args.path_image
    path_model = args.path_model
    size_of_image = args.size_of_image if args.size_of_image else SIZE_OF_IMAGE
    path_json = args.path_json if args.path_json else PATH_JSON
    top_k = args.top_k if args.top_k else 1

    # Load and preprocess images
    X, file_names = preprocess.load_images(path_image, size_of_image=size_of_image)
    print(X)
    X = preprocess.preprocess(X, size_of_image=size_of_image)

    # Load trained model
    model = models.load_model(path_model)
    # model.summary()

    # Load label_name from .json
    with open(path_json, 'r') as f:
        label_name = json.load(f)
    
    # Make predictions
    predictions_int = model.predict(X)
    print(predictions_int)
    prediction_name = [] # Stores top-k predicted pokemon names of the input images
    for prediction in predictions_int:
        # Find indices with top-k highest values (numpy array: [highest, ..., lowest])
        top_k_index = np.argsort(prediction)[::-1][:top_k]
        prediction_name.append([label_name[str(index.item())]
                                for index in top_k_index])

    for file, prediction in zip(file_names, prediction_name):
        print("Below are the top {} likely Pokemon in {}".format(top_k, file))
        print(prediction)
    
############################################################################################
if __name__ == "__main__":
    main()
