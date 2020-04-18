"""
    This file includes the functions for preprocessing the dataset.
"""

import os
import numpy as np
import cv2 as cv
import tensorflow as tf
import json
from sklearn.model_selection import train_test_split
from config import GRAYSCALE, CHANNEL, SIZE_OF_IMAGE

def save_dict(dict, path):
    """
    Saves a dictionary into a .json file.

    Parameters:
        dict (dict): The dictionary to be saved.
        path (string): The path of the .json file.
    """
    label_name_json = json.dumps(dict)
    f = open(path, 'w')
    f.write(label_name_json)
    f.close()

def load_from_dataset(path_dataset, path_json, size_of_image=SIZE_OF_IMAGE):
    """
    Load images from the Pokemon Generation One dataset.

    Save the dictionary (label_name) that maps y_label to Pokemon names into a .json file.

    Parameters:
        path_dataset (string): The path of the dataset.
        path_json (string): The path of the .json file to save label_name.
        size_of_image (int, optional): Size of image after resizing. Default: config.SIZE_OF_IMAGE.
    
    Returns:
        X (np array): A numpy array that stores the loaded images.
        y (np array): A numpy array that stores the corresponding labels of the images.
        label_name (dict): A dictionary that maps y_labels to Pokemon names, i.e. {'0': 'Venusaur', '1': 'Lapras', ......}.
        num_classes (int): Total number of different classes of Pokemon.
    """
    # Path to dataset
    label_name = {}  # Mapping the integer labels to Pokemon names, i.e. {'0': 'Venusaur', '1': 'Lapras', ......}
    X = []  # Store images
    y = []  # Store integer labels representing the image
    for index, pokemon in enumerate(os.listdir(path_dataset)): # For every folder
        folder_path = os.path.join(path_dataset, pokemon)
        # Add a label,name pair into label_name
        label_name[index] = pokemon
        for image in os.listdir(folder_path): # For every image
            if image[0] != '.': # To avoid .DS_Store files
                image_path = os.path.join(folder_path, image)
                try:
                    # Read in image
                    image = cv.imread(image_path)
                    # Resize image so that all images are of the same size
                    image = cv.resize(image, (size_of_image, size_of_image))
                    
                    X.append(image)
                    y.append(index)
                except:
                    print("Failed to read and resize image: ", image_path)
                    print("Image: ", image)

    # Calculate the number of classes of Pokemon
    num_classes = len(label_name.keys())
    # Save label_name into a .json file
    save_dict(label_name, path_json)
    # Convert X into a numpy array
    X = np.array(X)
    # Convert y into a numpy array
    y = np.array(y)

    return X, y, label_name, num_classes

def load_images(path, size_of_image=SIZE_OF_IMAGE):
    """
    Load images from a folder or a single image file.

    Parameters:
        path (string): Path to the folder or the image file.
        size_of_image (int, optional): Size of image after resizing. Default: config.SIZE_OF_IMAGE.

    Returns:
        X (np array): A numpy array that stores the loaded images.
    """

    X = [] # Stores images
    file_names = [] # Stores file names
    if os.path.isfile(path):  # Single image
        try:
            # Read in image
            image = cv.imread(path)
            # Resize image so that all images are of the same size
            image = cv.resize(image, (size_of_image, size_of_image))

            X.append(image)
            file_names.append(path)
        except:
            print("Failed to read and resize image: ", path)

    else: # Folder containing images
        for path_image in os.listdir(path):
            if path_image[0] != '.':  # To avoid .DS_Store files
                full_path = os.path.join(path, path_image)
                try:
                    # Read in image
                    image = cv.imread(full_path)
                    # Resize image so that all images are of the same size
                    image = cv.resize(image, (size_of_image, size_of_image))

                    X.append(image)
                    file_names.append(path_image)
                except:
                    print("Failed to read and resize image: ", full_path)
    
    # Convert X into a numpy array
    X = np.array(X)

    return X, file_names

def preprocess(X, size_of_image=SIZE_OF_IMAGE, grayscale=GRAYSCALE):
    """
    Preprocess X for later use in training the model or testing.

    Parameters:
        X (np array): A numpy array storing the images.
        size_of_image (int, optional): Size of image after resizing. Default: config.SIZE_OF_IMAGE.
        grayscale (bool, optional): Make X into grayscale if True, RGB if False. Default: config.GRAYSCALE.
    
    Returns:
        X (np array): The preprocessed numpy array of X.
    """
    if grayscale:
        X = tf.image.rgb_to_grayscale(X)
        X = X.numpy()  # Shape: (-1, size_of_image, size_of_image, 1)
        # Since pre-trained models (MobileNetV2, VGG19) can only take input with shape (-1, size_of_image, size_of_image, 3),
        # grayscale array X need to repeat its value 3 times
        # Shape: (-1, size_of_image, size_of_image, channel)
        X = np.repeat(X, CHANNEL, axis=-1)
    
    # Reshape X
    X = X.reshape(-1, size_of_image, size_of_image, CHANNEL)
    # Scale down the rgb value
    X = X / 255

    return X
