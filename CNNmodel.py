"""
    This file includes all functions for training the CNN model.
"""

import tensorflow as tf
import numpy as np
from tensorflow.keras import layers, models, applications
import sys
import matplotlib.pyplot as plt
import seaborn as sns

from config import CHANNEL, SIZE_OF_IMAGE, TEST_SIZE, RANDOM_SEED, EPOCHS, MODEL_NAME, TOP_K, PATH_CHECKPOINT

def get_model(num_classes, model_name=MODEL_NAME, size_of_image=SIZE_OF_IMAGE):
    """
    Returns a CNN model, or None if the parameter model is not recognised.

    Parameters:
        num_classes (int): Total number of different classes of Pokemon.
        model (string): One of the 3 models (custom, MobileNetV2, VGG19). Default: config.MODEL.
        size_of_image (int): Size of image after resizing. Default: config.SIZE_OF_IMAGE.
    
    Returns:
        model (tf.keras.Model): A CNN model.
    """
    if model_name == 'custom':
        """ Custom CNN Model """
        model = models.Sequential()
        model.add(layers.Conv2D(32, (3, 3), activation='relu',
                                input_shape=(size_of_image, size_of_image, CHANNEL)))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Conv2D(64, (3, 3), activation='relu'))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Conv2D(128, (3, 3), activation='relu'))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Conv2D(256, (3, 3), activation='relu'))
        model.add(layers.MaxPooling2D((2, 2)))

        model.add(layers.Flatten())
        model.add(layers.Dense(196, activation='relu'))
        # model.add(layers.Dropout(0.3))
        model.add(layers.Dense(num_classes))
    elif model_name == 'MobileNetV2':
        """ MobileNetV2 """
        base_model = applications.MobileNetV2(input_shape=(size_of_image, size_of_image, CHANNEL),
                                              include_top=False,
                                              weights='imagenet')
        base_model.trainable = False
        global_average_layer = layers.GlobalAveragePooling2D()
        output_layer = layers.Dense(num_classes)

        model = tf.keras.Sequential([
            base_model,
            global_average_layer,
            layers.Flatten(),
            layers.Dense(256, activation='relu'),
            layers.Dropout(0.3),
            output_layer
        ])
    elif model_name == 'VGG19':
        """ VGG19 """
        base_model = applications.VGG19(input_shape=(size_of_image, size_of_image, CHANNEL),
                                        include_top=False,
                                        weights='imagenet')
        base_model.trainable = False
        global_average_layer = layers.GlobalAveragePooling2D()
        output_layer = layers.Dense(num_classes)

        model = tf.keras.Sequential([
            base_model,
            global_average_layer,
            layers.Flatten(),
            layers.Dense(256, activation='relu'),
            layers.Dropout(0.3),
            output_layer
        ])
    else:
        print("ERROR: Cannot recognise model.")
        sys.exit(1)

    return model


def train_model(X_train, y_train, num_classes, path_model, model_name=MODEL_NAME, epochs=EPOCHS, 
                validation_data=None, size_of_image=SIZE_OF_IMAGE, data_gen=None, 
                path_checkpoint=PATH_CHECKPOINT):
    """
    Trains and saves a CNN model with the train set.

    Parameters:
        X_train (np array): A numpy array of the images in the train set.
        y_train (np array): A numpy array of the labels in the train set.
        num_classes (int): Total number of different classes of Pokemon.
        path_model (string): The path of the file for saving the model.
        model_name (string, optional): One of the 3 models (custom, MobileNetV2, VGG19). Default: config.MODEL.
        epochs (int, optional): Number of epochs. Default: config.EPOCHS.
        validation_data ((np array, np array), optional): A tuple of X_test and y_test. Default: None.
        size_of_image (int, optional): Size of image after resizing. Default: config.SIZE_OF_IMAGE.
        data_gen (tf.keras.preprocessing.image.ImageDataGenerator, optional): An image generator for data augmentation. Default: None.
        path_checkpoint (string, optional): The path of the file for saving checkpoints. Default: config.PATH_CHECKPOINT.

    Returns:
        model (tf.keras.Model): The CNN model after training.
        history (tf.keras.callbacks.History): The history of the training of the model.
    """

    # Get the CNN model
    model = get_model(num_classes, model_name=model_name, size_of_image=size_of_image)
    model.summary()

    # Save the epoch with the highest val_accuracy
    checkpoint = tf.keras.callbacks.ModelCheckpoint(path_checkpoint,
                                                    verbose=2, monitor='val_accuracy', save_best_only=True)

    model.compile(optimizer='adam',
                loss=tf.keras.losses.SparseCategoricalCrossentropy(
                    from_logits=True),
                metrics=['accuracy'])

    # Train model with/without data augmentation
    if data_gen is None:
        history = model.fit(X_train, y_train, epochs=EPOCHS,
                            validation_data=validation_data)
    else:
        history = model.fit(data_gen.flow(X_train, y_train), epochs=epochs,
                            validation_data=validation_data, callbacks=[checkpoint])

    # Load the weights of the epoch with the highest val_accuracy
    model.load_weights(path_checkpoint)
    # Save the model to reload it later
    model.save(path_model)

    return model, history


def top_k_acc(labels, predictions, top_k=TOP_K):
    """
    Returns the top_k accuracy of the model.

    Parameters:
        labels (np array): Truth labels.
        predictions (np array): Predicted labels.
        top_k (int, optional): top_k accuracy to be printed out when evaluating model. Default: None.
    
    Returns:
        top_k_acc (float): The top_k accuracy of the model.
    """
    in_top_k = tf.math.in_top_k(
        predictions=predictions, targets=labels, k=top_k) # Numpy array with 0s and 1s. [1, 0, 0, 1, 1, ...]
    # Get accuracy by calculating the mean of the array
    accuracy = tf.math.reduce_mean(tf.cast(in_top_k, tf.float32)) # Type: tf.Variable
    return tf.keras.backend.eval(accuracy)

def evaluate_model(model, X_test, y_test, y_map, history=None, top_k=None, title=None, ylim=[0, 1], print_acc_pokemon=False):
    """
    Evaluates the model.

    If history != None, plots the epoch-accuracy graph.
    Prints the loss, top-1 (and top_k) accuracy of the test set.
    Plot the the graph of accuracy by Pokemon.
    If print_acc_pokemon == True, prints out accuracy by pokemon.

    Parameters:
        model (tf.kears.Model): The model to be evaluated.
        X_test (np array): The numpy array storing the images in test set.
        y_test (np array): The numpy array storing the labels in test set.
        history (tf.keras.callbacks.History, optional): The history of the training of the model. Default: None.
        top_k (int, optional): top_k accuracy to be printed out when evaluating model. Default: None.
        title (string, optional): The title of the graph. Default: None.
        ylim ([low, high], optional): The boundaries of the graph's y-axis.
    """

    """Epoch-Accuracy Graph"""
    if history is not None:
        plt.plot(history.history['accuracy'], label='accuracy')
        plt.plot(history.history['val_accuracy'], label='val_accuracy')
        if title:
            plt.title(title)
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.ylim(ylim)
        plt.legend(loc='lower right')
        plt.show()

    """Loss, Top-1 Acc, Top-k Acc"""
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
    print("Loss: ", test_loss)
    print("Test Top-1 Accuracy: ", test_acc)
    predictions = model.predict(X_test)
    if top_k:
        print("Test Top-{} Accuracy: ".format(top_k), top_k_acc(y_test, predictions, top_k))
    
    """Accuracy by Pokemon Graph"""
    counts_by_pokemon = {} # {'0': [correct, total]}
    for index in range(len(X_test)):
        pokemon = y_map[y_test[index]]
        if pokemon in counts_by_pokemon:
            if np.argmax(predictions[index]) == y_test[index]:
                counts_by_pokemon[pokemon][0] += 1
                counts_by_pokemon[pokemon][1] += 1
            else:
                counts_by_pokemon[pokemon][1] += 1
        else:
            if np.argmax(predictions[index]) == y_test[index]:
                counts_by_pokemon[pokemon] = [1, 1]
            else:
                counts_by_pokemon[pokemon] = [0, 1]
    
    pokemon_list = []
    accuracy_by_pokemon = []
    for pokemon in counts_by_pokemon:
        acc = counts_by_pokemon[pokemon][0]/counts_by_pokemon[pokemon][1]
        pokemon_list.append(pokemon)
        accuracy_by_pokemon.append(acc)

    plt.figure(figsize=(20, 8))
    sns.lineplot(x=pokemon_list, y=accuracy_by_pokemon
                )
    plt.xticks(rotation=90)
    plt.margins(x=0)
    if title:
        plt.title(title)
    plt.ylim((0, 1))
    plt.show()

    """Accuracy By Pokemon"""
    if print_acc_pokemon:
        for index, pokemon in enumerate(pokemon_list):
            labels_pokemon = []
            pred_pokemon = []
            for idx, label in enumerate(y_test):
                if y_map[label] == pokemon:
                    labels_pokemon.append(label)
                    pred_pokemon.append(predictions[idx])    
            print(pokemon)
            print("Top-1 Accuracy: ", accuracy_by_pokemon[index])
            if top_k:
                print("Top-{} Accuracy: ".format(top_k),
                      top_k_acc(labels_pokemon, pred_pokemon, top_k))
