""" 
    This file includes all the constants to be used in the modules.
"""

GRAYSCALE = False  # Use grayscale if true, RGB if false
CHANNEL = 3  # 3 channels for RGB image
SIZE_OF_IMAGE = 128  # Size of image after resizing
TEST_SIZE = 0.2  # Portion of image that becomes the test set
RANDOM_SEED = 807  # Random seed for shuffling
EPOCHS = 10  # Number of epochs while training the model
MODEL_NAME = 'custom'  # The model to be trained
TOP_K = 5  # TOP_K accuracy to be printed out when evaluating model
PATH_DATASET = './dataset3'  # Path of the dataset
PATH_CHECKPOINT = './checkpoints/checkpoint.h5' # Path to save checkpoint while training the model
PATH_JSON = './label_name.json' # Path to save the label_name json string
PATH_MODEL = './saved_models/default.h5' # Path to save the trained model
