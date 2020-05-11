# Generation-1-Pokemon-Classifier
This is a simple project to classify Generation 1 Pokemon given an image, using a CNN model.

## Demo
You can input an image (HEIC files are not supported), and the model will output the top-5 possible Generation One Pokemon 
in the image.

<img width="300" alt="Screenshot 2020-05-09 at 3 35 03 PM" src="https://user-images.githubusercontent.com/50293176/81577694-58324880-93dc-11ea-8441-095c0aa801cf.png">
<img width="300" alt="Screenshot 2020-05-11 at 11 16 25 PM" src="https://user-images.githubusercontent.com/50293176/81578517-7c425980-93dd-11ea-96e1-65dc78dbf38e.png">

The above website can be accessed through the following link: https://pokemon-gen-1-classification.herokuapp.com
It may take several seconds to load the model. Please refresh the page if it does not finish loading after a long time.

## Model Training and Evaluation

If you are interested in the codes for model training and evaluation, please follow the guide below.

## Prerequisites

Please download the following libraries if you haven't already.

1. NumPy
2. Tensorflow 2.0
3. OpenCV
4. Scikit-learn
5. Seaborn
6. Matplotlib

## Dataset
You can download the dataset I used in the link below. You only need this if you are going to train
a new model, because a trained model is provided in the folder "saved_models" already.

https://drive.google.com/drive/folders/10UtpoOe2_xoE3V-CDdxzZB6mCZ_92Rky?usp=sharing

Put the dataset into the same folder as train.py, or you can edit the config.py if you want to place the dataset somewhere else.

## Usage

There are three python files that can be executed in the command line, they are train.py, evaluate.py, and predict.py.

Below shows how to run the files. 
Details explaination of the flags can be found in the comments in the code, or the help page when you run the files with -h.

### train.py

This is the program that trains, saves and evaluates a CNN model for classifying Generation 1 Pokemon.
A label_name.json file will be created or updated, it is used to map the interger labels to Pokemon names for the model.
A trained model will be saved to the path given, and it will automatically evaluates the model after training.

It can be called in command line (see usage below), or from another script (e.g. train.main(['-e', '5', '-i', '96'])).
    
usage: 
train.py [-h] [-e EPOCHS] [-i SIZE_OF_IMAGE] [-j PATH_JSON] [-k TOP_K] 
[-m MODEL_NAME] [-p PATH_MODEL] [-t TEST_SIZE] [-T TITLE]


```
# Example of training a custom CNN model for 20 epochs, and save the model in a file named 'model.h5'.

python train.py -e 20 -p model.h5
```

### evaluate.py

This file is for evaluating existing models. 

usage: 
evaluate.py [-h] [-k TOP_K] path_model

```
# Example of evaluating the model we just trained, and showing top-5 accuracy

python evaluate.py model.h5 -k 5
```

### predict.py

This file is used to predict the name of Pokemon given images.

usage: 
predict.py [-h] [-i SIZE_OF_IMAGE] [-j PATH_JSON] [-k TOP_K] path_model path_image

```
# Example of predicting lapras.jpg (provided in the project folder for testing) using the model we just trained

python predict.py model.h5 lapras.jpg
```               

```
# Example of predicting a folder of images (e.g. dataset3/Lapras), and show top-3 predictions

python predict.py model.h5 dataset3/Lapras -k 3
```

### Changing Global Constants

There are a few global constants stored in the file config.py. Feel free to change the values if you want to.

---

Feel free to make any suggestions and improvements! Have fun with Pokemon!
