# Multilayer-perceptron
From scratch multilayer-perceptron implementation for cancer detection, without using pytorch, tensorflow or keras 

## Dataset
The dataset was elaborated by the University of Wisconsin in 1995, it give us for a large number of patients their breast cancer diagnosis and a list of 30 features describing the characteristics of a cell nucleus of breast
mass extracted with fine-needle aspiration

## Usage
The first thing to do is `pip3 install .` to install all the dependencies

`python3 srcs/train.py [dataset]` is going to train the neural network for 3000 epochs with the given dataset, it is going to write a `network.py` file to save the final state of the network and create and/or append to the `validation_loss` file the final loss metrics generated with the valdiation side of the dataset. At the end a plot is generated and aim to describes the evolution of the training loss and the validation loss with the number of epochs.

`python3 srcs/predict.py [dataset] [network]` is going to make prediction about the breast cancer diagnosis of the different patients from the dataset according to given network file.
