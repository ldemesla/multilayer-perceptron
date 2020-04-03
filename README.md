# multilayer-perceptron
From scratch multilayer-perceptron implementation for cancer detection, without using pytorch, tensorflow or keras 

## dataset
The dataset was elaborated by the University of Wisconsin in 1995, it give us for a large number of patients their breast cancer diagnosis and a list of 30 features describing the characteristics of a cell nucleus of breast
mass extracted with fine-needle aspiration

## usage
The first thing to do is `pip3 install .` to install all the dependencies

`python3 srcs/train.py [dataset]` is going to train the neural network for 3000 epochs with the given dataset, it is going to write a `network.py` file to save the final state of the network and create and/or append to the `validation_loss` file the final loss metrics generated with the valdiation side of the dataset.

`python3 srcs/predict.py [dataset] [network]` is going to make prediction about the breast cancer diagnosis of the different patients from the dataset according to given network file.

## images
