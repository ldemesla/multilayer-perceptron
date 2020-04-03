import csv
import pandas as pd 
import numpy as np 
import sys
import time
from random import seed
from random import random
from math import exp
from math import log
import matplotlib.pyplot as plt

def sigmoid(linear):
    return 1.0 / (1.0 + exp(-linear))

def softmax(linear):
    return np.exp(linear) / float(sum(np.exp(linear)))

def weighted_sum(weights, inputs):
	linear = weights[-1]
	for i in range(len(weights)-1):
		linear += weights[i] * inputs[i]
	return linear

def forward_prop(network, x):
    inputs = x
    activation = 0
    for i, layer in enumerate(network):
        if (i == len(layer)):
            activation = 1
        new_inputs = []
        for neuron in layer:
            w_sum = weighted_sum(neuron["weights"], inputs)
            if (activation == 0):
                neuron["output"] = sigmoid(w_sum)
            else:
                neuron["output"] = w_sum
            new_inputs.append(neuron["output"])
        inputs = new_inputs
    inputs = softmax(inputs)
    return inputs

def init_network(layers_dim):
    network = list()
    for i in range(1, len(layers_dim)):
        layer = [{'weights':[random() for j in range(layers_dim[i - 1] + 1)]} for j in range(layers_dim[i])]
        network.append(layer)
    return network

def log_loss(Y_truth, Y_pred):
    loss = 0
    for i in range(len(Y_pred)):
       loss += Y_truth[i] * log(Y_pred[i]) + (1 - Y_truth[i]) * log(1 - Y_pred[i])
    return -(loss/2)

def data_preproccessing(df):
    for column in df:
        maxm = df[column].max()
        minm = df[column].min()
        for elem in df[column].iteritems():
            if (column > 1):
                df.at[elem[0], column] = (df.at[elem[0], column] - minm) / (maxm - minm)
    df = df.drop(0, axis=1)
    df = df.to_numpy()
    train_size = int((df.shape[0] * 0.8))
    valid_size = df.shape[0] - train_size
    x_train = np.zeros((train_size, df.shape[1] - 1))
    y_train = np.zeros((train_size, 2))
    x_valid = np.zeros((valid_size, df.shape[1] - 1))
    y_valid = np.zeros((valid_size, 2))
    output = {"M": 0, "B": 1}
    j = 0
    for i in range(len(df)):
        if (i < train_size):
            x_train[i] = df[i][1:]
            y_train[i] = [0, 0]
            y_train[i][output[df[i][0]]] = 1
        else:
            x_valid[j] = df[i][1:]
            y_valid[j] = [0, 0]
            y_valid[j][ output[df[i][0]]]  = 1
            j += 1
    return (x_train, y_train, x_valid, y_valid)

def backward_prop(network, expected, res):
    for i in reversed(range(len(network))):
        layer = network[i]
        errors = list()
        if i != len(network)-1:
            for j in range(len(layer)):
                error = 0.0
                for neuron in network[i + 1]:
                    error += (neuron['weights'][j] * neuron['delta'])
                errors.append(error)
        else:
            for j in range(len(layer)):
                neuron = layer[j]
                neuron["output"] = res[j]
                errors.append(expected[j] - neuron['output'])
        for j in range(len(layer)):
            neuron = layer[j]
            neuron['delta'] = errors[j] * (neuron['output'] * (1 - neuron["output"]))

def save_network(network, v_loss):
    for layer in network:
        for neuron in layer:
            del neuron["output"]
            del neuron["delta"]
    network = np.array(network)
    np.save("network", network)
    with open("validation_loss", "a") as f:
        f.write(str(v_loss))
        f.write("\n")

def validation(network, x_valid, y_valid):
    n = 0
    loss = 0
    for j in range(y_valid.shape[0]):
        res = forward_prop(network, x_valid[j])
        loss += log_loss(y_valid[j], res)
        n += 1
    total_loss = loss / n
    return total_loss

def update_weights(network, x_train, lr):
    for i in range(len(network)):
        inputs = x_train[:-1]
        if i != 0:
            inputs = [neuron['output'] for neuron in network[i - 1]]
        for neuron in network[i]:
            for j in range(len(inputs)):
                neuron['weights'][j] += lr * neuron['delta'] * inputs[j]
            neuron['weights'][-1] += lr * neuron['delta']

def train(dataset_csv, lr, epochs):
    seed(42)
    df = pd.read_csv(dataset_csv, header=None)
    x_train, y_train, x_valid, y_valid = data_preproccessing(df)
    network = init_network([x_train.shape[1], 2, 2, 2])
    train_loss = []
    validation_loss = []
    for i in range():
        loss = 0
        n = 0
        for j in range(x_train.shape[0]):
            res = forward_prop(network, x_train[j])
            loss += log_loss(y_train[j], res)
            backward_prop(network, y_train[j], res)
            update_weights(network, x_train[j], lr)
            n += 1
        t_loss = loss / n
        v_loss = validation(network, x_valid, y_valid)
        train_loss.append(t_loss)
        validation_loss.append(v_loss)
        print("epochs: {}/{} - train loss: {} - validation loss: {}".format(i, epochs, t_loss, v_loss))
    save_network(network, v_loss)
    l1 = plt.plot(train_loss, label="train loss")
    l2 = plt.plot(validation_loss, label="validation loss")
    plt.legend()
    plt.show()

if __name__ == "__main__":
    if (len(sys.argv) != 2):
        print("error: You need to provide one, and only one dataset parameter")
    else:
        train(sys.argv[1], 0.01, 30)
