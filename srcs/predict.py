import sys
import numpy as np
import pandas as pd
from train import forward_prop
from train import weighted_sum
from train import sigmoid
from train import softmax
from math import log
from train import log_loss

def data_preproccessing(df):
    for column in df:
        maxm = df[column].max()
        minm = df[column].min()
        for elem in df[column].iteritems():
            if (column > 1):
                df.at[elem[0], column] = (df.at[elem[0], column] - minm) / (maxm - minm)
    df = df.drop(0, axis=1)
    df = df.to_numpy()
    x = np.zeros((df.shape[0], df.shape[1] - 1))
    y = np.zeros((df.shape[0], 2))
    output = {"M": 0, "B": 1}
    j = 0
    for i in range(len(df)):
        x[i] = df[i][1:]
        y[i] = [0, 0]
        y[i][output[df[i][0]]] = 1
    return (x, y)

def predict(dataset, network_file):
    network = np.load(network_file)
    df = pd.read_csv(dataset, header=None)
    x, y = data_preproccessing(df)
    loss = 0
    n = 0
    output = {"M": 0, "B": 1}
    for i in range(x.shape[0]):
        res = forward_prop(network, x[i])
        loss += log_loss(y[i], res)
        n += 1
        if (res[0] > res[1]):
            prediction = "M"
        else:
            prediction = "B"
        if (y[i][0] > y[i][1]):
            target = "M"
        else:
            target = "B"
        print("prediction = {}, target = {}".format(prediction, target))
        
    total_loss = loss / n
    print("log_loss = {}".format(total_loss))

if __name__ == "__main__":
    if (len(sys.argv) != 3):
        print("error: You need to provide one dataset and one network file")
    else:
        predict(sys.argv[1], sys.argv[2])
