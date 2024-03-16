import csv
import sys

import numpy as np
import pandas as pd

from neural_network import NeuralNetwork

data = np.array(pd.read_csv("data/mnist_train.csv"))
np.random.shuffle(data)
m, n = data.shape

data_dev = data[0:1000].T
Y_dev = data_dev[0]
X_dev = data_dev[1:n]
X_dev = X_dev / 255.0
data_train = data[1000:m].T
Y_train = data_train[0]
X_train = data_train[1:n]
X_train = X_train / 255.0

setup = [784, 10, 10]
nn = NeuralNetwork(setup, X_dev, Y_dev, X_train, Y_train)

if __name__ == "__main__":
    iteration, learning_rate = int(sys.argv[1]), float(sys.argv[2])
    nn.gradientDescent(iteration, learning_rate, m)
    with open("nndata.csv", "w") as nndata:
        csv_writer = csv.writer(nndata)

        for i in range(nn.W1.shape[0]):
            temp = []
            for j in range(nn.W1.shape[1]):
                temp.append(nn.W1[i][j])
            csv_writer.writerow(temp)

        for i in range(nn.W2.shape[0]):
            temp = []
            for j in range(nn.W2.shape[1]):
                temp.append(nn.W2[i][j])
            csv_writer.writerow(temp)

        _b1 = []
        for i in range(nn.b1.shape[0]):
            _b1.append(nn.b1[i][0])
        csv_writer.writerow(_b1)

        _b2 = []
        for i in range(nn.b2.shape[0]):
            _b2.append(nn.b2[i][0])
        csv_writer.writerow(_b2)
