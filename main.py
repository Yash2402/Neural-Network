import sys

import numpy as np
import pandas as pd

from neural_network import NeuralNetwork

iteration, learning_rate = int(sys.argv[1]), float(sys.argv[2])

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
nn.gradientDescent(iteration, learning_rate, m)
