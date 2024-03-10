import numpy as np
import pandas as pd

from neural_network import NeuralNetwork

data = np.array(pd.read_csv("data/mnist_train.csv"))
np.random.shuffle(data)
m, n = data.shape

data_dev = data[0:1000].T
Y_dev = data_dev[0]
X_dev = data_dev[1:n]
data_train = data[1000:m].T
Y_train = data_train[0]
X_train = data_train[1:n]

setup = [784, 10, 10]
nn = NeuralNetwork(setup, X_dev, Y_dev, X_train, Y_train)
nn.gradientDescent(500, 0.01, m)
