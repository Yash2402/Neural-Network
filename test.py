import csv

import numpy as np

from train import X_dev, Y_dev, nn, setup

with open("nndata.csv") as file:
    data = list(csv.reader(file))
    W1 = []
    print(data[1])
    for i in range(len(data) - setup[1] + len(setup[1:])):
        temp = []
        for j in range(len(data[i])):
            temp.append(float(data[i][j]))
        W1.append(temp)

    W2 = []
    for i in range(setup[1], sum(setup[1:])):
        temp = []
        for j in range(len(data[i])):
            temp.append(float(data[i][j]))
        W2.append(temp)

    b1 = []
    for i in range(len(data[-2])):
        b1.append(float(data[-2][i]))

    b2 = []
    for i in range(len(data[-1])):
        b2.append(float(data[-1][i]))

W1 = np.array(W1)
W2 = np.array(W2)
b1 = np.array(b1)
b2 = np.array(b2)
nn.W1, nn.W2, nn.b1, nn.b2 = W1, W2, b1, b2
