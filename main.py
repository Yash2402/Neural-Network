import csv

import numpy as np
import pandas as pd
import pygame

from neural_network import NeuralNetwork

data = pd.read_csv("data/mnist_train.csv")
data = np.array(data)
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

k = 1
tile_size = 20
screen = pygame.display.set_mode((28 * tile_size, 28 * tile_size), pygame.RESIZABLE)
run = True
while run:
    screen.fill((255, 255, 255))
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            run = False
        if pygame.mouse.get_pressed()[0]:
            k += 1
        if pygame.mouse.get_pressed()[2]:
            k -= 1
        if pygame.key.get_pressed() and event.type == pygame.K_SPACE:
            pass

    u = 0

    for i in range(28):
        for j in range(28):
            if k != 0:
                pygame.draw.rect(
                    screen,
                    (int(data[k][u]), int(data[k][u]), int(data[k][u])),
                    ((j * tile_size), (i * tile_size), tile_size, tile_size),
                )
                u += 1
            else:
                pygame.draw.rect(
                    screen, (0, 0, 0), (0, 0, 28 * tile_size, 28 * tile_size)
                )

    pygame.display.update()
