import numpy as np


class NeuralNetwork:
    def __init__(
        self,
        setup,
        TestingData,
        TestingLables,
        TrainingData,
        TrainingLables,
    ):
        self.TrainingData = TrainingData
        self.TestingData = TestingData
        self.TrainingLables = TrainingLables
        self.TestingLables = TestingLables
        self.W1 = np.random.randn(setup[1], setup[0])
        self.W2 = np.random.randn(setup[1], setup[2])
        self.b1 = np.random.randn(setup[1], 1)
        self.b2 = np.random.randn(setup[2], 1)

    @staticmethod
    def softmax(Z):
        A = np.exp(Z) / np.sum(np.exp(Z))
        return A

    @staticmethod
    def ReLU(Z):
        return np.maximum(0, Z)

    @staticmethod
    def one_hot(Y):
        one_hot_Y = np.zeros((Y.size, Y.max() + 1))
        one_hot_Y[np.arange(Y.size), Y] = 1
        one_hot_Y = one_hot_Y.T
        return one_hot_Y

    @staticmethod
    def derivativeReLU(Z):
        return Z > 0

    @staticmethod
    def getPredictions(A2):
        return np.argmax(A2, 0)

    @staticmethod
    def getAccuracy(predictions, Y):
        print(predictions, Y)
        return np.sum(predictions == Y) / Y.size

    def forwardPropagation(self):
        self.Z1 = self.W1.dot(self.TrainingData) + self.b1
        self.A1 = NeuralNetwork.ReLU(self.Z1)
        self.Z2 = self.W2.dot(self.A1) + self.b2
        self.A2 = NeuralNetwork.softmax(self.Z2)

    def backPropagation(self, m):
        one_hot_Y = NeuralNetwork.one_hot(self.TrainingLables)
        self.dZ2 = self.A2 - one_hot_Y
        self.dW2 = (1 / m) * (self.dZ2.dot(self.A1.T))
        self.db2 = (1 / m) * (np.sum(self.dZ2))
        self.dZ1 = self.W2.T.dot(self.dZ2) * NeuralNetwork.derivativeReLU(self.Z1)
        self.dW1 = (1 / m) * (self.dZ1.dot(self.TrainingData.T))
        self.db1 = (1 / m) * (np.sum(self.dZ1))

    def updateParameters(self, alpha):
        self.W1 = self.W1 - alpha * self.dW1
        self.b1 = self.b1 - alpha * self.db1
        self.W2 = self.W2 - alpha * self.dW2
        self.b2 = self.b2 - alpha * self.db2

    def gradientDescent(self, iteration, alpha, m):
        for i in range(iteration):
            self.forwardPropagation()
            self.backPropagation(m)
            self.updateParameters(alpha)
            if i % 10 == 0:
                print(f"Iteration: {i}")
                predictions = NeuralNetwork.getPredictions(self.A2)
                print(
                    f"Accuracy: {NeuralNetwork.getAccuracy(predictions, self.TrainingLables)}"
                )
