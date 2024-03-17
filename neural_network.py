import numpy as np
from matplotlib import pyplot as plt


class NeuralNetwork:
    def __init__(
        self,
        setup,
        TestingData,
        TestingLabels,
        TrainingData,
        TrainingLabels,
    ):
        self.TrainingData = TrainingData
        self.TestingData = TestingData
        self.TrainingLabels = TrainingLabels
        self.TestingLabels = TestingLabels
        self.W1 = np.random.rand(setup[1], setup[0]) - 0.5
        self.W2 = np.random.rand(setup[1], setup[2]) - 0.5
        self.b1 = np.random.rand(setup[1], 1) - 0.5
        self.b2 = np.random.rand(setup[2], 1) - 0.5

    @staticmethod
    def softmax(Z):
        A = np.exp(Z) / sum(np.exp(Z))
        return A

    @staticmethod
    def ReLU(Z):
        return np.maximum(Z, 0)

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

    def forwardPropagation(self, X):
        self.Z1 = self.W1.dot(X) + self.b1
        self.A1 = NeuralNetwork.ReLU(self.Z1)
        self.Z2 = self.W2.dot(self.A1) + self.b2
        self.A2 = NeuralNetwork.softmax(self.Z2)

    def backPropagation(self, m):
        one_hot_Y = NeuralNetwork.one_hot(self.TrainingLabels)
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
            self.forwardPropagation(self.TrainingData)
            self.backPropagation(m)
            self.updateParameters(alpha)
            if i % 10 == 0:
                print(f"Iteration: {i}")
                predictions = NeuralNetwork.getPredictions(self.A2)
                print(
                    f"Accuracy: {NeuralNetwork.getAccuracy(predictions, self.TrainingLabels)}"
                )

    def makePrediction(self, X):
        self.forwardPropagation(X)
        prediction = NeuralNetwork.getPredictions(self.A2)
        return prediction

    def testPrediction(self, index):
        current_image = self.TestingData[:, index, None]
        prediction = self.makePrediction(current_image)
        label = self.TestingLabels[index]
        print("Prediction: ", prediction)
        print("Label: ", label)
        current_image = current_image.reshape((28, 28)) * 255
        plt.gray()
        plt.imshow(current_image, interpolation="nearest")
        plt.show()
