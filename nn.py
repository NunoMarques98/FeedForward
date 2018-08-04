import numpy as np

from sigmoid import *

class NeuralNetwork():
    
    def __init__(self, x, y):
        
        self.input = x
        self.weights1 = np.random.rand(self.input.shape[1], 4)
        self.weights2 = np.random.rand(4,1)
        self.y = y
        self.output = np.zeros(self.y.shape)

    def feedForward(self):
        
        self.layer1 = sigmoid(np.dot(self.input, self.weights1))
        self.output = sigmoid(np.dot(self.layer1, self.weights2))

    def backProp(self):

        weights2Err = 2*(self.y - self.output) * sigmoidPrime(self.output)
        weights1Err = np.dot(2*(self.y - self.output) * sigmoidPrime(self.output), self.weights2.T) * sigmoidPrime(self.layer1)   

        deltaWeights2 = np.dot(self.layer1.T, weights2Err)
        deltaWeights1 = np.dot(self.input.T, weights1Err)

        self.weights1 += deltaWeights1
        self.weights2 += deltaWeights2
