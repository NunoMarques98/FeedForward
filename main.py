from nn import *

X = np.array([[0,0,1], [0,1,1], [1,0,1], [1,1,1]])
y = np.array([[0],[1],[1],[0]])

nn = NeuralNetwork(X,y)

nn.train(1500)

print(nn.output)