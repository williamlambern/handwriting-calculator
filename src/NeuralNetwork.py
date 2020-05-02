import numpy as np


class NeuralNetwork:

    def __init__(self, inputSize, hiddenSize, outputSize):

        self.weights1 = np.loadtxt('weights1.csv', delimiter=',')
        self.weights2 = np.loadtxt('weights2.csv', delimiter=',')
        
    def sigmoid(self, x):  
        return 1/(1+np.exp(-x))

    def sigmoidPrime(self, x):
        return self.sigmoid(x) * (1 - self.sigmoid(x))
    
    def feedforward(self, inputs):
        self.hidden = self.sigmoid(np.dot(inputs, self.weights1))
        self.output = self.sigmoid(np.dot(self.hidden, self.weights2))
        return self.output

    def backward(self, inputs, actual, output):
        output_error = actual - output
        output_delta = output_error * self.sigmoidPrime(actual)
        
        hidden_error = output_delta.dot(self.weights2.T)
        hidden_delta = hidden_error*self.sigmoidPrime(self.hidden)
        
        self.weights1 = self.weights1 + inputs.T.dot(hidden_delta)
        self.weights2 = self.weights2 + self.hidden.T.dot(output_delta)

    def train(self):
        with open('data.txt', 'r') as f:                               #import training data
            train_inputs = f.read().split('\n')
        for i in range(len(train_inputs)):
            train_inputs[i] = train_inputs[i].split(',')
            for j in range(len(train_inputs[i])):
                train_inputs[i][j] = float(train_inputs[i][j])

        train_inputs = np.array(train_inputs)
        train_outputs = np.array([[1,0,0,0,0,0,0,0,0,0,0,0],[1,0,0,0,0,0,0,0,0,0,0,0],[1,0,0,0,0,0,0,0,0,0,0,0],
                                  [0,1,0,0,0,0,0,0,0,0,0,0],[0,1,0,0,0,0,0,0,0,0,0,0],[0,1,0,0,0,0,0,0,0,0,0,0],
                                  [0,0,1,0,0,0,0,0,0,0,0,0],[0,0,1,0,0,0,0,0,0,0,0,0],[0,0,1,0,0,0,0,0,0,0,0,0],
                                  [0,0,0,1,0,0,0,0,0,0,0,0],[0,0,0,1,0,0,0,0,0,0,0,0],[0,0,0,1,0,0,0,0,0,0,0,0],
                                  [0,0,0,0,1,0,0,0,0,0,0,0],[0,0,0,0,1,0,0,0,0,0,0,0],[0,0,0,0,1,0,0,0,0,0,0,0],
                                  [0,0,0,0,0,1,0,0,0,0,0,0],[0,0,0,0,0,1,0,0,0,0,0,0],[0,0,0,0,0,1,0,0,0,0,0,0],
                                  [0,0,0,0,0,0,1,0,0,0,0,0],[0,0,0,0,0,0,1,0,0,0,0,0],[0,0,0,0,0,0,1,0,0,0,0,0],
                                  [0,0,0,0,0,0,0,1,0,0,0,0],[0,0,0,0,0,0,0,1,0,0,0,0],[0,0,0,0,0,0,0,1,0,0,0,0],
                                  [0,0,0,0,0,0,0,0,1,0,0,0],[0,0,0,0,0,0,0,0,1,0,0,0],[0,0,0,0,0,0,0,0,1,0,0,0],
                                  [0,0,0,0,0,0,0,0,0,1,0,0],[0,0,0,0,0,0,0,0,0,1,0,0],[0,0,0,0,0,0,0,0,0,1,0,0],
                                  [0,0,0,0,0,0,0,0,0,0,1,0],[0,0,0,0,0,0,0,0,0,0,1,0],[0,0,0,0,0,0,0,0,0,0,1,0],
                                  [0,0,0,0,0,0,0,0,0,0,0,1],[0,0,0,0,0,0,0,0,0,0,0,1],[0,0,0,0,0,0,0,0,0,0,0,1]]) # 36x1 matrix (numbers 0-9 and +,-)
        
        for i in range(100000):
            self.trainFunction(train_inputs, train_outputs) #make it smart :)
            if i % 1000 == 0:
                print(i/1000, '% done')

        
        
        np.savetxt('weights1.csv', self.weights1, delimiter=',')
        np.savetxt('weights2.csv', self.weights2, delimiter=',')

    def trainFunction(self, X, y):
        o = self.feedforward(X)
        self.backward(X, y, o)
        
