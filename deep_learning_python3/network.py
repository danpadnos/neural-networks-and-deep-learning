import numpy as np


class Network:

    def __init__(self, num_units):
        self.num_layers = len(num_units)
        self.num_units = num_units
        self.biases = [np.random.randn(x,1) for x in num_units]
        self.weights = [np.random.randn(y,x)
                        for x,y in zip(num_units[:-1], num_units[1:])]

    def feedforward(self, a):
        pass

    def SGD(self, training_data, num_epochs, mini_batch_size, learning_rate, test_data=None):
        pass

    def update_mini_batch(self, mini_batch, learning_rate):
        pass

    def backprop(self, x, y):
        pass

    def evaluate(self, test_data):
        pass

    def cost_derivative(self, output_activation, y):
        pass



def sigmoid(z):
    return 1.0/(1.0+np.exp(-z))


def sigmod_prime(z):
    return sigmoid(z)*(1-sigmoid(z))