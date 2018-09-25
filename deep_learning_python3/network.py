import copy
import random
import numpy as np


class Network:

    def __init__(self, num_units):
        self.num_layers = len(num_units)
        self.num_units = num_units
        self.biases = [np.random.randn(x,1) for x in num_units[1:]]
        self.weights = [np.random.randn(y,x)
                        for x,y in zip(num_units[:-1], num_units[1:])]

    def feedforward(self, a):
        for w,b in zip(self.weights, self.biases):
            a = sigmoid(np.dot(w, a) + b)
        return a

    def SGD(self, training_data, num_epochs, batch_size, learning_rate, test_data=None):
        if test_data:
            num_test = len(test_data)
        num_training = len(training_data)
        print (f'num_training={num_training}')
        for ne in range(num_epochs):
            random.shuffle(training_data)
            for i in range(0, num_training, batch_size):
                self.update_batch(training_data[i:i+batch_size-1], learning_rate)
            if test_data:
                print(f'Epoch {ne}: {self.evaluate(test_data)} / {num_test}')
            else:
                print(f'Epoch {ne} complete')

    def update_batch(self, batch, learning_rate):
        batch_size = len(batch)
        grad_b = [np.zeros(b.shape) for b in self.biases]
        grad_w = [np.zeros(w.shape) for w in self.weights]
        for x, y in batch:
            gb, gw = self.backprop(x,y)
            grad_b = [xx + yy/batch_size for xx, yy in zip(grad_b, gb)]
            grad_w = [xx + yy/batch_size for xx, yy in zip(grad_w, gw)]
        for b, w, gb, gw in zip(self.biases, self.weights, grad_b, grad_w):
            b -= gb*learning_rate
            w -= gw*learning_rate

    def backprop(self, x, y):
        grad_w = [np.zeros(w.shape) for w in self.weights]
        grad_b = [np.zeros(b.shape) for b in self.biases]
        # forward sweep
        activations = [x]
        zs = []
        for w,b in zip(self.weights, self.biases):
            z = np.dot(w, activations[-1]) + b
            zs.append(z)
            activations.append(sigmoid(z))
        # backward sweep
        delta = self.cost_derivative(activations[-1],y) * sigmoid_prime(zs[-1])
        for l in range(0,self.num_layers-1)[::-1]:
            grad_b[l] = delta
            grad_w[l] = np.dot(delta,activations[l].transpose())
            if l==0: break
            delta = np.dot(self.weights[l].transpose(),delta)*sigmoid_prime(zs[l-1])
        return grad_b, grad_w

    def numerical_gradient(self, x, y, dparam = 1e-4):
        cost = self.cost(x,y)
        grad_w = []
        grad_b = []
        for w in self.weights:
            grad_w.append(np.zeros(w.shape))
            for i,j in np.ndindex(w.shape):
                w[i,j] += dparam
                dcost = self.cost(x,y)-cost
                w[i,j] -= dparam
                grad_w[-1][i,j] = dcost/dparam
        for b in self.biases:
            grad_b.append(np.zeros(b.shape))
            for i in range(len(b)):
                b[i] += dparam
                dcost = self.cost(x,y)-cost
                b[i] -= dparam
                grad_b[-1][i] = dcost/dparam
        return grad_b, grad_w

    def evaluate(self, test_data):
        test_results = [(np.argmax(self.feedforward(x)), y) for (x, y) in test_data]
        return sum(int(x == y) for (x, y) in test_results)

    def cost(self, x, y):
        output_activation = self.feedforward(x)
        return 0.5*np.sum((output_activation-y)**2)

    def cost_derivative(self, output_activation, y):
        return output_activation-y



def sigmoid(z):
    return 1.0/(1.0+np.exp(-z))


def sigmoid_prime(z):
    return sigmoid(z)*(1-sigmoid(z))


def gradient_check(num_units, relative_tolerance = 1e-4):
    x = np.ones([num_units[0],1])*0.5
    y = np.random.rand(num_units[-1],1)

    net = Network(num_units)
    numerical_grad_b, numerical_grad_w = net.numerical_gradient(x,y)
    backprop_grad_b, backprop_grad_w = net.backprop(x,y)

    for numerical, backprop in zip(numerical_grad_w,backprop_grad_w):
        diff = numerical-backprop
        if np.linalg.norm(diff) > relative_tolerance*np.linalg.norm(backprop):
            return False

    return True


if __name__ == "__main__":
    print("hello world")