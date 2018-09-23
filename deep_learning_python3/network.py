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
        for ne in range(num_epochs):
            random.shuffle(training_data)
            for i in range(0, num_training, batch_size):
                self.update_batch(training_data[i:i+batch_size-1], learning_rate)
            if test_data:
                print(f'Epoch {ne}: {self.evaluate(test_data)} / {num_test}')
            else:
                print(f'Epoch {ne} complete')

    def update_batch(self, batch, learning_rate):
        pass

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
        delta = self.cost_derivative(activations[-1],y) * sigmod_prime(zs[-1])
        for l in range(0,self.num_layers-1)[::-1]:
            grad_b[l] = delta
            grad_w[l] = np.dot(delta,activations[l].transpose())
            if l==0: break
            delta = np.dot(self.weights[l].transpose(),delta)*sigmoid_prime(zs[l-1])
        return grad_b, grad_w

    def evaluate(self, test_data):
        pass

    def cost(self, output_activation, y):
        return 0.5*np.sum((output_activation-y)**2)

    def cost_derivative(self, output_activation, y):
        return output_activation-y



def sigmoid(z):
    return 1.0/(1.0+np.exp(-z))


def sigmoid_prime(z):
    return sigmoid(z)*(1-sigmoid(z))


def gradient_check(num_units, tiny_grad = 1e-6, epsilon = 1e-4, relative_tolerance = 1e-3):
    x = np.ones([num_units[0],1])*0.5
    y = np.random.rand(num_units[-1],1)

    net1 = Network(num_units)
    out1 = net1.feedforward(x)
    backprop_grad_b, backprop_grad_w = net1.backprop(x,y)

    net2 = copy.deepcopy(net1)

    for nl in range(net2.num_layers-2):
        for i in range(num_units[nl + 1]):
            # weights
            for j in range(num_units[nl]):
                net2.weights[nl][i,j] += epsilon
                out2 = net2.feedforward(x)
                grad = (net2.cost(out2, y) - net1.cost(out1, y)) / epsilon
                if np.abs(grad-backprop_grad_w[nl][i,j])/(np.abs(grad)+tiny_grad) > relative_tolerance:
                    return False
                else:
                    net2.weights[nl][i,j] -= epsilon
            # biases
            net2.biases[nl][i] += epsilon
            out2 = net2.feedforward(x)
            grad = (net2.cost(out2, y) - net1.cost(out1, y)) / epsilon
            if np.abs(grad-backprop_grad_b[nl][i])/(np.abs(grad)+tiny_grad) > relative_tolerance:
                return False
            else:
                net2.biases[nl][i] -= epsilon

    return True


if __name__ == "__main__":
    print("hello world")