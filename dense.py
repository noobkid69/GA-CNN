import numpy as np

class Dense():
    def __init__(self, input_size, output_size, l2_lambda=0.01):
        self.weights = np.random.randn(output_size, input_size)
        self.bias = np.random.randn(output_size, 1)
        self.l2_lambda = l2_lambda

    def forward(self, input):
        self.input = input
        return np.dot(self.weights, input) + self.bias

    def backward(self, output_gradient, learning_rate):
        weights_gradient = np.dot(output_gradient, self.input.T)

        weights_gradient += self.l2_lambda * self.weights  # L2 regularization

        input_gradient = np.dot(self.weights.T, output_gradient)

        self.weights -= learning_rate * weights_gradient
        self.bias -= learning_rate * output_gradient

        return input_gradient