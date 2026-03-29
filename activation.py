import numpy as np 
def sigmoid(x):
      return 1/(1+np.exp(-x))
def sigmoid_prime(x):
    s = sigmoid(x)
    return s*(1-s)
def relu(x):
    return np.maximum(0, x)
def relu_prime(x):
    return np.where(x > 0, 1, 0)
def softmax(x):
    exp = np.exp(x - np.max(x))   
    return exp / np.sum(exp)

class Activation():
  def __init__(self, func):
     if func == "sigmoid":
        self.func = sigmoid
        self.func_prime = sigmoid_prime
     elif func == "relu":
        self.func = relu
        self.func_prime = relu_prime
     elif func == "softmax":
        self.func = softmax
        self.func_prime = None
     else:
             raise ValueError("Unknown activation")
     self.type = func

  def forward(self, input):
    self.input = input 
    self.output = self.func(input)
    return self.output
  def backward(self, output_gradient, learning_rate):
    if self.type == "softmax":
      # when used with cross entropy this simplifies
      return output_gradient

    return np.multiply(output_gradient, sigmoid_prime(self.input))

