import numpy as np
from activation import softmax
def binary_cross_entropy(y_true, y_pred):
  y_pred = np.clip(y_pred, 1e-7, 1 - 1e-7)
  return -np.mean(y_true*np.log(y_pred)+(1-y_true)*np.log(1-y_pred))
def binary_cross_entropy_prime(y_true, y_pred):
    y_pred = np.clip(y_pred, 1e-7, 1 - 1e-7)
    return ((1 - y_true) / (1 - y_pred) - y_true / y_pred) / np.size(y_true)
def categorical_cross_entropy(y_true, logits):
    exp = np.exp(logits - np.max(logits))
    probs = softmax(logits)
    return -np.sum(y_true * np.log(probs + 1e-9))


def categorical_cross_entropy_prime(y_true, logits):
    probs = softmax(logits)
    return probs - y_true