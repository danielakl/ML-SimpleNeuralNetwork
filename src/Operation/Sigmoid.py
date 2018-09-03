import numpy as np

from Operation.Operation import Operation


class Sigmoid(Operation):
    def __init__(self, z):
        super(Sigmoid, self).__init__([z])

    def compute(self, z):
        return 1 / (1 + np.exp(-z))
