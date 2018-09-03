from Operation.Operation import Operation


class matmul(Operation):

    def __init__(self, a, b):
        super().__init__([a, b])

    def compute(self, a, b):
        self.inputs = [a, b]
        return a.dot(b)
