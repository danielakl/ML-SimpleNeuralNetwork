import matplotlib.pyplot as plt
import numpy as np

from Graph import Graph
from Operation.Sigmoid import Sigmoid
from Operation.add import add
from Operation.matmul import matmul
from Operation.multiply import multiply
from Placeholder import Placeholder
from Session import Session
from Variable import Variable


# Method for executing calculation.
def calculate(A, b, x1):
    # Setup graph
    g = Graph()
    g.set_as_default()

    # Setup variables
    A = Variable(A)
    b = Variable(b)

    # Setup placeholder
    x = Placeholder()

    # Setup operations
    y = multiply(A, x)
    if type(A) == list:
        y = matmul(A, x)

    z = add(y, b)

    # Setup session to execute calculation.
    sess = Session()
    return sess.run(operation=z, feed_dict={x: x1})


def classify(W, b, x1):
    # Setup graph
    g = Graph()
    g.set_as_default()

    # Setup variables
    W = Variable(W)
    b = Variable(b)

    # Setup placeholder
    x = Placeholder()

    # Setup operations
    z = matmul(W, x)
    z = add(z, b)
    z = Sigmoid(z)

    # Setup session to execute operations.
    sess = Session()
    return sess.run(operation=z, feed_dict={x: x1})


# Change these values to get different results.
A = 10
b = 1
x = 10

# Setting up calculation for this expression:
# z = Ax + b -> z = 10x + 1
print("Expression: z = Ax + b")
print("A:", A)
print("b:", b)
print("Result:", calculate(A, b, x), "\n\n")


# Same as above only with arrays.
# Define vector/matrix.
A = [[10, 20], [30, 40]]
b = [2, 2]
x = 10

# Setting up calculation for this expression:
# z = Ax + b -> z = 10x + 1
print("Expression: z = Ax + b")
print("A:", A)
print("b:", b)
print("Result:\n", calculate(A, b, x), "\n\n")

# Setting up classification example
features = [[7.3402781, 9.36149154], [9.13332743, 8.74906102], [1.99243535, -8.85885722],
            [7.38443759, 7.72520389], [7.97613887, 8.80878209], [7.76974352, 9.50899462],
            [8.3186688,  10.1026025], [8.79588546,   7.28046702], [9.81270381,   9.46968531],
            [1.57961049,  -8.17089971], [0.06441546,  -9.04982817], [7.2075117,   7.04533624],
            [9.10704928,   9.0272212], [1.82921897,  -9.86956281], [7.85036314,   7.986659],
            [3.04605603,  -7.50486114], [1.85582689,  -6.74473432], [2.88603902,  -8.85261704],
            [-1.20046211,  -9.55928542], [2.00890845,  -9.78471782], [7.68945113,   9.01706723],
            [6.42356167,   8.33356412], [8.15467319,   7.87489634], [1.92000795,  -7.50953708],
            [1.90073973,  -7.24386675], [7.7605855,   7.05124418], [6.90561582,   9.23493842],
            [0.65582768,  -9.5920878], [1.41804346,  -8.10517372], [9.65371965,   9.35409538],
            [1.23053506,  -7.98873571], [1.96322881,  -9.50169117], [6.11644251,   9.26709393],
            [7.70630321,  10.78862346], [0.79580385,  -9.00301023], [3.13114921,  -8.6849493],
            [1.3970852,  -7.25918415], [7.27808709,   7.15201886], [1.06965742,  -8.1648251],
            [6.37298915,   9.77705761], [7.24898455,   8.85834104], [2.09335725,  -7.66278316],
            [1.05865542,  -8.43841416], [6.43807502,   7.85483418], [6.94948313,   8.75248232],
            [-0.07326715, -11.69999644], [0.61463602,  -9.51908883], [1.31977821,  -7.2710667],
            [2.72532584,  -7.51956557], [8.20949206,  11.90419283]]

labels = [1, 1, 0, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 0, 1, 0, 0, 0, 0, 0, 1, 1, 1,
          0, 0, 1, 1, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 1, 0, 1, 1, 0, 0, 1, 1, 0,
          0, 0, 0, 1]

# Reshape array
features = np.reshape(features, (50, 2))

# Visualize data
# Red dots will be classified as 1
# Blue dots will be classified as 0
plt.scatter(features[:, 0], features[:, 1], c=labels, cmap='coolwarm')
plt.title("Data Visualization")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.xlim(-13, 13)
plt.ylim(-13, 13)
plt.show()

# Change x and y to check classification of different dots.
x = 8
y = 10
print("Dot at x:", x, "y:", y)
print("Value:", classify([1, 1], -5, [x, y]))
