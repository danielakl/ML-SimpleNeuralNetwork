import globvar


class Operation:
    """
    An Operation is a node in a "Graph". TensorFlow will also use this concept of a Graph.

    This Operation class will be inherited by other classes that actually compute the specific
    operation, such as adding or matrix multiplication.
    """

    def __init__(self, input_nodes=[]):
        """
        Initialize an Operation
        """
        self.input_nodes = input_nodes  # The list of input nodes
        self.output_nodes = []  # List of nodes consuming this node's output

        # For every node in the input, we append this operation (self) to the list of
        # the consumers of the input nodes
        for node in input_nodes:
            node.output_nodes.append(self)

        # We will then append this particular operation
        # Append this operation to the list of operations in the currently active default graph
            globvar.default_graph.operations.append(self)

    def compute(self, *args):
        """
        This is a placeholder function. It will be overwritten by the actual specific operation
        that inherits from this class.
        """

        pass
