import globvar


class Variable:
    """
    This variable is a changeable parameter of the Graph.
    """

    def __init__(self, initial_value=None):
        self.value = initial_value
        self.output_nodes = []

        globvar.default_graph.variables.append(self)
