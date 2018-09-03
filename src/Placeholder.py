import globvar


class Placeholder:
    """
    A placeholder is a node that needs to be provided a value for computing the output in the Graph.
    """

    def __init__(self):
        self.output_nodes = []

        globvar.default_graph.placeholders.append(self)
