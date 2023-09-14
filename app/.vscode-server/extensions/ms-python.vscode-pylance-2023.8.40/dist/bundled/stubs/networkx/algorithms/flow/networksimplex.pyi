__all__ = ["network_simplex"]

from itertools import chain, islice, repeat
from math import ceil, sqrt

from ...classes.graph import Graph
from ...utils import not_implemented_for

class _DataEssentialsAndFunctions:
    def __init__(
        self,
        G: Graph,
        multigraph,
        demand="demand",
        capacity="capacity",
        weight="weight",
    ): ...
    def initialize_spanning_tree(self, n, faux_inf): ...
    def find_apex(self, p, q): ...
    def trace_path(self, p, w): ...
    def find_cycle(self, i, p, q): ...
    def augment_flow(self, Wn, We, f): ...
    def trace_subtree(self, p): ...
    def remove_edge(self, s, t): ...
    def make_root(self, q): ...
    def add_edge(self, i, p, q): ...
    def update_potentials(self, i, p, q): ...
    def reduced_cost(self, i): ...
    def find_entering_edges(self): ...
    def residual_capacity(self, i, p): ...
    def find_leaving_edge(self, Wn, We): ...

def network_simplex(
    G: Graph, demand: str = "demand", capacity: str = "capacity", weight: str = "weight"
): ...
