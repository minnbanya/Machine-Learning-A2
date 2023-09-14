from math import sqrt

from ..classes.graph import Graph
from ..utils import py_random_state

__all__ = ["is_threshold_graph", "find_threshold_graph"]

def is_threshold_graph(G: Graph) -> bool: ...
def is_threshold_sequence(degree_sequence): ...
def creation_sequence(degree_sequence, with_labels=False, compact=False): ...
def make_compact(creation_sequence): ...
def uncompact(creation_sequence): ...
def creation_sequence_to_weights(creation_sequence): ...
def weights_to_creation_sequence(
    weights, threshold=1, with_labels=False, compact=False
): ...

# Manipulating NetworkX.Graphs in context of threshold graphs
def threshold_graph(creation_sequence, create_using=None): ...
def find_alternating_4_cycle(G: Graph): ...
def find_threshold_graph(G: Graph, create_using=None): ...
def find_creation_sequence(G: Graph): ...

# Properties of Threshold Graphs
def triangles(creation_sequence): ...
def triangle_sequence(creation_sequence): ...
def cluster_sequence(creation_sequence): ...
def degree_sequence(creation_sequence): ...
def density(creation_sequence): ...
def degree_correlation(creation_sequence): ...
def shortest_path(creation_sequence, u, v): ...
def shortest_path_length(creation_sequence, i): ...
def betweenness_sequence(creation_sequence, normalized=True): ...
def eigenvectors(creation_sequence): ...
def spectral_projection(u, eigenpairs): ...
def eigenvalues(creation_sequence): ...

# Threshold graph creation routines

@py_random_state(2)
def random_threshold_sequence(n, p, seed=None): ...

# maybe *_d_threshold_sequence routines should
# be (or be called from) a single routine with a more descriptive name
# and a keyword parameter?
def right_d_threshold_sequence(n, m): ...
def left_d_threshold_sequence(n, m): ...
@py_random_state(3)
def swap_d(cs, p_split=1.0, p_combine=1.0, seed=None): ...
