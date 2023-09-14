# Python: 3.8.2 (tags/v3.8.2:7b3ab59, Feb 25 2020, 23:03:10) [MSC v.1916 64 bit (AMD64)]
# Library: scipy, version: 1.6.2
# Module: scipy.sparse.csgraph._matching, version: unspecified
import typing
import builtins as _mod_builtins
import numpy as _mod_numpy
import scipy.sparse.csr as _mod_scipy_sparse_csr

BTYPE = _mod_numpy.uint8
DTYPE = _mod_numpy.float64
ITYPE = _mod_numpy.int32
__doc__: typing.Any
__file__: str
__name__: str
__package__: str
def __pyx_unpickle_Enum() -> typing.Any:
    ...

__test__: dict
csr_matrix = _mod_scipy_sparse_csr.csr_matrix
def isspmatrix_coo(x) -> typing.Any:
    'Is x of coo_matrix type?\n\n    Parameters\n    ----------\n    x\n        object to check for being a coo matrix\n\n    Returns\n    -------\n    bool\n        True if x is a coo matrix, False otherwise\n\n    Examples\n    --------\n    >>> from scipy.sparse import coo_matrix, isspmatrix_coo\n    >>> isspmatrix_coo(coo_matrix([[5]]))\n    True\n\n    >>> from scipy.sparse import coo_matrix, csr_matrix, isspmatrix_coo\n    >>> isspmatrix_coo(csr_matrix([[5]]))\n    False\n    '
    ...

def isspmatrix_csc(x) -> typing.Any:
    'Is x of csc_matrix type?\n\n    Parameters\n    ----------\n    x\n        object to check for being a csc matrix\n\n    Returns\n    -------\n    bool\n        True if x is a csc matrix, False otherwise\n\n    Examples\n    --------\n    >>> from scipy.sparse import csc_matrix, isspmatrix_csc\n    >>> isspmatrix_csc(csc_matrix([[5]]))\n    True\n\n    >>> from scipy.sparse import csc_matrix, csr_matrix, isspmatrix_csc\n    >>> isspmatrix_csc(csr_matrix([[5]]))\n    False\n    '
    ...

def isspmatrix_csr(x) -> typing.Any:
    'Is x of csr_matrix type?\n\n    Parameters\n    ----------\n    x\n        object to check for being a csr matrix\n\n    Returns\n    -------\n    bool\n        True if x is a csr matrix, False otherwise\n\n    Examples\n    --------\n    >>> from scipy.sparse import csr_matrix, isspmatrix_csr\n    >>> isspmatrix_csr(csr_matrix([[5]]))\n    True\n\n    >>> from scipy.sparse import csc_matrix, csr_matrix, isspmatrix_csc\n    >>> isspmatrix_csr(csc_matrix([[5]]))\n    False\n    '
    ...

def maximum_bipartite_matching(graph, perm_type=...) -> typing.Any:
    '\n    maximum_bipartite_matching(graph, perm_type=\'row\')\n\n    Returns a matching of a bipartite graph whose cardinality is as least that\n    of any given matching of the graph.\n\n    Parameters\n    ----------\n    graph : sparse matrix\n        Input sparse in CSR format whose rows represent one partition of the\n        graph and whose columns represent the other partition. An edge between\n        two vertices is indicated by the corresponding entry in the matrix\n        existing in its sparse representation.\n    perm_type : str, {\'row\', \'column\'}\n        Which partition to return the matching in terms of: If ``\'row\'``, the\n        function produces an array whose length is the number of columns in the\n        input, and whose :math:`j`\'th element is the row matched to the\n        :math:`j`\'th column. Conversely, if ``perm_type`` is ``\'column\'``, this\n        returns the columns matched to each row.\n\n    Returns\n    -------\n    perm : ndarray\n        A matching of the vertices in one of the two partitions. Unmatched\n        vertices are represented by a ``-1`` in the result.\n\n    Notes\n    -----\n    This function implements the Hopcroft--Karp algorithm [1]_. Its time\n    complexity is :math:`O(\\lvert E \\rvert \\sqrt{\\lvert V \\rvert})`, and its\n    space complexity is linear in the number of rows. In practice, this\n    asymmetry between rows and columns means that it can be more efficient to\n    transpose the input if it contains more columns than rows.\n\n    By Konig\'s theorem, the cardinality of the matching is also the number of\n    vertices appearing in a minimum vertex cover of the graph.\n\n    Note that if the sparse representation contains explicit zeros, these are\n    still counted as edges.\n\n    The implementation was changed in SciPy 1.4.0 to allow matching of general\n    bipartite graphs, where previous versions would assume that a perfect\n    matching existed. As such, code written against 1.4.0 will not necessarily\n    work on older versions.\n\n    References\n    ----------\n    .. [1] John E. Hopcroft and Richard M. Karp. "An n^{5 / 2} Algorithm for\n           Maximum Matchings in Bipartite Graphs" In: SIAM Journal of Computing\n           2.4 (1973), pp. 225--231. :doi:`10.1137/0202019`\n\n    Examples\n    --------\n    >>> from scipy.sparse import csr_matrix\n    >>> from scipy.sparse.csgraph import maximum_bipartite_matching\n\n    As a simple example, consider a bipartite graph in which the partitions\n    contain 2 and 3 elements respectively. Suppose that one partition contains\n    vertices labelled 0 and 1, and that the other partition contains vertices\n    labelled A, B, and C. Suppose that there are edges connecting 0 and C,\n    1 and A, and 1 and B. This graph would then be represented by the following\n    sparse matrix:\n\n    >>> graph = csr_matrix([[0, 0, 1], [1, 1, 0]])\n\n    Here, the 1s could be anything, as long as they end up being stored as\n    elements in the sparse matrix. We can now calculate maximum matchings as\n    follows:\n\n    >>> print(maximum_bipartite_matching(graph, perm_type=\'column\'))\n    [2 0]\n    >>> print(maximum_bipartite_matching(graph, perm_type=\'row\'))\n    [ 1 -1  0]\n\n    The first output tells us that 1 and 2 are matched with C and A\n    respectively, and the second output tells us that A, B, and C are matched\n    with 1, nothing, and 0 respectively.\n\n    Note that explicit zeros are still converted to edges. This means that a\n    different way to represent the above graph is by using the CSR structure\n    directly as follows:\n\n    >>> data = [0, 0, 0]\n    >>> indices = [2, 0, 1]\n    >>> indptr = [0, 1, 3]\n    >>> graph = csr_matrix((data, indices, indptr))\n    >>> print(maximum_bipartite_matching(graph, perm_type=\'column\'))\n    [2 0]\n    >>> print(maximum_bipartite_matching(graph, perm_type=\'row\'))\n    [ 1 -1  0]\n\n    When one or both of the partitions are empty, the matching is empty as\n    well:\n\n    >>> graph = csr_matrix((2, 0))\n    >>> print(maximum_bipartite_matching(graph, perm_type=\'column\'))\n    [-1 -1]\n    >>> print(maximum_bipartite_matching(graph, perm_type=\'row\'))\n    []\n\n    When the input matrix is square, and the graph is known to admit a perfect\n    matching, i.e. a matching with the property that every vertex in the graph\n    belongs to some edge in the matching, then one can view the output as the\n    permutation of rows (or columns) turning the input matrix into one with the\n    property that all diagonal elements are non-empty:\n\n    >>> a = [[0, 1, 2, 0], [1, 0, 0, 1], [2, 0, 0, 3], [0, 1, 3, 0]]\n    >>> graph = csr_matrix(a)\n    >>> perm = maximum_bipartite_matching(graph, perm_type=\'row\')\n    >>> print(graph[perm].toarray())\n    [[1 0 0 1]\n     [0 1 2 0]\n     [0 1 3 0]\n     [2 0 0 3]]\n\n    '
    ...

def min_weight_full_bipartite_matching(biadjacency_matrix, maximize=...) -> typing.Any:
    '\n    min_weight_full_bipartite_matching(biadjacency_matrix, maximize=False)\n\n    Returns the minimum weight full matching of a bipartite graph.\n\n    .. versionadded:: 1.6.0\n\n    Parameters\n    ----------\n    biadjacency_matrix : sparse matrix\n        Biadjacency matrix of the bipartite graph: A sparse matrix in CSR, CSC,\n        or COO format whose rows represent one partition of the graph and whose\n        columns represent the other partition. An edge between two vertices is\n        indicated by the corresponding entry in the matrix, and the weight of\n        the edge is given by the value of that entry. This should not be\n        confused with the full adjacency matrix of the graph, as we only need\n        the submatrix defining the bipartite structure.\n\n    maximize : bool (default: False)\n        Calculates a maximum weight matching if true.\n\n    Returns\n    -------\n    row_ind, col_ind : array\n        An array of row indices and one of corresponding column indices giving\n        the optimal matching. The total weight of the matching can be computed\n        as ``graph[row_ind, col_ind].sum()``. The row indices will be\n        sorted; in the case of a square matrix they will be equal to\n        ``numpy.arange(graph.shape[0])``.\n\n    Notes\n    -----\n\n    Let :math:`G = ((U, V), E)` be a weighted bipartite graph with non-zero\n    weights :math:`w : E \\to \\mathbb{R} \\setminus \\{0\\}`. This function then\n    produces a matching :math:`M \\subseteq E` with cardinality\n\n    .. math::\n       \\lvert M \\rvert = \\min(\\lvert U \\rvert, \\lvert V \\rvert),\n\n    which minimizes the sum of the weights of the edges included in the\n    matching, :math:`\\sum_{e \\in M} w(e)`, or raises an error if no such\n    matching exists.\n\n    When :math:`\\lvert U \\rvert = \\lvert V \\rvert`, this is commonly\n    referred to as a perfect matching; here, since we allow\n    :math:`\\lvert U \\rvert` and :math:`\\lvert V \\rvert` to differ, we\n    follow Karp [1]_ and refer to the matching as *full*.\n\n    This function implements the LAPJVsp algorithm [2]_, short for "Linear\n    assignment problem, Jonker--Volgenant, sparse".\n\n    The problem it solves is equivalent to the rectangular linear assignment\n    problem. [3]_ As such, this function can be used to solve the same problems\n    as :func:`scipy.optimize.linear_sum_assignment`. That function may perform\n    better when the input is dense, or for certain particular types of inputs,\n    such as those for which the :math:`(i, j)`\'th entry is the distance between\n    two points in Euclidean space.\n\n    If no full matching exists, this function raises a ``ValueError``. For\n    determining the size of the largest matching in the graph, see\n    :func:`maximum_bipartite_matching`.\n\n    We require that weights are non-zero only to avoid issues with the handling\n    of explicit zeros when converting between different sparse representations.\n    Zero weights can be handled by adding a constant to all weights, so that\n    the resulting matrix contains no zeros.\n\n    References\n    ----------\n    .. [1] Richard Manning Karp:\n       An algorithm to Solve the m x n Assignment Problem in Expected Time\n       O(mn log n).\n       Networks, 10(2):143-152, 1980.\n    .. [2] Roy Jonker and Anton Volgenant:\n       A Shortest Augmenting Path Algorithm for Dense and Sparse Linear\n       Assignment Problems.\n       Computing 38:325-340, 1987.\n    .. [3] https://en.wikipedia.org/wiki/Assignment_problem\n\n    Examples\n    --------\n    >>> from scipy.sparse import csr_matrix\n    >>> from scipy.sparse.csgraph import min_weight_full_bipartite_matching\n\n    Let us first consider an example in which all weights are equal:\n\n    >>> biadjacency_matrix = csr_matrix([[1, 1, 1], [1, 0, 0], [0, 1, 0]])\n\n    Here, all we get is a perfect matching of the graph:\n\n    >>> print(min_weight_full_bipartite_matching(biadjacency_matrix)[1])\n    [2 0 1]\n\n    That is, the first, second, and third rows are matched with the third,\n    first, and second column respectively. Note that in this example, the 0\n    in the input matrix does *not* correspond to an edge with weight 0, but\n    rather a pair of vertices not paired by an edge.\n\n    Note also that in this case, the output matches the result of applying\n    :func:`maximum_bipartite_matching`:\n\n    >>> from scipy.sparse.csgraph import maximum_bipartite_matching\n    >>> biadjacency = csr_matrix([[1, 1, 1], [1, 0, 0], [0, 1, 0]])\n    >>> print(maximum_bipartite_matching(biadjacency, perm_type=\'column\'))\n    [2 0 1]\n\n    When multiple edges are available, the ones with lowest weights are\n    preferred:\n\n    >>> biadjacency = csr_matrix([[3, 3, 6], [4, 3, 5], [10, 1, 8]])\n    >>> row_ind, col_ind = min_weight_full_bipartite_matching(biadjacency)\n    >>> print(col_ind)\n    [0 2 1]\n\n    The total weight in this case is :math:`3 + 5 + 1 = 9`:\n\n    >>> print(biadjacency[row_ind, col_ind].sum())\n    9\n\n    When the matrix is not square, i.e. when the two partitions have different\n    cardinalities, the matching is as large as the smaller of the two\n    partitions:\n\n    >>> biadjacency = csr_matrix([[0, 1, 1], [0, 2, 3]])\n    >>> row_ind, col_ind = min_weight_full_bipartite_matching(biadjacency)\n    >>> print(row_ind, col_ind)\n    [0 1] [2 1]\n    >>> biadjacency = csr_matrix([[0, 1], [3, 1], [1, 4]])\n    >>> row_ind, col_ind = min_weight_full_bipartite_matching(biadjacency)\n    >>> print(row_ind, col_ind)\n    [0 2] [1 0]\n\n    When one or both of the partitions are empty, the matching is empty as\n    well:\n\n    >>> biadjacency = csr_matrix((2, 0))\n    >>> row_ind, col_ind = min_weight_full_bipartite_matching(biadjacency)\n    >>> print(row_ind, col_ind)\n    [] []\n\n    In general, we will always reach the same sum of weights as if we had used\n    :func:`scipy.optimize.linear_sum_assignment` but note that for that one,\n    missing edges are represented by a matrix entry of ``float(\'inf\')``. Let us\n    generate a random sparse matrix with integer entries between 1 and 10:\n\n    >>> import numpy as np\n    >>> from scipy.sparse import random\n    >>> from scipy.optimize import linear_sum_assignment\n    >>> sparse = random(10, 10, random_state=42, density=.5, format=\'coo\') * 10\n    >>> sparse.data = np.ceil(sparse.data)\n    >>> dense = sparse.toarray()\n    >>> dense = np.full(sparse.shape, np.inf)\n    >>> dense[sparse.row, sparse.col] = sparse.data\n    >>> sparse = sparse.tocsr()\n    >>> row_ind, col_ind = linear_sum_assignment(dense)\n    >>> print(dense[row_ind, col_ind].sum())\n    28.0\n    >>> row_ind, col_ind = min_weight_full_bipartite_matching(sparse)\n    >>> print(sparse[row_ind, col_ind].sum())\n    28.0\n\n    '
    ...

def __getattr__(name) -> typing.Any:
    ...

