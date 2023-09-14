# Python: 3.9.2 (tags/v3.9.2:1a79785, Feb 19 2021, 13:44:55) [MSC v.1928 64 bit (AMD64)]
# Library: numpy, version: 1.20.2
# Module: numpy.random._philox, version: unspecified
import typing
import builtins as _mod_builtins
import numpy.random.bit_generator as _mod_numpy_random_bit_generator

def Lock() -> typing.Any:
    'allocate_lock() -> lock object\n(allocate() is an obsolete synonym)\n\nCreate a new lock object. See help(type(threading.Lock())) for\ninformation about locks.'
    ...

class Philox(_mod_numpy_random_bit_generator.BitGenerator):
    '\n    Philox(seed=None, counter=None, key=None)\n\n    Container for the Philox (4x64) pseudo-random number generator.\n\n    Parameters\n    ----------\n    seed : {None, int, array_like[ints], SeedSequence}, optional\n        A seed to initialize the `BitGenerator`. If None, then fresh,\n        unpredictable entropy will be pulled from the OS. If an ``int`` or\n        ``array_like[ints]`` is passed, then it will be passed to\n        `SeedSequence` to derive the initial `BitGenerator` state. One may also\n        pass in a `SeedSequence` instance.\n    counter : {None, int, array_like}, optional\n        Counter to use in the Philox state. Can be either\n        a Python int (long in 2.x) in [0, 2**256) or a 4-element uint64 array.\n        If not provided, the RNG is initialized at 0.\n    key : {None, int, array_like}, optional\n        Key to use in the Philox state.  Unlike ``seed``, the value in key is\n        directly set. Can be either a Python int in [0, 2**128) or a 2-element\n        uint64 array. `key` and ``seed`` cannot both be used.\n\n    Attributes\n    ----------\n    lock: threading.Lock\n        Lock instance that is shared so that the same bit git generator can\n        be used in multiple Generators without corrupting the state. Code that\n        generates values from a bit generator should hold the bit generator\'s\n        lock.\n\n    Notes\n    -----\n    Philox is a 64-bit PRNG that uses a counter-based design based on weaker\n    (and faster) versions of cryptographic functions [1]_. Instances using\n    different values of the key produce independent sequences.  Philox has a\n    period of :math:`2^{256} - 1` and supports arbitrary advancing and jumping\n    the sequence in increments of :math:`2^{128}`. These features allow\n    multiple non-overlapping sequences to be generated.\n\n    ``Philox`` provides a capsule containing function pointers that produce\n    doubles, and unsigned 32 and 64- bit integers. These are not\n    directly consumable in Python and must be consumed by a ``Generator``\n    or similar object that supports low-level access.\n\n    **State and Seeding**\n\n    The ``Philox`` state vector consists of a 256-bit value encoded as\n    a 4-element uint64 array and a 128-bit value encoded as a 2-element uint64\n    array. The former is a counter which is incremented by 1 for every 4 64-bit\n    randoms produced. The second is a key which determined the sequence\n    produced. Using different keys produces independent sequences.\n\n    The input ``seed`` is processed by `SeedSequence` to generate the key. The\n    counter is set to 0.\n\n    Alternately, one can omit the ``seed`` parameter and set the ``key`` and\n    ``counter`` directly.\n\n    **Parallel Features**\n\n    The preferred way to use a BitGenerator in parallel applications is to use\n    the `SeedSequence.spawn` method to obtain entropy values, and to use these\n    to generate new BitGenerators:\n\n    >>> from numpy.random import Generator, Philox, SeedSequence\n    >>> sg = SeedSequence(1234)\n    >>> rg = [Generator(Philox(s)) for s in sg.spawn(10)]\n\n    ``Philox`` can be used in parallel applications by calling the ``jumped``\n    method  to advances the state as-if :math:`2^{128}` random numbers have\n    been generated. Alternatively, ``advance`` can be used to advance the\n    counter for any positive step in [0, 2**256). When using ``jumped``, all\n    generators should be chained to ensure that the segments come from the same\n    sequence.\n\n    >>> from numpy.random import Generator, Philox\n    >>> bit_generator = Philox(1234)\n    >>> rg = []\n    >>> for _ in range(10):\n    ...    rg.append(Generator(bit_generator))\n    ...    bit_generator = bit_generator.jumped()\n\n    Alternatively, ``Philox`` can be used in parallel applications by using\n    a sequence of distinct keys where each instance uses different key.\n\n    >>> key = 2**96 + 2**33 + 2**17 + 2**9\n    >>> rg = [Generator(Philox(key=key+i)) for i in range(10)]\n\n    **Compatibility Guarantee**\n\n    ``Philox`` makes a guarantee that a fixed ``seed`` will always produce\n    the same random integer stream.\n\n    Examples\n    --------\n    >>> from numpy.random import Generator, Philox\n    >>> rg = Generator(Philox(1234))\n    >>> rg.standard_normal()\n    0.123  # random\n\n    References\n    ----------\n    .. [1] John K. Salmon, Mark A. Moraes, Ron O. Dror, and David E. Shaw,\n           "Parallel Random Numbers: As Easy as 1, 2, 3," Proceedings of\n           the International Conference for High Performance Computing,\n           Networking, Storage and Analysis (SC11), New York, NY: ACM, 2011.\n    '
    def __init__(self, seed=..., counter=..., key=...) -> None:
        '\n    Philox(seed=None, counter=None, key=None)\n\n    Container for the Philox (4x64) pseudo-random number generator.\n\n    Parameters\n    ----------\n    seed : {None, int, array_like[ints], SeedSequence}, optional\n        A seed to initialize the `BitGenerator`. If None, then fresh,\n        unpredictable entropy will be pulled from the OS. If an ``int`` or\n        ``array_like[ints]`` is passed, then it will be passed to\n        `SeedSequence` to derive the initial `BitGenerator` state. One may also\n        pass in a `SeedSequence` instance.\n    counter : {None, int, array_like}, optional\n        Counter to use in the Philox state. Can be either\n        a Python int (long in 2.x) in [0, 2**256) or a 4-element uint64 array.\n        If not provided, the RNG is initialized at 0.\n    key : {None, int, array_like}, optional\n        Key to use in the Philox state.  Unlike ``seed``, the value in key is\n        directly set. Can be either a Python int in [0, 2**128) or a 2-element\n        uint64 array. `key` and ``seed`` cannot both be used.\n\n    Attributes\n    ----------\n    lock: threading.Lock\n        Lock instance that is shared so that the same bit git generator can\n        be used in multiple Generators without corrupting the state. Code that\n        generates values from a bit generator should hold the bit generator\'s\n        lock.\n\n    Notes\n    -----\n    Philox is a 64-bit PRNG that uses a counter-based design based on weaker\n    (and faster) versions of cryptographic functions [1]_. Instances using\n    different values of the key produce independent sequences.  Philox has a\n    period of :math:`2^{256} - 1` and supports arbitrary advancing and jumping\n    the sequence in increments of :math:`2^{128}`. These features allow\n    multiple non-overlapping sequences to be generated.\n\n    ``Philox`` provides a capsule containing function pointers that produce\n    doubles, and unsigned 32 and 64- bit integers. These are not\n    directly consumable in Python and must be consumed by a ``Generator``\n    or similar object that supports low-level access.\n\n    **State and Seeding**\n\n    The ``Philox`` state vector consists of a 256-bit value encoded as\n    a 4-element uint64 array and a 128-bit value encoded as a 2-element uint64\n    array. The former is a counter which is incremented by 1 for every 4 64-bit\n    randoms produced. The second is a key which determined the sequence\n    produced. Using different keys produces independent sequences.\n\n    The input ``seed`` is processed by `SeedSequence` to generate the key. The\n    counter is set to 0.\n\n    Alternately, one can omit the ``seed`` parameter and set the ``key`` and\n    ``counter`` directly.\n\n    **Parallel Features**\n\n    The preferred way to use a BitGenerator in parallel applications is to use\n    the `SeedSequence.spawn` method to obtain entropy values, and to use these\n    to generate new BitGenerators:\n\n    >>> from numpy.random import Generator, Philox, SeedSequence\n    >>> sg = SeedSequence(1234)\n    >>> rg = [Generator(Philox(s)) for s in sg.spawn(10)]\n\n    ``Philox`` can be used in parallel applications by calling the ``jumped``\n    method  to advances the state as-if :math:`2^{128}` random numbers have\n    been generated. Alternatively, ``advance`` can be used to advance the\n    counter for any positive step in [0, 2**256). When using ``jumped``, all\n    generators should be chained to ensure that the segments come from the same\n    sequence.\n\n    >>> from numpy.random import Generator, Philox\n    >>> bit_generator = Philox(1234)\n    >>> rg = []\n    >>> for _ in range(10):\n    ...    rg.append(Generator(bit_generator))\n    ...    bit_generator = bit_generator.jumped()\n\n    Alternatively, ``Philox`` can be used in parallel applications by using\n    a sequence of distinct keys where each instance uses different key.\n\n    >>> key = 2**96 + 2**33 + 2**17 + 2**9\n    >>> rg = [Generator(Philox(key=key+i)) for i in range(10)]\n\n    **Compatibility Guarantee**\n\n    ``Philox`` makes a guarantee that a fixed ``seed`` will always produce\n    the same random integer stream.\n\n    Examples\n    --------\n    >>> from numpy.random import Generator, Philox\n    >>> rg = Generator(Philox(1234))\n    >>> rg.standard_normal()\n    0.123  # random\n\n    References\n    ----------\n    .. [1] John K. Salmon, Mark A. Moraes, Ron O. Dror, and David E. Shaw,\n           "Parallel Random Numbers: As Easy as 1, 2, 3," Proceedings of\n           the International Conference for High Performance Computing,\n           Networking, Storage and Analysis (SC11), New York, NY: ACM, 2011.\n    '
        ...
    
    @classmethod
    def __init_subclass__(cls) -> None:
        'This method is called when a class is subclassed.\n\nThe default implementation does nothing. It may be\noverridden to extend subclasses.\n'
        ...
    
    __pyx_vtable__: PyCapsule
    def __reduce_cython__(self) -> typing.Any:
        ...
    
    def __setstate_cython__(self) -> typing.Any:
        ...
    
    @classmethod
    def __subclasshook__(cls, subclass: typing.Any) -> bool:
        'Abstract classes can override this to customize issubclass().\n\nThis is invoked early on by abc.ABCMeta.__subclasscheck__().\nIt should return True, False or NotImplemented.  If it returns\nNotImplemented, the normal algorithm is used.  Otherwise, it\noverrides the normal algorithm (and the outcome is cached).\n'
        ...
    
    def advance(self, delta) -> typing.Any:
        '\n        advance(delta)\n\n        Advance the underlying RNG as-if delta draws have occurred.\n\n        Parameters\n        ----------\n        delta : integer, positive\n            Number of draws to advance the RNG. Must be less than the\n            size state variable in the underlying RNG.\n\n        Returns\n        -------\n        self : Philox\n            RNG advanced delta steps\n\n        Notes\n        -----\n        Advancing a RNG updates the underlying RNG state as-if a given\n        number of calls to the underlying RNG have been made. In general\n        there is not a one-to-one relationship between the number output\n        random values from a particular distribution and the number of\n        draws from the core RNG.  This occurs for two reasons:\n\n        * The random values are simulated using a rejection-based method\n          and so, on average, more than one value from the underlying\n          RNG is required to generate an single draw.\n        * The number of bits required to generate a simulated value\n          differs from the number of bits generated by the underlying\n          RNG.  For example, two 16-bit integer values can be simulated\n          from a single draw of a 32-bit RNG.\n\n        Advancing the RNG state resets any pre-computed random numbers.\n        This is required to ensure exact reproducibility.\n        '
        ...
    
    def jumped(self, jumps=...) -> typing.Any:
        '\n        jumped(jumps=1)\n\n        Returns a new bit generator with the state jumped\n\n        The state of the returned big generator is jumped as-if\n        2**(128 * jumps) random numbers have been generated.\n\n        Parameters\n        ----------\n        jumps : integer, positive\n            Number of times to jump the state of the bit generator returned\n\n        Returns\n        -------\n        bit_generator : Philox\n            New instance of generator jumped iter times\n        '
        ...
    
    @property
    def state(self) -> typing.Any:
        '\n        Get or set the PRNG state\n\n        Returns\n        -------\n        state : dict\n            Dictionary containing the information required to describe the\n            state of the PRNG\n        '
        ...
    
    def __getattr__(self, name) -> typing.Any:
        ...
    

__all__: list
__doc__: typing.Any
__file__: str
__name__: str
__package__: str
__test__: dict
def __getattr__(name) -> typing.Any:
    ...

