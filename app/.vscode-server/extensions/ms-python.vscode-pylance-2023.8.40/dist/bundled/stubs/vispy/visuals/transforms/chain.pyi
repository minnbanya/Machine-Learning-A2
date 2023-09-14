from vispy.util.svg.transform import Transform
from typing import Sequence
from numpy.typing import ArrayLike, NDArray

# -*- coding: utf-8 -*-
# Copyright (c) Vispy Development Team. All Rights Reserved.
# Distributed under the (new) BSD License. See LICENSE.txt for more info.

from ..shaders import FunctionChain
from .base_transform import BaseTransform
from .linear import NullTransform

class ChainTransform(BaseTransform):

    glsl_map = ...
    glsl_imap = ...

    def __init__(self, *transforms: Sequence[BaseTransform]): ...
    @property
    def transforms(self): ...
    @transforms.setter
    def transforms(self, tr): ...
    @property
    def simplified(self): ...
    @property
    def Linear(self): ...
    @property
    def Orthogonal(self): ...
    @property
    def NonScaling(self): ...
    @property
    def Isometric(self): ...
    def map(self, coords: ArrayLike) -> NDArray: ...
    def imap(self, coords: ArrayLike) -> NDArray: ...
    def shader_map(self): ...
    def shader_imap(self): ...
    def _rebuild_shaders(self): ...
    def append(self, tr: Transform): ...
    def prepend(self, tr: Transform): ...
    def _subtr_changed(self, ev): ...
    def __setitem__(self, index, tr): ...
    def __mul__(self, tr): ...
    def __rmul__(self, tr): ...
    def __str__(self): ...
    def __repr__(self): ...
    def __del__(self): ...

class SimplifiedChainTransform(ChainTransform):
    def __init__(self, chain): ...
    def source_changed(self, event): ...
