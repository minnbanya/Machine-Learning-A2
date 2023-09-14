from matplotlib.path import Path
import numpy as np
from ._typing import *
from typing import Callable

class NonIntersectingPathException(ValueError): ...

def get_intersection(cx1, cy1, cos_t1, sin_t1, cx2, cy2, cos_t2, sin_t2)-> tuple[float, float]: ...
def get_normal_points(cx, cy, cos_t, sin_t, length)-> tuple[float, float, float, float]: ...
def split_de_casteljau(beta, t)-> float: ...
def find_bezier_t_intersecting_with_closedpath(
    bezier_point_at_t: Callable,
    inside_closedpath: Callable,
    t0: float = ...,
    t1: float = ...,
    tolerance: float = ...,
) -> tuple[float, float]: ...

class BezierSegment:
    def __init__(self, control_points) -> None: ...
    def __call__(self, t: ArrayLike) -> tuple: ...
    def point_at_t(self, t) -> tuple[float]: ...
    @property
    def control_points(self)-> np.ndarray: ...
    @property
    def dimension(self)-> int: ...
    @property
    def degree(self) -> int: ...
    @property
    def polynomial_coefficients(self): ...
    def axis_aligned_extrema(self) -> tuple: ...

def split_bezier_intersecting_with_closedpath(
    bezier, inside_closedpath: Callable, tolerance: float = ...
) -> tuple[list, list]: ...
def split_path_inout(path, inside, tolerance=..., reorder_inout=...)-> tuple[Path, Path]: ...
def inside_circle(cx, cy, r) -> Callable: ...
def get_cos_sin(x0, y0, x1, y1)-> tuple[float, float]: ...
def check_if_parallel(
    dx1: float, dy1: float, dx2: float, dy2: float, tolerance: float = ...
) -> bool: ...
def get_parallels(bezier2, width)-> tuple[list[tuple[float, float]], list[tuple[float, float]]]: ...
def find_control_points(c1x, c1y, mmx, mmy, c2x, c2y)-> list[tuple[float, float]]: ...
def make_wedged_bezier2(bezier2, width, w1=..., wm=..., w2=...)-> tuple[list[tuple[float, float]], list[tuple[float, float]]]: ...
