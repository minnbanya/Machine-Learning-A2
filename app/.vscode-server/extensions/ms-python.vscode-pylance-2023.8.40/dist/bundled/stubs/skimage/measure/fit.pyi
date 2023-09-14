from numpy.typing import ArrayLike
from typing import Any
import math
from warnings import warn

import numpy as np
from numpy.linalg import inv
from scipy import optimize, spatial

def _check_data_dim(data, dim): ...
def _check_data_atleast_2D(data): ...

class BaseModel(object):
    def __init__(self): ...

class LineModelND(BaseModel):
    def estimate(self, data) -> bool: ...
    def residuals(self, data, params=None): ...
    def predict(self, x, axis: int = 0, params=None): ...
    def predict_x(self, y: ArrayLike, params=None) -> ArrayLike: ...
    def predict_y(self, x: ArrayLike, params=None) -> ArrayLike: ...

class CircleModel(BaseModel):
    def estimate(self, data) -> bool: ...
    def residuals(self, data): ...
    def predict_xy(self, t: ArrayLike, params=None): ...

class EllipseModel(BaseModel):
    def estimate(self, data) -> bool: ...
    def residuals(self, data): ...
    def predict_xy(self, t: ArrayLike, params=None): ...

def _dynamic_max_trials(n_inliers, n_samples, min_samples, probability): ...
def ransac(
    data,
    model_class: Any,
    min_samples,
    residual_threshold,
    is_data_valid=None,
    is_model_valid=None,
    max_trials: int = 100,
    stop_sample_num: int = ...,
    stop_residuals_sum: float = 0,
    stop_probability=1,
    random_state=None,
    initial_inliers=None,
): ...
