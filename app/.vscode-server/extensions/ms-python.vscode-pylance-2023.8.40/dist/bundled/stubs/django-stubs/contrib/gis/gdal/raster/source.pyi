from typing import Any, List, Optional

from django.contrib.gis.gdal.raster.base import GDALRasterBase as GDALRasterBase

class TransformPoint(List[Any]):
    indices: Any = ...
    def __init__(self, raster: Any, prop: Any) -> None: ...
    @property
    def x(self) -> Any: ...
    @x.setter
    def x(self, value: Any) -> None: ...
    @property
    def y(self) -> Any: ...
    @y.setter
    def y(self, value: Any) -> None: ...

class GDALRaster(GDALRasterBase):
    destructor: Any = ...
    def __init__(self, ds_input: Any, write: bool = ...) -> None: ...
    def __del__(self) -> None: ...
    @property
    def vsi_buffer(self) -> Any: ...
    def is_vsi_based(self) -> Any: ...
    @property
    def name(self) -> Any: ...
    def driver(self) -> Any: ...
    @property
    def width(self) -> Any: ...
    @property
    def height(self) -> Any: ...
    @property
    def srs(self) -> Any: ...
    @srs.setter
    def srs(self, value: Any) -> None: ...
    @property
    def srid(self) -> Any: ...
    @srid.setter
    def srid(self, value: Any) -> None: ...
    @property
    def geotransform(self) -> Any: ...
    @geotransform.setter
    def geotransform(self, values: Any) -> None: ...
    @property
    def origin(self) -> Any: ...
    @property
    def scale(self) -> Any: ...
    @property
    def skew(self) -> Any: ...
    @property
    def extent(self) -> Any: ...
    @property
    def bands(self) -> Any: ...
    def warp(
        self, ds_input: Any, resampling: str = ..., max_error: float = ...
    ) -> Any: ...
    def transform(
        self,
        srid: Any,
        driver: Optional[Any] = ...,
        name: Optional[Any] = ...,
        resampling: str = ...,
        max_error: float = ...,
    ) -> Any: ...
    @property
    def info(self) -> Any: ...
