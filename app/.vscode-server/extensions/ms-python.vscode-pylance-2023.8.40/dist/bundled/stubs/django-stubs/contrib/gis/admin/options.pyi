from typing import Any, Dict, Type

from django.contrib.admin import ModelAdmin as ModelAdmin
from django.contrib.gis.forms.widgets import OSMWidget

spherical_mercator_srid: int

class GeoModelAdminMixin:
    gis_widget: Type[OSMWidget]
    gis_widget_kwargs: Dict[Any, Any]

class GISModelAdmin(GeoModelAdminMixin, ModelAdmin[Any]): ...

# NOTE: The model admins bellow will be removed in Django 5.0

class GeoModelAdmin(ModelAdmin[Any]):
    default_lon: int = ...
    default_lat: int = ...
    default_zoom: int = ...
    display_wkt: bool = ...
    display_srid: bool = ...
    extra_js: Any = ...
    num_zoom: int = ...
    max_zoom: bool = ...
    min_zoom: bool = ...
    units: bool = ...
    max_resolution: bool = ...
    max_extent: bool = ...
    modifiable: bool = ...
    mouse_position: bool = ...
    scale_text: bool = ...
    layerswitcher: bool = ...
    scrollable: bool = ...
    map_width: int = ...
    map_height: int = ...
    map_srid: int = ...
    map_template: str = ...
    openlayers_url: str = ...
    point_zoom: Any = ...
    wms_url: str = ...
    wms_layer: str = ...
    wms_name: str = ...
    wms_options: Any = ...
    debug: bool = ...
    widget: Any = ...
    @property
    def media(self) -> Any: ...
    def formfield_for_dbfield(
        self, db_field: Any, request: Any, **kwargs: Any
    ) -> Any: ...
    def get_map_widget(self, db_field: Any) -> Any: ...

class OSMGeoAdmin(GeoModelAdmin):
    map_template: str = ...
    num_zoom: int = ...
    map_srid: Any = ...
    point_zoom: Any = ...
