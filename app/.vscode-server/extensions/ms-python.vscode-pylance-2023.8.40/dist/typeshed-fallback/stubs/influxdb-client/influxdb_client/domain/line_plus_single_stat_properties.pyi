from _typeshed import Incomplete

from influxdb_client.domain.view_properties import ViewProperties

class LinePlusSingleStatProperties(ViewProperties):
    openapi_types: Incomplete
    attribute_map: Incomplete
    discriminator: Incomplete
    def __init__(
        self,
        time_format: Incomplete | None = None,
        type: Incomplete | None = None,
        queries: Incomplete | None = None,
        colors: Incomplete | None = None,
        shape: Incomplete | None = None,
        note: Incomplete | None = None,
        show_note_when_empty: Incomplete | None = None,
        axes: Incomplete | None = None,
        static_legend: Incomplete | None = None,
        x_column: Incomplete | None = None,
        generate_x_axis_ticks: Incomplete | None = None,
        x_total_ticks: Incomplete | None = None,
        x_tick_start: Incomplete | None = None,
        x_tick_step: Incomplete | None = None,
        y_column: Incomplete | None = None,
        generate_y_axis_ticks: Incomplete | None = None,
        y_total_ticks: Incomplete | None = None,
        y_tick_start: Incomplete | None = None,
        y_tick_step: Incomplete | None = None,
        shade_below: Incomplete | None = None,
        hover_dimension: Incomplete | None = None,
        position: Incomplete | None = None,
        prefix: Incomplete | None = None,
        suffix: Incomplete | None = None,
        decimal_places: Incomplete | None = None,
        legend_colorize_rows: Incomplete | None = None,
        legend_hide: Incomplete | None = None,
        legend_opacity: Incomplete | None = None,
        legend_orientation_threshold: Incomplete | None = None,
    ) -> None: ...
    @property
    def time_format(self): ...
    @time_format.setter
    def time_format(self, time_format) -> None: ...
    @property
    def type(self): ...
    @type.setter
    def type(self, type) -> None: ...
    @property
    def queries(self): ...
    @queries.setter
    def queries(self, queries) -> None: ...
    @property
    def colors(self): ...
    @colors.setter
    def colors(self, colors) -> None: ...
    @property
    def shape(self): ...
    @shape.setter
    def shape(self, shape) -> None: ...
    @property
    def note(self): ...
    @note.setter
    def note(self, note) -> None: ...
    @property
    def show_note_when_empty(self): ...
    @show_note_when_empty.setter
    def show_note_when_empty(self, show_note_when_empty) -> None: ...
    @property
    def axes(self): ...
    @axes.setter
    def axes(self, axes) -> None: ...
    @property
    def static_legend(self): ...
    @static_legend.setter
    def static_legend(self, static_legend) -> None: ...
    @property
    def x_column(self): ...
    @x_column.setter
    def x_column(self, x_column) -> None: ...
    @property
    def generate_x_axis_ticks(self): ...
    @generate_x_axis_ticks.setter
    def generate_x_axis_ticks(self, generate_x_axis_ticks) -> None: ...
    @property
    def x_total_ticks(self): ...
    @x_total_ticks.setter
    def x_total_ticks(self, x_total_ticks) -> None: ...
    @property
    def x_tick_start(self): ...
    @x_tick_start.setter
    def x_tick_start(self, x_tick_start) -> None: ...
    @property
    def x_tick_step(self): ...
    @x_tick_step.setter
    def x_tick_step(self, x_tick_step) -> None: ...
    @property
    def y_column(self): ...
    @y_column.setter
    def y_column(self, y_column) -> None: ...
    @property
    def generate_y_axis_ticks(self): ...
    @generate_y_axis_ticks.setter
    def generate_y_axis_ticks(self, generate_y_axis_ticks) -> None: ...
    @property
    def y_total_ticks(self): ...
    @y_total_ticks.setter
    def y_total_ticks(self, y_total_ticks) -> None: ...
    @property
    def y_tick_start(self): ...
    @y_tick_start.setter
    def y_tick_start(self, y_tick_start) -> None: ...
    @property
    def y_tick_step(self): ...
    @y_tick_step.setter
    def y_tick_step(self, y_tick_step) -> None: ...
    @property
    def shade_below(self): ...
    @shade_below.setter
    def shade_below(self, shade_below) -> None: ...
    @property
    def hover_dimension(self): ...
    @hover_dimension.setter
    def hover_dimension(self, hover_dimension) -> None: ...
    @property
    def position(self): ...
    @position.setter
    def position(self, position) -> None: ...
    @property
    def prefix(self): ...
    @prefix.setter
    def prefix(self, prefix) -> None: ...
    @property
    def suffix(self): ...
    @suffix.setter
    def suffix(self, suffix) -> None: ...
    @property
    def decimal_places(self): ...
    @decimal_places.setter
    def decimal_places(self, decimal_places) -> None: ...
    @property
    def legend_colorize_rows(self): ...
    @legend_colorize_rows.setter
    def legend_colorize_rows(self, legend_colorize_rows) -> None: ...
    @property
    def legend_hide(self): ...
    @legend_hide.setter
    def legend_hide(self, legend_hide) -> None: ...
    @property
    def legend_opacity(self): ...
    @legend_opacity.setter
    def legend_opacity(self, legend_opacity) -> None: ...
    @property
    def legend_orientation_threshold(self): ...
    @legend_orientation_threshold.setter
    def legend_orientation_threshold(self, legend_orientation_threshold) -> None: ...
    def to_dict(self): ...
    def to_str(self): ...
    def __eq__(self, other): ...
    def __ne__(self, other): ...
