from _typeshed import Incomplete

from influxdb_client.domain.view_properties import ViewProperties

class ScatterViewProperties(ViewProperties):
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
        fill_columns: Incomplete | None = None,
        symbol_columns: Incomplete | None = None,
        x_domain: Incomplete | None = None,
        y_domain: Incomplete | None = None,
        x_axis_label: Incomplete | None = None,
        y_axis_label: Incomplete | None = None,
        x_prefix: Incomplete | None = None,
        x_suffix: Incomplete | None = None,
        y_prefix: Incomplete | None = None,
        y_suffix: Incomplete | None = None,
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
    def fill_columns(self): ...
    @fill_columns.setter
    def fill_columns(self, fill_columns) -> None: ...
    @property
    def symbol_columns(self): ...
    @symbol_columns.setter
    def symbol_columns(self, symbol_columns) -> None: ...
    @property
    def x_domain(self): ...
    @x_domain.setter
    def x_domain(self, x_domain) -> None: ...
    @property
    def y_domain(self): ...
    @y_domain.setter
    def y_domain(self, y_domain) -> None: ...
    @property
    def x_axis_label(self): ...
    @x_axis_label.setter
    def x_axis_label(self, x_axis_label) -> None: ...
    @property
    def y_axis_label(self): ...
    @y_axis_label.setter
    def y_axis_label(self, y_axis_label) -> None: ...
    @property
    def x_prefix(self): ...
    @x_prefix.setter
    def x_prefix(self, x_prefix) -> None: ...
    @property
    def x_suffix(self): ...
    @x_suffix.setter
    def x_suffix(self, x_suffix) -> None: ...
    @property
    def y_prefix(self): ...
    @y_prefix.setter
    def y_prefix(self, y_prefix) -> None: ...
    @property
    def y_suffix(self): ...
    @y_suffix.setter
    def y_suffix(self, y_suffix) -> None: ...
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
