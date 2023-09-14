from typing import Callable, Type
from matplotlib import backend_tools
from matplotlib._api import classproperty
from matplotlib.backend_bases import FigureCanvasBase, ToolContainerBase
from gi.repository import Gtk
from ._backend_gtk import (
    TimerGTK as TimerGTK3,
    _BackendGTK,
    _FigureManagerGTK,
    _NavigationToolbar2GTK,
)

class __getattr__:
    @property
    def cursord(self)-> dict: ...

    icon_filename: Callable = ...
    window_icon: Callable = ...

class FigureCanvasGTK3(Gtk.DrawingArea, FigureCanvasBase):
    required_interactive_framework: str = ...
    _timer_cls: Type[TimerGTK3] = TimerGTK3
    manager_class: classproperty = ...
    event_mask: int = ...

    def __init__(self, figure=...) -> None: ...
    def destroy(self)-> None: ...
    def set_cursor(self, cursor)-> None: ...
    def scroll_event(self, widget, event)-> bool: ...
    def button_press_event(self, widget, event)-> bool: ...
    def button_release_event(self, widget, event)-> bool: ...
    def key_press_event(self, widget, event)-> bool: ...
    def key_release_event(self, widget, event)-> bool: ...
    def motion_notify_event(self, widget, event)-> bool: ...
    def leave_notify_event(self, widget, event)-> None: ...
    def enter_notify_event(self, widget, event)-> None: ...
    def size_allocate(self, widget, allocation)-> None: ...
    def configure_event(self, widget, event)-> bool: ...
    def on_draw_event(self, widget, ctx)-> None: ...
    def draw(self)-> None: ...
    def draw_idle(self)-> None: ...
    def flush_events(self)-> None: ...

class NavigationToolbar2GTK3(_NavigationToolbar2GTK, Gtk.Toolbar):
    def __init__(self, canvas, window=...) -> None: ...

    win: Callable = ...
    def save_figure(self, *args)-> None: ...

class ToolbarGTK3(ToolContainerBase, Gtk.Box):
    def __init__(self, toolmanager) -> None: ...
    def add_toolitem(self, name, group, position, image_file, description, toggle)-> None: ...
    def toggle_toolitem(self, name, toggled)-> None: ...
    def remove_toolitem(self, name)-> None: ...
    def set_message(self, s: str)-> None: ...

class SaveFigureGTK3(backend_tools.SaveFigureBase):
    def trigger(self, *args, **kwargs)-> None: ...

class SetCursorGTK3(backend_tools.SetCursorBase):
    def set_cursor(self, cursor)-> None: ...

class HelpGTK3(backend_tools.ToolHelpBase):
    def trigger(self, *args)-> None: ...

class ToolCopyToClipboardGTK3(backend_tools.ToolCopyToClipboardBase):
    def trigger(self, *args, **kwargs)-> None: ...

def error_msg_gtk(msg, parent=...)-> None: ...

Toolbar: Type[ToolbarGTK3] = ToolbarGTK3

class FigureManagerGTK3(_FigureManagerGTK):
    ...

class _BackendGTK3(_BackendGTK):
    FigureCanvas: Type[FigureCanvasGTK3] = FigureCanvasGTK3
    FigureManager: Type[FigureManagerGTK3] = FigureManagerGTK3
