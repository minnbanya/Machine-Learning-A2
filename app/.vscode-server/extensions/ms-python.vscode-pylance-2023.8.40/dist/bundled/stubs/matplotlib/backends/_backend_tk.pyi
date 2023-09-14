import tkinter as tk
from typing import Literal, Sequence
from matplotlib._api import classproperty
from matplotlib._typing import *
from matplotlib import backend_tools
from matplotlib.backend_bases import (
    FigureCanvasBase,
    FigureManagerBase,
    NavigationToolbar2,
    TimerBase,
    ToolContainerBase,
    _Backend,
)

backend_version: float = ...
cursord: dict[backend_tools.Cursors, str] = ...

TK_PHOTO_COMPOSITE_OVERLAY: Literal[0] = ...
TK_PHOTO_COMPOSITE_SET: Literal[1] = ...

def blit(photoimage, aggimage, offsets: Sequence[int], bbox=...)-> None: ...

class TimerTk(TimerBase):
    def __init__(self, parent, *args, **kwargs) -> None: ...

class FigureCanvasTk(FigureCanvasBase):
    required_interactive_framework: str = ...
    manager_class: classproperty = ...

    def __init__(self, figure=..., master=...) -> None: ...
    def resize(self, event)-> None: ...
    def draw_idle(self)-> None: ...
    def get_tk_widget(self)-> tk.Canvas: ...
    def motion_notify_event(self, event)-> None: ...
    def enter_notify_event(self, event)-> None: ...
    def button_press_event(self, event, dblclick=...)-> None: ...
    def button_dblclick_event(self, event)-> None: ...
    def button_release_event(self, event)-> None: ...
    def scroll_event(self, event) -> None: ...
    def scroll_event_windows(self, event)-> None: ...
    def key_press(self, event)-> None: ...
    def key_release(self, event)-> None: ...
    def new_timer(self, *args, **kwargs)-> TimerTk: ...
    def flush_events(self)-> None: ...
    def start_event_loop(self, timeout: int=0)-> None: ...
    def stop_event_loop(self)-> None: ...
    def set_cursor(self, cursor)-> None: ...

class FigureManagerTk(FigureManagerBase):

    canvas: FigureCanvasBase
    num: int | str
    toolbar: tk.Toolbar
    window: tk.Window

    def __init__(self, canvas, num, window) -> None: ...
    @classmethod
    def create_with_canvas(cls, canvas_class, figure, num)-> FigureManagerTk: ...
    def resize(self, width: float, height: float) -> None: ...
    def show(self)-> None: ...
    def destroy(self, *args)-> None: ...
    def get_window_title(self): ...
    def set_window_title(self, title)-> None: ...
    def full_screen_toggle(self)-> None: ...

class NavigationToolbar2Tk(NavigationToolbar2, tk.Frame):
    window: tk.Window = ...
    def __init__(
        self, canvas: FigureCanvasBase, window=..., *, pack_toolbar: bool = True
    ) -> None: ...
    def pan(self, *args)-> None: ...
    def zoom(self, *args)-> None: ...
    def set_message(self, s: str)-> None: ...
    def draw_rubberband(self, event, x0: float, y0: float, x1: float, y1: float)-> None: ...
    def remove_rubberband(self)-> None: ...

    def save_figure(self, *args)-> None: ...
    def set_history_buttons(self)-> None: ...

class ToolTip:
    @staticmethod
    def createToolTip(widget, text)-> None: ...
    def __init__(self, widget) -> None: ...
    def showtip(self, text)-> None: ...
    def hidetip(self)-> None: ...

class RubberbandTk(backend_tools.RubberbandBase):
    def draw_rubberband(self, x0, y0, x1, y1)-> None: ...
    def remove_rubberband(self)-> None: ...

class SetCursorTk(backend_tools.SetCursorBase):
    def set_cursor(self, cursor)-> None: ...

class ToolbarTk(ToolContainerBase, tk.Frame):
    def __init__(self, toolmanager, window=...) -> None: ...
    def add_toolitem(self, name, group, position, image_file, description, toggle)-> None: ...
    def toggle_toolitem(self, name, toggled)-> None: ...
    def remove_toolitem(self, name)-> None: ...
    def set_message(self, s)-> None: ...

class SaveFigureTk(backend_tools.SaveFigureBase):
    def trigger(self, *args)-> None: ...

class ConfigureSubplotsTk(backend_tools.ConfigureSubplotsBase):
    def trigger(self, *args)-> None: ...

class HelpTk(backend_tools.ToolHelpBase):
    def trigger(self, *args)-> None: ...

Toolbar = ToolbarTk

class _BackendTk(_Backend):
    FigureManager = FigureManagerTk
    @staticmethod
    def mainloop()-> None: ...
