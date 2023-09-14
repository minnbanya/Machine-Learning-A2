import sys
from typing_extensions import Final

if sys.platform == "win32":
    from . import win32

    class WinColor:
        BLACK: Final = 0
        BLUE: Final = 1
        GREEN: Final = 2
        CYAN: Final = 3
        RED: Final = 4
        MAGENTA: Final = 5
        YELLOW: Final = 6
        GREY: Final = 7

    class WinStyle:
        NORMAL: Final = 0x00
        BRIGHT: Final = 0x08
        BRIGHT_BACKGROUND: Final = 0x80

    class WinTerm:
        def __init__(self) -> None: ...
        def get_attrs(self) -> int: ...
        def set_attrs(self, value: int) -> None: ...
        def reset_all(self, on_stderr: bool | None = None) -> None: ...
        def fore(self, fore: int | None = None, light: bool = False, on_stderr: bool = False) -> None: ...
        def back(self, back: int | None = None, light: bool = False, on_stderr: bool = False) -> None: ...
        def style(self, style: int | None = None, on_stderr: bool = False) -> None: ...
        def set_console(self, attrs: int | None = None, on_stderr: bool = False) -> None: ...
        def get_position(self, handle: int) -> win32.COORD: ...
        def set_cursor_position(self, position: win32.COORD | None = None, on_stderr: bool = False) -> None: ...
        def cursor_adjust(self, x: int, y: int, on_stderr: bool = False) -> None: ...
        def erase_screen(self, mode: int = 0, on_stderr: bool = False) -> None: ...
        def erase_line(self, mode: int = 0, on_stderr: bool = False) -> None: ...
        def set_title(self, title: str) -> None: ...

def enable_vt_processing(fd: int) -> bool: ...
