from _typeshed import Incomplete
from io import BytesIO, StringIO

from .exceptions import EOF, TIMEOUT
from .spawnbase import SpawnBase

class searcher_string:
    eof_index: int
    timeout_index: int
    longest_string: int
    def __init__(self, strings) -> None: ...
    match: Incomplete
    start: Incomplete
    end: Incomplete
    def search(self, buffer, freshlen, searchwindowsize: Incomplete | None = None): ...

class searcher_re:
    eof_index: int
    timeout_index: int
    def __init__(self, patterns) -> None: ...
    start: Incomplete
    match: Incomplete
    end: Incomplete
    def search(self, buffer, freshlen: int, searchwindowsize: int | None = None): ...

class Expecter:
    spawn: BytesIO | StringIO
    searcher: searcher_re | searcher_string
    searchwindowsize: int | None
    lookback: searcher_string | searcher_re | int | None
    def __init__(self, spawn: SpawnBase, searcher: searcher_re | searcher_string, searchwindowsize: int = -1) -> None: ...
    def do_search(self, window: str, freshlen: int): ...
    def existing_data(self): ...
    def new_data(self, data: Incomplete): ...
    def eof(self, err: Incomplete | None = None) -> int | EOF: ...
    def timeout(self, err: object | None = None) -> int | TIMEOUT: ...
    def errored(self) -> None: ...
    def expect_loop(self, timeout: int = -1): ...
