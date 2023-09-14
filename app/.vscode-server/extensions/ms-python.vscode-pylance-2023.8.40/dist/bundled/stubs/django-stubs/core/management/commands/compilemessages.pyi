import os
from typing import Any, List, Tuple, Union

from django.core.management.base import BaseCommand as BaseCommand
from django.core.management.base import CommandError as CommandError
from django.core.management.base import CommandParser as CommandParser
from django.core.management.utils import find_command as find_command
from django.core.management.utils import popen_wrapper as popen_wrapper

_PathType = Union[str, bytes, os.PathLike[Any]]

def has_bom(fn: _PathType) -> bool: ...
def is_writable(path: _PathType) -> bool: ...

class Command(BaseCommand):
    program: str = ...
    program_options: List[str] = ...
    verbosity: int = ...
    has_errors: bool = ...
    def compile_messages(
        self, locations: List[Tuple[_PathType, _PathType]]
    ) -> None: ...
