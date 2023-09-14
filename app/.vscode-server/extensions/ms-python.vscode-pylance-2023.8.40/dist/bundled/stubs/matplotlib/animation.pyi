from typing import Callable, Generator, Iterable, Iterator

from matplotlib.backend_bases import NonGuiException
from .figure import Figure

from itertools import count
from typing import Callable, List

import abc
import contextlib

subprocess_creation_flags = ...

def adjusted_figsize(w: float, h: float, dpi: float, n: int) -> tuple[float, float]: ...

class MovieWriterRegistry:
    def __init__(self) -> None: ...
    def register(self, name: str) -> Callable: ...
    def is_available(self, name: str) -> bool: ...
    def __iter__(self)-> Generator: ...
    def list(self) -> list[MovieWriter]: ...
    def __getitem__(self, name) -> MovieWriter: ...

writers: MovieWriterRegistry = ...

class AbstractMovieWriter(abc.ABC):
    def __init__(
        self,
        fps: float = 5,
        metadata: dict[str, str]|None = None,
        codec=None,
        bitrate=None,
    ) -> None: ...
    @abc.abstractmethod
    def setup(self, fig: Figure, outfile: str, dpi: float|None = None)-> None: ...
    @property
    def frame_size(self) -> tuple[int, int]: ...
    @abc.abstractmethod
    def grab_frame(self, **savefig_kwargs)-> None: ...
    @abc.abstractmethod
    def finish(self)-> None: ...
    @contextlib.contextmanager
    def saving(self, fig: Figure, outfile, dpi: float, *args, **kwargs)->Generator[AbstractMovieWriter, None, None]: ...

class MovieWriter(AbstractMovieWriter):

    frame_format: str
    fig: Figure

    supported_formats = ...
    def __init__(
        self,
        fps: int = 5,
        codec: str | None = None,
        bitrate: int|None = None,
        extra_args: list[str] | None = None,
        metadata: dict[str, str]|None = None,
    ) -> None: ...
    def setup(self, fig: Figure, outfile: str, dpi: float|None = None)-> None: ...
    def finish(self)-> None: ...
    def grab_frame(self, **savefig_kwargs)-> None: ...
    @classmethod
    def bin_path(cls) -> str: ...
    @classmethod
    def isAvailable(cls) -> bool: ...

class FileMovieWriter(MovieWriter):
    def __init__(self, *args, **kwargs) -> None: ...
    def setup(
        self, fig: Figure, outfile: str, dpi: float|None = None, frame_prefix: str|None = None
    ): ...
    def __del__(self)-> None: ...
    @property
    def frame_format(self)-> str: ...
    @frame_format.setter
    def frame_format(self, frame_format: str)-> None: ...
    def grab_frame(self, **savefig_kwargs)-> None: ...
    def finish(self)-> None: ...

class PillowWriter(AbstractMovieWriter):
    @classmethod
    def isAvailable(cls)-> bool: ...
    def setup(self, fig: Figure, outfile: str, dpi: float|None = None)-> None: ...
    def grab_frame(self, **savefig_kwargs)-> None: ...
    def finish(self)-> None: ...

class FFMpegBase:
    @property
    def output_args(self) -> list[str]: ...

class FFMpegWriter(FFMpegBase, MovieWriter): ...

class FFMpegFileWriter(FFMpegBase, FileMovieWriter):

    supported_formats: list[str] = ...

class ImageMagickBase:
    @classmethod
    def bin_path(cls)-> str: ...
    @classmethod
    def isAvailable(cls)-> bool: ...

class ImageMagickWriter(ImageMagickBase, MovieWriter):

    input_names: str = ...

class ImageMagickFileWriter(ImageMagickBase, FileMovieWriter):

    supported_formats: list[str] = ...
    input_names: str = ...

class HTMLWriter(FileMovieWriter):

    supported_formats: list[str] = ...
    @classmethod
    def isAvailable(cls): ...
    def __init__(
        self,
        fps: float=30,
        codec=None,
        bitrate: float|None=None,
        extra_args=None,
        metadata=None,
        embed_frames: bool=False,
        default_mode: str='loop',
        embed_limit=None,
    ) -> None: ...
    def setup(self, fig: Figure, outfile: str, dpi: float, frame_dir=None)-> None: ...
    def grab_frame(self, **savefig_kwargs)-> None: ...
    def finish(self)-> None: ...

class Animation:
    def __init__(
        self, fig: Figure, event_source: object = None, blit: bool = False
    ) -> None: ...
    def __del__(self)-> None: ...
    def save(
        self,
        filename: str,
        writer: MovieWriter | str|None = None,
        fps: int|None = None,
        dpi: float|None = None,
        codec: str|None = None,
        bitrate: int|None = None,
        extra_args: list[str] | None = None,
        metadata: dict[str, str]|None = None,
        extra_anim: list|None = None,
        savefig_kwargs: dict|None = None,
        *,
        progress_callback: Callable|None = None
    )-> None: ...
    def new_frame_seq(self) -> Iterator: ...
    def new_saved_frame_seq(self)-> Iterable: ...
    def to_html5_video(self, embed_limit: float|None = None) -> str: ...
    def to_jshtml(
        self, fps: int|None = None, embed_frames: bool = True, default_mode: str|None = None
    ): ...
    def pause(self)-> None: ...
    def resume(self)-> None: ...

class TimedAnimation(Animation):
    def __init__(
        self,
        fig: Figure,
        interval: int = 200,
        repeat_delay: int = 0,
        repeat: bool = True,
        event_source=None,
        *args,
        **kwargs
    ) -> None: ...

class ArtistAnimation(TimedAnimation):
    def __init__(self, fig: Figure, artists: list, *args, **kwargs) -> None: ...

class FuncAnimation(TimedAnimation):
    def __init__(
        self,
        fig: Figure,
        func: Callable,
        frames: Callable | Iterable | int | None = None,
        init_func: Callable|None = None,
        fargs: tuple | None =None,
        save_count: int|None = None,
        *,
        cache_frame_data: bool = True,
        **kwargs
    ) -> None: ...
    def new_frame_seq(self) -> count: ...
    def new_saved_frame_seq(self)-> Iterator: ...
