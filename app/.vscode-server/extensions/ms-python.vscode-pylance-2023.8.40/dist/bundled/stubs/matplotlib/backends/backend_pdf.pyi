import numpy as np
from io import BytesIO
from enum import Enum
from functools import total_ordering
from typing import Any, Callable, Optional, Set

from matplotlib._typing import *
from matplotlib.text import Text
from matplotlib._enums import CapStyle, JoinStyle
from matplotlib.font_manager import FontProperties
from matplotlib.figure import Figure
from matplotlib.transforms import Affine2DBase, Transform
from matplotlib.backend_bases import (
    FigureCanvasBase,
    FigureManagerBase,
    GraphicsContextBase,
    _Backend,
)
from . import _backend_pdf_ps

def fill(strings, linelen: int=75)-> bytes: ...
def pdfRepr(obj) -> bytes: ...

class Reference:
    def __init__(self, id: int) -> None: ...
    def __repr__(self)-> str: ...
    def pdfRepr(self) -> bytes: ...
    def write(self, contentsy, file: "PdfFile") -> None: ...

@total_ordering
class Name:
    def __init__(self, name: bytes | Name | str) -> None: ...
    def __repr__(self)-> str: ...
    def __str__(self) -> str: ...
    def __eq__(self, other) -> bool: ...
    def __lt__(self, other) -> bool: ...
    def __hash__(self) -> int: ...
    @staticmethod
    def hexify(match)-> str: ...
    def pdfRepr(self) -> bytes: ...

class Operator:
    def __init__(self, op) -> None: ...
    def __repr__(self)-> str: ...
    def pdfRepr(self)-> bytes: ...

class Verbatim:
    def __init__(self, x: bytes) -> None: ...
    def pdfRepr(self) -> bytes: ...

class Op(Enum):
    close_fill_stroke: Op
    fill_stroke: Op
    fill: Op
    closepath: Op
    close_stroke: Op
    stroke: Op
    endpath: Op
    begin_text: Op
    end_text: Op
    curveto: Op
    rectangle: Op
    lineto: Op
    moveto: Op
    concat_matrix: Op
    use_xobject: Op
    setgray_stroke: Op
    setgray_nonstroke: Op
    setrgb_stroke: Op
    setrgb_nonstroke: Op
    setcolorspace_stroke: Op
    setcolorspace_nonstroke: Op
    setcolor_stroke: Op
    setcolor_nonstroke: Op
    setdash: Op
    setlinejoin: Op
    setlinecap: Op
    setgstate: Op
    gsave: Op
    grestore: Op
    textpos: Op
    selectfont: Op
    textmatrix : Op
    show: Op
    showkern: Op
    setlinewidth: Op
    clip: Op
    shading: Op
    op: Op
    def pdfRepr(self) -> bytes: ...
    @classmethod
    def paint_path(cls, fill: bool, stroke: bool) -> Op: ...

class Stream:
    def __init__(
        self,
        id: int,
        len: Reference | None,
        file: PdfFile,
        extra: None | dict[str, Any] = ...,
        png: None | dict[str, int] = ...,
    ) -> None: ...
    def end(self) -> None: ...
    def write(self, data: bytes) -> None: ...

class PdfFile:
    def __init__(
        self, filename: BytesIO | str | PathLike| FileLike, metadata: dict[str, Any] = ...
    ) -> None: ...
    def newPage(self, width, height)-> None: ...
    def newTextnote(self, text: str, positionRect: list[int] = ...) -> None: ...
    def finalize(self) -> None: ...
    def close(self) -> None: ...
    def write(self, data: bytes) -> None: ...
    def output(self, *data) -> None: ...
    def beginStream(
        self,
        id: int,
        len: None | Reference,
        extra: None
        | dict[str, int | Name]
        | dict[str, Name | list[int]]
        | dict[str, Name | int | Verbatim]
        | dict[str, Name | int | Verbatim | Reference] = ...,
        png: Optional[dict[str, int]] = ...,
    ) -> None: ...
    def endStream(self) -> None: ...
    def outputStream(self, ref: Reference, data: bytes, *, extra=...) -> None: ...
    def fontName(self, fontprop: FontProperties | str) -> Name: ...
    def dviFontName(self, dvifont)-> Name: ...
    def writeFonts(self) -> None: ...
    def createType1Descriptor(self, t1font, fontfile)-> Reference: ...
    def embedTTF(self, filename: str, characters: Set[int]) -> Reference: ...
    def alphaState(self, alpha: tuple[float, float]) -> Name: ...
    def writeExtGSTates(self) -> None: ...
    def hatchPattern(self, hatch_style) -> Name: ...
    def writeHatches(self) -> None: ...
    def addGouraudTriangles(
        self, points: np.ndarray, colors: np.ndarray
    ) -> tuple[Name, Reference]: ...
    def writeGouraudTriangles(self) -> None: ...
    def imageObject(self, image) -> Name: ...
    def writeImages(self) -> None: ...
    def markerObject(
        self, path, trans, fill, stroke, lw, joinstyle, capstyle
    ) -> Name: ...
    def writeMarkers(self) -> None: ...
    def pathCollectionObject(self, gc, path, trans, padding, filled, stroked)-> Name: ...
    def writePathCollectionTemplates(self) -> None: ...
    @staticmethod
    def pathOperations(path, transform, clip=..., simplify=..., sketch=...)-> list[Verbatim]: ...
    def writePath(self, path, transform, clip=..., sketch=...)-> None: ...
    def reserveObject(self, name: str = ...) -> Reference: ...
    def recordXref(self, id: int) -> None: ...
    def writeObject(self, object: Reference, contents: Any) -> None: ...
    def writeXref(self) -> None: ...
    def writeInfoDict(self) -> None: ...
    def writeTrailer(self) -> None: ...

class RendererPdf(_backend_pdf_ps.RendererPDFPSBase):
    def __init__(self, file, image_dpi, height, width) -> None: ...
    def finalize(self) -> None: ...
    def check_gc(
        self,
        gc: "GraphicsContextPdf",
        fillcolor: None | tuple[float, float, float, float] = ...,
    ) -> None: ...
    def get_image_magnification(self)-> float: ...
    def draw_image(
        self,
        gc: GraphicsContextBase,
        x: Scalar,
        y: Scalar,
        im,
        transform: Affine2DBase = ...,
    )-> None: ...
    def draw_path(self, gc, path, transform, rgbFace=...)-> None: ...
    def draw_path_collection(
        self,
        gc,
        master_transform,
        paths,
        all_transforms,
        offsets,
        offsetTrans,
        facecolors,
        edgecolors,
        linewidths,
        linestyles,
        antialiaseds,
        urls,
        offset_position,
    )-> None: ...
    def draw_markers(
        self,
        gc: GraphicsContextBase,
        marker_path,
        marker_trans: Transform,
        path,
        trans: Transform,
        rgbFace=...,
    )-> None: ...
    def draw_gouraud_triangle(self, gc: GraphicsContextBase, points, colors, trans)-> None: ...
    def draw_gouraud_triangles(self, gc, points, colors, trans)-> None: ...
    def draw_mathtext(self, gc, x, y, s, prop, angle)-> None: ...
    def draw_tex(self, gc, x, y, s, prop, angle, *, mtext=...)-> None: ...
    def encode_string(self, s: str, fonttype: int) -> bytes: ...
    def draw_text(
        self,
        gc: GraphicsContextBase,
        x: float,
        y: float,
        s: str,
        prop: FontProperties,
        angle: float,
        ismath=...,
        mtext: Text = ...,
    )-> None: ...
    def new_gc(self) -> GraphicsContextPdf: ...

class GraphicsContextPdf(GraphicsContextBase):
    def __init__(self, file: PdfFile) -> None: ...
    def __repr__(self)->str: ...
    def stroke(self) -> bool: ...
    def fill(self, *args) -> bool: ...
    def paint(
        self,
    ) -> Op: ...
    capstyles: dict[str, int] = ...
    joinstyles: dict[str, int] = ...
    def capstyle_cmd(self, style: CapStyle) -> list[int | Op]: ...
    def joinstyle_cmd(self, style: JoinStyle) -> list[int | Op]: ...
    def linewidth_cmd(self, width: int | float) -> list[float | Op]: ...
    def dash_cmd(self, dashes)-> list: ...
    def alpha_cmd(
        self, alpha: float, forced: bool, effective_alphas: tuple[float, float]
    ) -> list[Name | Op]: ...
    def hatch_cmd(
        self, hatch: None, hatch_color: tuple[float, float, float, float]
    ) -> list[Name | Op | float]: ...
    def rgb_cmd(
        self,
        rgb: tuple[float, float, float, float],
    ) -> list[float | Op]: ...
    def fillcolor_cmd(
        self,
        rgb: None | tuple[float, float, float, float],
    ) -> list[float | Op]: ...
    def push(self) -> list[Op]: ...
    def pop(self) -> list[Op]: ...
    def clip_cmd(self, cliprect, clippath) -> list[Op]: ...
    commands: tuple = ...
    def delta(self, other: "GraphicsContextPdf") -> list[Op | Name | float]: ...
    def copy_properties(self, other: GraphicsContextPdf) -> None: ...
    def finalize(self) -> list[Op]: ...

class PdfPages:
    def __init__(
        self, filename: str, keep_empty: bool = ..., metadata: dict[str, Any] = ...
    ) -> None: ...
    def __enter__(self) -> PdfPages: ...
    def __exit__(self, exc_type, exc_val, exc_tb)-> None: ...
    def close(self) -> None: ...
    def infodict(self) -> dict[str, Any]: ...
    def savefig(self, figure: Figure | int = ..., **kwargs) -> None: ...
    def get_pagecount(self) -> int: ...
    def attach_note(self, text: str, positionRect: list[int] = ...) -> None: ...

class FigureCanvasPdf(FigureCanvasBase):
    fixed_dpi: float= ...
    filetypes: dict[str, str] = ...
    def get_default_filetype(self)-> str: ...
    def print_pdf(
        self,
        filename: BytesIO | PdfPages | str,
        *,
        bbox_inches_restore=...,
        metadata=...
    ) -> None: ...
    def draw(self)-> None: ...

FigureManagerPdf = FigureManagerBase

class _BackendPdf(_Backend):
    FigureCanvas = FigureCanvasPdf
