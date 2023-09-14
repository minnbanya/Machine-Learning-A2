from _typeshed import Incomplete
from typing import ClassVar
from typing_extensions import Literal

from .ImageFile import ImageFile

def isInt(f: object) -> Literal[0, 1]: ...

iforms: Incomplete

def isSpiderHeader(t): ...
def isSpiderImage(filename): ...

class SpiderImageFile(ImageFile):
    format: ClassVar[Literal["SPIDER"]]
    format_description: ClassVar[str]
    @property
    def n_frames(self): ...
    @property
    def is_animated(self): ...
    def tell(self): ...
    stkoffset: Incomplete
    fp: Incomplete
    def seek(self, frame) -> None: ...
    def convert2byte(self, depth: int = 255): ...
    def tkPhotoImage(self): ...

def loadImageSeries(filelist: Incomplete | None = None): ...
def makeSpiderHeader(im): ...
