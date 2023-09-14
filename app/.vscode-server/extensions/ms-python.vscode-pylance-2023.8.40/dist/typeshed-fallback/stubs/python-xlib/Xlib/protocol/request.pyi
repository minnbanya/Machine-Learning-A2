from typing import NoReturn
from typing_extensions import Final

from Xlib import display
from Xlib.protocol import rq

class CreateWindow(rq.Request): ...
class ChangeWindowAttributes(rq.Request): ...
class GetWindowAttributes(rq.ReplyRequest): ...
class DestroyWindow(rq.Request): ...
class DestroySubWindows(rq.Request): ...
class ChangeSaveSet(rq.Request): ...
class ReparentWindow(rq.Request): ...
class MapWindow(rq.Request): ...
class MapSubwindows(rq.Request): ...
class UnmapWindow(rq.Request): ...
class UnmapSubwindows(rq.Request): ...
class ConfigureWindow(rq.Request): ...
class CirculateWindow(rq.Request): ...
class GetGeometry(rq.ReplyRequest): ...
class QueryTree(rq.ReplyRequest): ...
class InternAtom(rq.ReplyRequest): ...
class GetAtomName(rq.ReplyRequest): ...
class ChangeProperty(rq.Request): ...
class DeleteProperty(rq.Request): ...
class GetProperty(rq.ReplyRequest): ...
class ListProperties(rq.ReplyRequest): ...
class SetSelectionOwner(rq.Request): ...
class GetSelectionOwner(rq.ReplyRequest): ...
class ConvertSelection(rq.Request): ...
class SendEvent(rq.Request): ...
class GrabPointer(rq.ReplyRequest): ...
class UngrabPointer(rq.Request): ...
class GrabButton(rq.Request): ...
class UngrabButton(rq.Request): ...
class ChangeActivePointerGrab(rq.Request): ...
class GrabKeyboard(rq.ReplyRequest): ...
class UngrabKeyboard(rq.Request): ...
class GrabKey(rq.Request): ...
class UngrabKey(rq.Request): ...
class AllowEvents(rq.Request): ...
class GrabServer(rq.Request): ...
class UngrabServer(rq.Request): ...
class QueryPointer(rq.ReplyRequest): ...
class GetMotionEvents(rq.ReplyRequest): ...
class TranslateCoords(rq.ReplyRequest): ...
class WarpPointer(rq.Request): ...
class SetInputFocus(rq.Request): ...
class GetInputFocus(rq.ReplyRequest): ...
class QueryKeymap(rq.ReplyRequest): ...
class OpenFont(rq.Request): ...
class CloseFont(rq.Request): ...
class QueryFont(rq.ReplyRequest): ...
class QueryTextExtents(rq.ReplyRequest): ...
class ListFonts(rq.ReplyRequest): ...

class ListFontsWithInfo(rq.ReplyRequest):
    def __init__(self, display: display.Display, defer: bool = False, *args: object, **keys: object) -> None: ...
    def __getattr__(self, attr: object) -> NoReturn: ...
    def __getitem__(self, item: str) -> object: ...
    def __len__(self) -> int: ...

class SetFontPath(rq.Request): ...
class GetFontPath(rq.ReplyRequest): ...
class CreatePixmap(rq.Request): ...
class FreePixmap(rq.Request): ...
class CreateGC(rq.Request): ...
class ChangeGC(rq.Request): ...
class CopyGC(rq.Request): ...
class SetDashes(rq.Request): ...
class SetClipRectangles(rq.Request): ...
class FreeGC(rq.Request): ...
class ClearArea(rq.Request): ...
class CopyArea(rq.Request): ...
class CopyPlane(rq.Request): ...
class PolyPoint(rq.Request): ...
class PolyLine(rq.Request): ...
class PolySegment(rq.Request): ...
class PolyRectangle(rq.Request): ...
class PolyArc(rq.Request): ...
class FillPoly(rq.Request): ...
class PolyFillRectangle(rq.Request): ...
class PolyFillArc(rq.Request): ...
class PutImage(rq.Request): ...
class GetImage(rq.ReplyRequest): ...
class PolyText8(rq.Request): ...
class PolyText16(rq.Request): ...
class ImageText8(rq.Request): ...
class ImageText16(rq.Request): ...
class CreateColormap(rq.Request): ...
class FreeColormap(rq.Request): ...
class CopyColormapAndFree(rq.Request): ...
class InstallColormap(rq.Request): ...
class UninstallColormap(rq.Request): ...
class ListInstalledColormaps(rq.ReplyRequest): ...
class AllocColor(rq.ReplyRequest): ...
class AllocNamedColor(rq.ReplyRequest): ...
class AllocColorCells(rq.ReplyRequest): ...
class AllocColorPlanes(rq.ReplyRequest): ...
class FreeColors(rq.Request): ...
class StoreColors(rq.Request): ...
class StoreNamedColor(rq.Request): ...
class QueryColors(rq.ReplyRequest): ...
class LookupColor(rq.ReplyRequest): ...
class CreateCursor(rq.Request): ...
class CreateGlyphCursor(rq.Request): ...
class FreeCursor(rq.Request): ...
class RecolorCursor(rq.Request): ...
class QueryBestSize(rq.ReplyRequest): ...
class QueryExtension(rq.ReplyRequest): ...
class ListExtensions(rq.ReplyRequest): ...
class ChangeKeyboardMapping(rq.Request): ...
class GetKeyboardMapping(rq.ReplyRequest): ...
class ChangeKeyboardControl(rq.Request): ...
class GetKeyboardControl(rq.ReplyRequest): ...
class Bell(rq.Request): ...
class ChangePointerControl(rq.Request): ...
class GetPointerControl(rq.ReplyRequest): ...
class SetScreenSaver(rq.Request): ...
class GetScreenSaver(rq.ReplyRequest): ...
class ChangeHosts(rq.Request): ...
class ListHosts(rq.ReplyRequest): ...
class SetAccessControl(rq.Request): ...
class SetCloseDownMode(rq.Request): ...
class KillClient(rq.Request): ...
class RotateProperties(rq.Request): ...
class ForceScreenSaver(rq.Request): ...
class SetPointerMapping(rq.ReplyRequest): ...
class GetPointerMapping(rq.ReplyRequest): ...
class SetModifierMapping(rq.ReplyRequest): ...
class GetModifierMapping(rq.ReplyRequest): ...
class NoOperation(rq.Request): ...

major_codes: Final[dict[int, type[rq.Request]]]
