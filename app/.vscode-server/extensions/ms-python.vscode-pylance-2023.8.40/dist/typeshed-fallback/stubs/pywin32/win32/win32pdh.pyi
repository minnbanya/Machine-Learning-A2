from _typeshed import Incomplete

from win32.lib.pywintypes import error as error

def AddCounter(hQuery, path: str, userData: int = ...): ...
def AddEnglishCounter(hQuery, path: str, userData: int = ...): ...
def RemoveCounter(handle) -> None: ...
def EnumObjectItems(DataSource: str | None, machine: str | None, _object: str, detailLevel, flags=...): ...
def EnumObjects(DataSource: str | None, machine: str | None, detailLevel: int, refresh: bool = ...): ...
def OpenQuery(DataSource: Incomplete | None = ..., userData: int = ...): ...
def CloseQuery(handle) -> None: ...
def MakeCounterPath(
    elements: tuple[Incomplete, Incomplete, Incomplete, Incomplete, Incomplete, Incomplete], flags=...
) -> None: ...
def GetCounterInfo(handle, bRetrieveExplainText) -> None: ...
def GetFormattedCounterValue(handle, _format) -> tuple[Incomplete, Incomplete]: ...
def CollectQueryData(hQuery) -> None: ...
def ValidatePath(path: str): ...
def ExpandCounterPath(wildCardPath: str) -> tuple[Incomplete, Incomplete]: ...
def ParseCounterPath(path: str, flags=...) -> tuple[Incomplete, Incomplete, Incomplete, Incomplete, Incomplete, Incomplete]: ...
def ParseInstanceName(instanceName: str) -> tuple[Incomplete, Incomplete, Incomplete]: ...
def SetCounterScaleFactor(hCounter, factor) -> None: ...
def BrowseCounters(
    Flags: tuple[Incomplete, ...] | None,
    hWndOwner: int,
    CallBack1,
    CallBack2,
    DialogBoxCaption: str | None = ...,
    InitialPath: Incomplete | None = ...,
    DataSource: Incomplete | None = ...,
    ReturnMultiple: bool = ...,
    CallBackArg: Incomplete | None = ...,
) -> str: ...
def ConnectMachine(machineName: str) -> str: ...
def LookupPerfIndexByName(machineName: str, instanceName: str): ...
def LookupPerfNameByIndex(machineName: str | None, index) -> str: ...
def GetFormattedCounterArray(*args, **kwargs): ...  # incomplete

PDH_FMT_1000: int
PDH_FMT_ANSI: int
PDH_FMT_DOUBLE: int
PDH_FMT_LARGE: int
PDH_FMT_LONG: int
PDH_FMT_NODATA: int
PDH_FMT_NOSCALE: int
PDH_FMT_RAW: int
PDH_FMT_UNICODE: int
PDH_MAX_SCALE: int
PDH_MIN_SCALE: int
PDH_PATH_WBEM_INPUT: int
PDH_PATH_WBEM_RESULT: int
PDH_VERSION: int
PERF_DETAIL_ADVANCED: int
PERF_DETAIL_EXPERT: int
PERF_DETAIL_NOVICE: int
PERF_DETAIL_WIZARD: int

class counter_status_error(Exception): ...

PDH_FMT_NOCAP100: int
