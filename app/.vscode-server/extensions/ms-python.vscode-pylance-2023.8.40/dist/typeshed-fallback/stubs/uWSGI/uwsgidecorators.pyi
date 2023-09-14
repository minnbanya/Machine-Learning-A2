from collections.abc import Callable
from typing import Any, Generic, TypeVar, overload
from typing_extensions import Literal, ParamSpec

from uwsgi import _RPCCallable

_T = TypeVar("_T")
_T2 = TypeVar("_T2")
_SR = TypeVar("_SR", bound=Literal[0, -1, -2] | None)
_SignalCallbackT = TypeVar("_SignalCallbackT", bound=Callable[[int], Any])
_RPCCallableT = TypeVar("_RPCCallableT", bound=_RPCCallable)
_P = ParamSpec("_P")
_P2 = ParamSpec("_P2")

spooler_functions: dict[str, Callable[..., Literal[0, -1, -2] | None]]
mule_functions: dict[str, Callable[..., Any]]
postfork_chain: list[Callable[[], None]]

def get_free_signal() -> int: ...
def manage_spool_request(vars: dict[bytes, Any]) -> Literal[0, -1, -2]: ...
def postfork_chain_hook() -> None: ...

class postfork(Generic[_P, _T]):
    wid: int
    f: Callable[_P, _T] | None
    @overload
    def __init__(self: postfork[..., Any], f: int) -> None: ...
    @overload
    def __init__(self: postfork[_P, _T], f: Callable[_P, _T]) -> None: ...
    @overload
    def __call__(self, __f: Callable[_P2, _T2]) -> postfork[_P2, _T2]: ...
    @overload
    def __call__(self, *args: _P.args, **kwargs: _P.kwargs) -> _T: ...

class _spoolraw(Generic[_P, _SR]):
    f: Callable[_P, _SR]
    pass_arguments: bool
    base_dict: dict[str, Any]
    def __init__(self, f: Callable[_P, _SR], pass_arguments: bool) -> None: ...
    def __call__(self, *args: _P.args, **kwargs: _P.kwargs) -> _SR: ...
    def spool(self, *args: _P.args, **kwargs: _P.kwargs) -> _SR: ...

class _spool(_spoolraw[_P, _SR]): ...
class _spoolforever(_spoolraw[_P, _SR]): ...

@overload
def spool_decorate(
    f: Callable[_P, _SR], pass_arguments: bool = False, _class: type[_spoolraw[_P, _SR]] = ...
) -> _spoolraw[_P, _SR]: ...
@overload
def spool_decorate(
    f: None = None, pass_arguments: bool = False, _class: type[_spoolraw[..., Any]] = ...
) -> Callable[[Callable[_P, _SR]], _spoolraw[_P, _SR]]: ...
@overload
def spoolraw(f: Callable[_P, _SR], pass_arguments: bool = False) -> _spoolraw[_P, _SR]: ...
@overload
def spoolraw(f: None = None, pass_arguments: bool = False) -> Callable[[Callable[_P, _SR]], _spoolraw[_P, _SR]]: ...
@overload
def spool(f: Callable[_P, _SR], pass_arguments: bool = False) -> _spool[_P, _SR]: ...
@overload
def spool(f: None = None, pass_arguments: bool = False) -> Callable[[Callable[_P, _SR]], _spool[_P, _SR]]: ...
@overload
def spoolforever(f: Callable[_P, _SR], pass_arguments: bool = False) -> _spoolforever[_P, _SR]: ...
@overload
def spoolforever(f: None = None, pass_arguments: bool = False) -> Callable[[Callable[_P, _SR]], _spoolforever[_P, _SR]]: ...

class mulefunc(Generic[_P, _T]):
    fname: str | None
    mule: int
    @overload
    def __init__(self: mulefunc[..., Any], f: int) -> None: ...
    @overload
    def __init__(self: mulefunc[_P, _T], f: Callable[_P, _T]) -> None: ...
    def real_call(self, *args: _P.args, **kwargs: _P.kwargs) -> None: ...
    @overload
    def __call__(self, __f: Callable[_P2, _T2]) -> mulefunc[_P2, _T2]: ...
    @overload
    def __call__(self, *args: _P.args, **kwargs: _P.kwargs) -> _T: ...

def mule_msg_dispatcher(message: bytes) -> Any: ...

class rpc:
    name: str
    def __init__(self, name: str) -> None: ...
    def __call__(self, f: _RPCCallableT) -> _RPCCallableT: ...

class farm_loop:
    f: Callable[[bytes], Any]
    farm: str | None
    def __init__(self, f: Callable[[bytes], Any], farm: str | None) -> None: ...
    def __call__(self) -> None: ...

class farm:
    name: str
    def __init__(self, name: str | None = None, **kwargs: Any) -> None: ...
    def __call__(self, f: Callable[[bytes], Any]) -> None: ...

class mule_brain:
    f: Callable[[], Any]
    num: int
    def __init__(self, f: Callable[[], Any], num: int) -> None: ...
    def __call__(self) -> None: ...

class mule_brainloop(mule_brain): ...

class mule:
    num: int
    def __init__(self, num: int) -> None: ...
    def __call__(self, f: Callable[[], Any]) -> None: ...

class muleloop(mule): ...

class mulemsg_loop:
    f: Callable[[bytes], Any]
    num: int
    def __init__(self, f: Callable[[bytes], Any], num: int) -> None: ...
    def __call__(self) -> None: ...

class mulemsg:
    num: int
    def __init__(self, num: int) -> None: ...
    def __call__(self, f: Callable[[bytes], Any]) -> None: ...

class signal:
    num: int
    target: str
    def __init__(self, num: int, *, target: str = "", **kwargs: Any) -> None: ...
    def __call__(self, f: _SignalCallbackT) -> _SignalCallbackT: ...

class timer:
    num: int
    secs: int
    target: str
    def __init__(self, secs: int, *, signum: int = ..., target: str = "", **kwargs: Any) -> None: ...
    def __call__(self, f: _SignalCallbackT) -> _SignalCallbackT: ...

class cron:
    num: int
    minute: int
    hour: int
    day: int
    month: int
    dayweek: int
    target: str
    def __init__(
        self, minute: int, hour: int, day: int, month: int, dayweek: int, *, signum: int = ..., target: str = "", **kwargs: Any
    ) -> None: ...
    def __call__(self, f: _SignalCallbackT) -> _SignalCallbackT: ...

class rbtimer:
    num: int
    secs: int
    target: str
    def __init__(self, secs: int, *, signum: int = ..., target: str = "", **kwargs: Any) -> None: ...
    def __call__(self, f: _SignalCallbackT) -> _SignalCallbackT: ...

class filemon:
    num: int
    fsobj: str
    target: str
    def __init__(self, fsobj: str, *, signum: int = ..., target: str = "", **kwargs: Any) -> None: ...
    def __call__(self, f: _SignalCallbackT) -> _SignalCallbackT: ...

class lock(Generic[_P, _T]):
    f: Callable[_P, _T]
    def __init__(self, f: Callable[_P, _T]) -> None: ...
    def __call__(self, *args: _P.args, **kwargs: _P.kwargs) -> _T: ...

# FIXME: Technically this only allows positional arguments, but there is not really
#        an adequate way yet to express this, once bound on ParamSpec does something
#        we could probably enforce this
class thread(Generic[_P, _T]):
    f: Callable[_P, _T]
    def __init__(self, f: Callable[_P, _T]) -> None: ...
    def __call__(self, *args: _P.args, **kwargs: _P.kwargs) -> Callable[_P, _T]: ...

class harakiri:
    s: int
    def __init__(self, seconds: int) -> None: ...
    def __call__(self, f: Callable[_P, _T]) -> Callable[_P, _T]: ...
