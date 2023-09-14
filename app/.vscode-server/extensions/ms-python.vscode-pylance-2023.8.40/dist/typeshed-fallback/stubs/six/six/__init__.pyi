import builtins
import operator
import types
import unittest
from _typeshed import IdentityFunction, Unused, _KT_contra, _VT_co
from builtins import next as next
from collections.abc import Callable, ItemsView, Iterable, Iterator as _Iterator, KeysView, Mapping, ValuesView
from functools import wraps as wraps
from importlib.util import spec_from_loader as spec_from_loader
from io import BytesIO as BytesIO, StringIO as StringIO
from re import Pattern
from typing import Any, AnyStr, NoReturn, Protocol, TypeVar, overload
from typing_extensions import Literal

from six import moves as moves

# TODO: We should switch to the _typeshed version of SupportsGetItem
# once mypy updates its vendored copy of typeshed and makes a new release
class _SupportsGetItem(Protocol[_KT_contra, _VT_co]):
    def __contains__(self, __x: Any) -> bool: ...
    def __getitem__(self, __key: _KT_contra) -> _VT_co: ...

_T = TypeVar("_T")
_K = TypeVar("_K")
_V = TypeVar("_V")

__author__: str
__version__: str

PY2: Literal[False]
PY3: Literal[True]
PY34: Literal[True]

string_types: tuple[type[str]]
integer_types: tuple[type[int]]
class_types: tuple[type[type]]
text_type = str
binary_type = bytes

MAXSIZE: int

callable = builtins.callable

def get_unbound_function(unbound: types.FunctionType) -> types.FunctionType: ...

create_bound_method = types.MethodType

def create_unbound_method(func: types.FunctionType, cls: type) -> types.FunctionType: ...

Iterator = object

def get_method_function(meth: types.MethodType) -> types.FunctionType: ...
def get_method_self(meth: types.MethodType) -> object: ...
def get_function_closure(fun: types.FunctionType) -> tuple[types._Cell, ...] | None: ...
def get_function_code(fun: types.FunctionType) -> types.CodeType: ...
def get_function_defaults(fun: types.FunctionType) -> tuple[Any, ...] | None: ...
def get_function_globals(fun: types.FunctionType) -> dict[str, Any]: ...
def iterkeys(d: Mapping[_K, Any]) -> _Iterator[_K]: ...
def itervalues(d: Mapping[Any, _V]) -> _Iterator[_V]: ...
def iteritems(d: Mapping[_K, _V]) -> _Iterator[tuple[_K, _V]]: ...
def viewkeys(d: Mapping[_K, Any]) -> KeysView[_K]: ...
def viewvalues(d: Mapping[Any, _V]) -> ValuesView[_V]: ...
def viewitems(d: Mapping[_K, _V]) -> ItemsView[_K, _V]: ...
def b(s: str) -> bytes: ...
def u(s: str) -> str: ...

unichr = chr

def int2byte(i: int) -> bytes: ...

# Should be `byte2int: operator.itemgetter[int]`. But a bug in mypy prevents using TypeVar in itemgetter.__call__
def byte2int(obj: _SupportsGetItem[int, _T]) -> _T: ...

indexbytes = operator.getitem
iterbytes = iter

def assertCountEqual(self: unittest.TestCase, first: Iterable[_T], second: Iterable[_T], msg: str | None = ...) -> None: ...
@overload
def assertRaisesRegex(self: unittest.TestCase, msg: str | None = ...) -> Any: ...
@overload
def assertRaisesRegex(self: unittest.TestCase, callable_obj: Callable[..., object], *args: Any, **kwargs: Any) -> Any: ...
def assertRegex(self: unittest.TestCase, text: AnyStr, expected_regex: AnyStr | Pattern[AnyStr], msg: Any = ...) -> None: ...
def assertNotRegex(self: unittest.TestCase, text: AnyStr, expected_regex: AnyStr | Pattern[AnyStr], msg: Any = ...) -> None: ...

exec_ = exec

def reraise(tp: type[BaseException] | None, value: BaseException | None, tb: types.TracebackType | None = None) -> NoReturn: ...
def raise_from(value: BaseException | type[BaseException], from_value: BaseException | None) -> NoReturn: ...

print_ = print

def with_metaclass(meta: type, *bases: type) -> type: ...
def add_metaclass(metaclass: type) -> IdentityFunction: ...
def ensure_binary(s: bytes | str, encoding: str = "utf-8", errors: str = "strict") -> bytes: ...
def ensure_str(s: bytes | str, encoding: str = "utf-8", errors: str = "strict") -> str: ...
def ensure_text(s: bytes | str, encoding: str = "utf-8", errors: str = "strict") -> str: ...
def python_2_unicode_compatible(klass: _T) -> _T: ...

class _LazyDescr:
    name: str
    def __init__(self, name: str) -> None: ...
    def __get__(self, obj: object, tp: Unused) -> Any: ...

class MovedModule(_LazyDescr):
    mod: str
    def __init__(self, name: str, old: str, new: str | None = None) -> None: ...
    def __getattr__(self, attr: str) -> Any: ...

class MovedAttribute(_LazyDescr):
    mod: str
    attr: str
    def __init__(
        self, name: str, old_mod: str, new_mod: str, old_attr: str | None = None, new_attr: str | None = None
    ) -> None: ...

def add_move(move: MovedModule | MovedAttribute) -> None: ...
def remove_move(name: str) -> None: ...
