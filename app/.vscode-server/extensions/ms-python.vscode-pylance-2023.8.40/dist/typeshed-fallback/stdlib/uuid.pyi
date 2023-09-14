import builtins
import sys
from _typeshed import Unused
from enum import Enum
from typing_extensions import TypeAlias

_FieldsType: TypeAlias = tuple[int, int, int, int, int, int]

class SafeUUID(Enum):
    safe: int
    unsafe: int
    unknown: None

class UUID:
    def __init__(
        self,
        hex: str | None = None,
        bytes: builtins.bytes | None = None,
        bytes_le: builtins.bytes | None = None,
        fields: _FieldsType | None = None,
        int: builtins.int | None = None,
        version: builtins.int | None = None,
        *,
        is_safe: SafeUUID = ...,
    ) -> None: ...
    @property
    def is_safe(self) -> SafeUUID: ...
    @property
    def bytes(self) -> builtins.bytes: ...
    @property
    def bytes_le(self) -> builtins.bytes: ...
    @property
    def clock_seq(self) -> builtins.int: ...
    @property
    def clock_seq_hi_variant(self) -> builtins.int: ...
    @property
    def clock_seq_low(self) -> builtins.int: ...
    @property
    def fields(self) -> _FieldsType: ...
    @property
    def hex(self) -> str: ...
    @property
    def int(self) -> builtins.int: ...
    @property
    def node(self) -> builtins.int: ...
    @property
    def time(self) -> builtins.int: ...
    @property
    def time_hi_version(self) -> builtins.int: ...
    @property
    def time_low(self) -> builtins.int: ...
    @property
    def time_mid(self) -> builtins.int: ...
    @property
    def urn(self) -> str: ...
    @property
    def variant(self) -> str: ...
    @property
    def version(self) -> builtins.int | None: ...
    def __int__(self) -> builtins.int: ...
    def __eq__(self, other: object) -> bool: ...
    def __lt__(self, other: UUID) -> bool: ...
    def __le__(self, other: UUID) -> bool: ...
    def __gt__(self, other: UUID) -> bool: ...
    def __ge__(self, other: UUID) -> bool: ...
    def __hash__(self) -> builtins.int: ...

if sys.version_info >= (3, 9):
    def getnode() -> int: ...

else:
    def getnode(*, getters: Unused = None) -> int: ...  # undocumented

def uuid1(node: int | None = None, clock_seq: int | None = None) -> UUID: ...

if sys.version_info >= (3, 12):
    def uuid3(namespace: UUID, name: str | bytes) -> UUID: ...

else:
    def uuid3(namespace: UUID, name: str) -> UUID: ...

def uuid4() -> UUID: ...

if sys.version_info >= (3, 12):
    def uuid5(namespace: UUID, name: str | bytes) -> UUID: ...

else:
    def uuid5(namespace: UUID, name: str) -> UUID: ...

NAMESPACE_DNS: UUID
NAMESPACE_URL: UUID
NAMESPACE_OID: UUID
NAMESPACE_X500: UUID
RESERVED_NCS: str
RFC_4122: str
RESERVED_MICROSOFT: str
RESERVED_FUTURE: str

if sys.version_info >= (3, 12):
    def main() -> None: ...
