from collections.abc import Iterable, Sequence
from re import Pattern
from typing import ClassVar
from typing_extensions import Literal

AF_LINK: Literal[48]
width: Literal[48]
family: Literal[48]
family_name: Literal["MAC"]
version: Literal[48]
max_int: int

class mac_eui48:
    word_size: ClassVar[int]
    num_words: ClassVar[int]
    max_word: ClassVar[int]
    word_sep: ClassVar[str]
    word_fmt: ClassVar[str]
    word_base: ClassVar[int]

class mac_unix(mac_eui48): ...
class mac_unix_expanded(mac_unix): ...
class mac_cisco(mac_eui48): ...
class mac_bare(mac_eui48): ...
class mac_pgsql(mac_eui48): ...

DEFAULT_DIALECT: type[mac_eui48]
RE_MAC_FORMATS: list[Pattern[str]]

def valid_str(addr: str) -> bool: ...
def str_to_int(addr: str) -> int: ...
def int_to_str(int_val: int, dialect: type[mac_eui48] | None = None) -> str: ...
def int_to_packed(int_val: int) -> bytes: ...
def packed_to_int(packed_int: bytes) -> int: ...
def valid_words(words: Iterable[int], dialect: type[mac_eui48] | None = None) -> bool: ...
def int_to_words(int_val: int, dialect: type[mac_eui48] | None = None) -> tuple[int, ...]: ...
def words_to_int(words: Sequence[int], dialect: type[mac_eui48] | None = None) -> int: ...
def valid_bits(bits: str, dialect: type[mac_eui48] | None = None) -> bool: ...
def bits_to_int(bits: str, dialect: type[mac_eui48] | None = None) -> int: ...
def int_to_bits(int_val: int, dialect: type[mac_eui48] | None = None) -> str: ...
def valid_bin(bin_val: str, dialect: type[mac_eui48] | None = None) -> bool: ...
def int_to_bin(int_val: int) -> str: ...
def bin_to_int(bin_val: str) -> int: ...
