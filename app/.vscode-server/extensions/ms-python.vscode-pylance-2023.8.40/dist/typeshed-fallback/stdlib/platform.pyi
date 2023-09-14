import sys

if sys.version_info < (3, 8):
    import os

    DEV_NULL = os.devnull
from typing import NamedTuple

if sys.version_info >= (3, 8):
    def libc_ver(executable: str | None = None, lib: str = "", version: str = "", chunksize: int = 16384) -> tuple[str, str]: ...

else:
    def libc_ver(
        executable: str = sys.executable, lib: str = "", version: str = "", chunksize: int = 16384
    ) -> tuple[str, str]: ...

if sys.version_info < (3, 8):
    def linux_distribution(
        distname: str = "",
        version: str = "",
        id: str = "",
        supported_dists: tuple[str, ...] = ...,
        full_distribution_name: bool = ...,
    ) -> tuple[str, str, str]: ...
    def dist(
        distname: str = "", version: str = "", id: str = "", supported_dists: tuple[str, ...] = ...
    ) -> tuple[str, str, str]: ...

def win32_ver(release: str = "", version: str = "", csd: str = "", ptype: str = "") -> tuple[str, str, str, str]: ...

if sys.version_info >= (3, 8):
    def win32_edition() -> str: ...
    def win32_is_iot() -> bool: ...

def mac_ver(
    release: str = "", versioninfo: tuple[str, str, str] = ("", "", ""), machine: str = ""
) -> tuple[str, tuple[str, str, str], str]: ...
def java_ver(
    release: str = "", vendor: str = "", vminfo: tuple[str, str, str] = ("", "", ""), osinfo: tuple[str, str, str] = ("", "", "")
) -> tuple[str, str, tuple[str, str, str], tuple[str, str, str]]: ...
def system_alias(system: str, release: str, version: str) -> tuple[str, str, str]: ...
def architecture(executable: str = sys.executable, bits: str = "", linkage: str = "") -> tuple[str, str]: ...

class uname_result(NamedTuple):
    system: str
    node: str
    release: str
    version: str
    machine: str
    processor: str

def uname() -> uname_result: ...
def system() -> str: ...
def node() -> str: ...
def release() -> str: ...
def version() -> str: ...
def machine() -> str: ...
def processor() -> str: ...
def python_implementation() -> str: ...
def python_version() -> str: ...
def python_version_tuple() -> tuple[str, str, str]: ...
def python_branch() -> str: ...
def python_revision() -> str: ...
def python_build() -> tuple[str, str]: ...
def python_compiler() -> str: ...
def platform(aliased: bool = ..., terse: bool = ...) -> str: ...

if sys.version_info >= (3, 10):
    def freedesktop_os_release() -> dict[str, str]: ...
