from collections.abc import Mapping
from typing import Any

from markdown.core import Markdown

class Extension:
    config: Mapping[str, list[Any]]
    def __init__(self, **kwargs: Any) -> None: ...
    def getConfig(self, key: str, default: Any = "") -> Any: ...
    def getConfigs(self) -> dict[str, Any]: ...
    def getConfigInfo(self) -> list[tuple[str, str]]: ...
    def setConfig(self, key: str, value: Any) -> None: ...
    def setConfigs(self, items: Mapping[str, Any]) -> None: ...
    def extendMarkdown(self, md: Markdown) -> None: ...
