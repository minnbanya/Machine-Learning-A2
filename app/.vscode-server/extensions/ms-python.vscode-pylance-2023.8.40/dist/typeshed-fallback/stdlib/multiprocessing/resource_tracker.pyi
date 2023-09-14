from _typeshed import FileDescriptorOrPath, Incomplete
from collections.abc import Sized

__all__ = ["ensure_running", "register", "unregister"]

class ResourceTracker:
    def getfd(self) -> int | None: ...
    def ensure_running(self) -> None: ...
    def register(self, name: Sized, rtype: Incomplete) -> None: ...
    def unregister(self, name: Sized, rtype: Incomplete) -> None: ...

_resource_tracker: ResourceTracker
ensure_running = _resource_tracker.ensure_running
register = _resource_tracker.register
unregister = _resource_tracker.unregister
getfd = _resource_tracker.getfd

def main(fd: FileDescriptorOrPath) -> None: ...
