class FrozenList(list):
    def union(self, other) -> FrozenList: ...
    def difference(self, other) -> FrozenList: ...
    def __getitem__(self, n): ...
    def __radd__(self, other): ...
    def __eq__(self, other) -> bool: ...
    def __mul__(self, other): ...
    def __reduce__(self): ...
    def __hash__(self) -> int: ...  # type: ignore[override]
